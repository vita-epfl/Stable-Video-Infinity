import types
from ..models import ModelManager
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import torch.nn as nn
from ..models.wan_video_dit import WanModel
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from transformers import Wav2Vec2FeatureExtractor
from einops import rearrange
from utils.audio_process import get_embedding, audio_prepare_single
from utils.src.audio_analysis.wav2vec2 import Wav2Vec2Model

def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states

def model_fn_wan_talk_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    add_condition = None,
    audio_embed_tuple = None,
    use_unified_sequence_parallel: bool = False,
    use_controlnet: bool = False,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(dtype=x.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = dit.patchify(x)
    
    if add_condition is not None:
        x = add_condition + x
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    grid_size = (f, h, w)
    audio_embed = dit.audio_proj(audio_embed_tuple[0].to(x.dtype).to(x.device), audio_embed_tuple[1].to(x.dtype).to(x.device)) 
    audio_embed = torch.concat(audio_embed.split(1), dim=2).to(x.dtype)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]


    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        # blocks
        for block in dit.blocks:
            if use_controlnet:
                x, add_condition = block(x, context, t_mod, freqs, grid_size=grid_size, audio_embed=audio_embed, add_condition=add_condition)
            else:
                x = block(x, context, t_mod, freqs, grid_size=grid_size, audio_embed=audio_embed)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x

class SVITalkVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None, wav2vec_path='./weights/chinese-wav2vec2-base/', is_test=False):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.is_test = is_test

        if is_test:
            self.wav2vec_feature_extractor, self.audio_encoder= custom_init('cpu', wav2vec_path)    

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

        if self.is_test: 
            self.dit.blocks.to(self.device)
            self.vae.to(self.device)
            self.dit.blocks.eval()

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False, is_test=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = SVITalkVideoPipeline(device=device, torch_dtype=torch_dtype, is_test=is_test)
        pipe.fetch_models(model_manager)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True

        return pipe
    
    
    def denoising_model(self):
        return self.dit
    

    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}
    

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    

    def encode_images_adaptive(self, first_frames, random_ref_frame, num_frames, height, width, use_first_aug=False, ref_pad_cfg=False, ref_pad_num=None):
        '''
        first_frames: List of Images
        random_ref_frame: 1 Image
        '''
        # Save original dtype for final conversion
        original_dtype = self.torch_dtype
        
        # Temporarily set models to float32 for precise computation
        original_vae_dtype = None
        original_image_encoder_dtype = None
        
        if hasattr(self, 'vae') and self.vae is not None:
            original_vae_dtype = next(self.vae.parameters()).dtype
            self.vae = self.vae.to(dtype=torch.float32)
        
        if hasattr(self, 'image_encoder') and self.image_encoder is not None:
            original_image_encoder_dtype = next(self.image_encoder.parameters()).dtype
            self.image_encoder = self.image_encoder.to(dtype=torch.float32)

        num_condition_frames = len(first_frames)
        remaining_frames = num_frames - num_condition_frames

        # Use float32 for all computations
        random_ref_frame = self.preprocess_image(random_ref_frame.resize((width, height))).to(device=self.device, dtype=torch.float32)
        first_frame = self.preprocess_image(first_frames[0].resize((width, height))).to(device=self.device, dtype=torch.float32)
        clip_context = self.image_encoder.encode_image([first_frame])

        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device, dtype=torch.float32)
        if ref_pad_cfg:
            msk[:, len(first_frames):] = 0
        else:
            msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        if len(first_frames) > 1:
            first_frame_list = []
            for frame in first_frames:
                first_frame_list.append(self.preprocess_image(frame.resize((width, height)), use_aug=use_first_aug).to(device=self.device, dtype=torch.float32))
            vae_input_condition = torch.cat(first_frame_list, dim=0).permute(1,0,2,3)  # (3, num_video_frames, H, W)
        else:
            vae_input_condition = self.preprocess_image(first_frames[0].resize((width, height)), use_aug=use_first_aug).to(device=self.device, dtype=torch.float32).transpose(0, 1)  # (3, 1, H, W)

        if ref_pad_num == 0:
            vae_input_pad = torch.zeros(3, remaining_frames, height, width, 
                                                device=self.device, dtype=torch.float32)
        elif ref_pad_num > 0 and ref_pad_num != -1:
            pad_imgs = []
            for i in range(ref_pad_num):
                pad_imgs.append(random_ref_frame.transpose(0, 1))
            if remaining_frames > ref_pad_num:
                pad_imgs += [torch.zeros(3, 1, height, width, device=self.device, dtype=torch.float32)] * (remaining_frames - ref_pad_num)
            vae_input_pad = torch.cat(pad_imgs, dim=1)
        elif ref_pad_num == -1:
            vae_input_pad = random_ref_frame.transpose(0, 1).repeat(1, remaining_frames, 1, 1)
        
        vae_input = torch.concat([vae_input_condition, vae_input_pad], dim=1)  # (3, num_frames, H, W)
        y = self.vae.encode([vae_input.to(dtype=torch.float32, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)

        # Convert results to original dtype (bf16) before returning
        clip_context = clip_context.to(dtype=original_dtype, device=self.device)
        y = y.to(dtype=original_dtype, device=self.device)
        
        # Restore original model dtypes
        if original_vae_dtype is not None:
            self.vae = self.vae.to(dtype=original_vae_dtype)
        if original_image_encoder_dtype is not None:
            self.image_encoder = self.image_encoder.to(dtype=original_image_encoder_dtype)

        return {"clip_feature": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        self.vae = self.vae.to(device=self.device, dtype=torch.float32)
        input_video = input_video.to(device=self.device, dtype=torch.float32)
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents = latents.to(device=self.device, dtype=self.torch_dtype)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        self.vae = self.vae.to(device=self.device, dtype=torch.float32)
        latents = latents.to(device=self.device, dtype=torch.float32)
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def get_audio_embedding(self, audio_path, num_frames, audio_start_idx=0):
            human_speech = audio_prepare_single(audio_path) 
            audio_embed = get_embedding(human_speech, self.wav2vec_feature_extractor, self.audio_encoder)
            # audio_embedding_list.append(audio_embedding)
            # frame_num = (f-1) * 4 + 1
            indices = (torch.arange(2 * 2 + 1) - 2) * 1 
            clip_length = num_frames
            # audio_start_idx = 0
            audio_end_idx = audio_start_idx + clip_length
            center_indices = torch.arange(
                audio_start_idx,
                audio_end_idx,
                1,
            ).unsqueeze(
                1
            ) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=audio_embed.shape[0]-1)
            audio_embed = audio_embed[center_indices][None,...]
            return audio_embed

    def preprocess_audio(self, audio_embed, audio_window=5, vae_scale=4):
        first_frame_audio_emb_s = audio_embed[:, :1, ...] 
        latter_frame_audio_emb = audio_embed[:, 1:, ...] 
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=vae_scale) 
        middle_index = audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 
        audio_embed_tuple = (first_frame_audio_emb_s.to(self.device).to(torch.bfloat16) , latter_frame_audio_emb_s.to(self.device).to(torch.bfloat16) )

        return audio_embed_tuple  
    
    def _sample_with_multitalk(self, latents, prompt_emb_posi, prompt_emb_nega, image_emb, extra_input, tea_cache_posi, tea_cache_nega, usp_kwargs, condition, audio_embed_tuple, audio_embed_tuple_null, use_controlnet, cfg_scale, progress_bar_cmd):
        """
        Sampling function for MultiTalk mode
        """
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(device=self.device)
            # Inference
            noise_pred_cond = model_fn_wan_talk_video(self.dit, latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input, **tea_cache_posi, **usp_kwargs, add_condition=condition, audio_embed_tuple=audio_embed_tuple, use_controlnet=use_controlnet)
            if cfg_scale["text"] != 1.0 or cfg_scale["audio"] != 1.0:
                noise_pred_uncond = model_fn_wan_talk_video(self.dit, latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input, **tea_cache_nega, **usp_kwargs, audio_embed_tuple=audio_embed_tuple_null, use_controlnet=use_controlnet)
                noise_pred_drop_text = model_fn_wan_talk_video(self.dit, latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input, **tea_cache_nega, **usp_kwargs, add_condition=condition, audio_embed_tuple=audio_embed_tuple, use_controlnet=use_controlnet)

                noise_pred = noise_pred_uncond + cfg_scale["text"] * (
                    noise_pred_cond - noise_pred_drop_text) + \
                    cfg_scale["audio"] * (noise_pred_drop_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        random_ref_frame=None,
        audio_path=None,
        use_controlnet=False,
        audio_start_idx=0,
        args=None,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            random_ref_img = random_ref_frame.clone()
            random_ref_img = Image.fromarray(random_ref_img.cpu().numpy())
            if not isinstance(input_image, list):
                input_image = [input_image]
            image_emb = self.encode_images_adaptive(input_image, random_ref_img, num_frames, height, width, use_first_aug=False, ref_pad_cfg=args.ref_pad_cfg, ref_pad_num=args.ref_pad_num)

        else:
            image_emb = {} 
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(["dit"])
        usp_kwargs = self.prepare_unified_sequence_parallel()

        audio_embed = self.get_audio_embedding(audio_path, num_frames, audio_start_idx=audio_start_idx)
        audio_embed_tuple = self.preprocess_audio(audio_embed, audio_window=5, vae_scale=4)
        audio_embed_tuple_null = self.preprocess_audio(torch.zeros_like(audio_embed)[-1:], audio_window=5, vae_scale=4)

        latents = self._sample_with_multitalk(
            latents, prompt_emb_posi, prompt_emb_nega, image_emb, extra_input, 
            tea_cache_posi, tea_cache_nega, usp_kwargs, None, 
            audio_embed_tuple, audio_embed_tuple_null, use_controlnet, cfg_scale, progress_bar_cmd
        )

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
