import torch
import os
import imageio
import argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
from diffsynth import SVIDanceVideoPipeline, ModelManager, load_state_dict, load_state_dict_from_folder
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import random
import pickle
from io import BytesIO
import torch.nn.functional as F
import torch.nn as nn
import cv2
from utils.project_utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--exp_prefix",
        default='',
        type=str,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='data/toy_train/svi-dance/preprocessed/',
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="/mnt/data/hnqiu/wanx2.1_t2v/WanX2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="/mnt/data/hnqiu/wanx2.1_t2v/WanX2.1-T2V-14B/WanX2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=True,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=832,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use for distributed training.",
    )
    parser.add_argument(
        "--use_first_aug",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_error_recycling",
        default=False,
        action="store_true",
        help="Whether to enable error recycling for image_emb['y'].",
    )
    parser.add_argument(
        "--add_error",
        type=str,
        default="both",
        choices=["none", "y", "latent", "both"],
        help="Type of augmentation to use.",
    )
    parser.add_argument(
        "--error_buffer_k",
        type=int,
        default=500,
        help="Maximum number of error samples to store per timestep grid.",
    )
    parser.add_argument(
        "--timestep_grid_size",
        type=int,
        default=25,
        help="Size of timestep grid for buffer organization. Timesteps will be grouped into grids of this size.",
    )
    parser.add_argument(
        "--num_grids",
        type=int,
        default=50,
        help="Size of timestep grid for buffer organization. Timesteps will be grouped into grids of this size.",
    )
    parser.add_argument(
        "--buffer_replacement_strategy",
        type=str,
        default="random",
        choices=["random", "l2_similarity", "l2_batch", "fifo"],
        help="Strategy for replacing samples when buffer is full. 'random': random replacement (fastest), 'l2_similarity': replace most similar (slowest but best quality), 'l2_batch': batch L2 computation (balanced), 'fifo': first-in-first-out.",
    )
    parser.add_argument(
        "--buffer_warmup_iter",
        type=int,
        default=50,
        help="Number of warmup iterations. During warmup, all GPUs will update the error buffers. After warmup, only local GPU updates the buffer.",
    )
    parser.add_argument(
        "--error_modulate_factor",
        default=0.0,
        type=float,
        help="Factor to modulate error intensity.",
    )
    parser.add_argument(
        "--ref_pad_num",
        type=int,
        default=-1,  # 0 -> no padding k-> padding k ,-1 -> full padding
        help="Number of reference frames to pad with",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping. Set to 0 to disable gradient clipping.",
    )
    parser.add_argument(
        "--num_motion_frames",
        type=int,
        default=1,
        help="Number of non-padded latents to apply augmentation on.",
    )
    parser.add_argument(
        "--y_error_num",
        type=int,
        default=1,
        help="Number of non-padded latents to apply augmentation on.",
    )
    parser.add_argument(
        "--y_error_sample_from_all_grids",
        default=False,
        action="store_true",
        help="Whether to sample y_error from all timestep grids instead of just the current grid.",
    )
    parser.add_argument(
        "--y_error_sample_range",
        type=str,
        default=None,
        help="Custom timestep range for y_error sampling in format 'start,end' (e.g., '0,50'). If not specified, uses default behavior.",
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.01,
        help="Probability threshold for noise error in error settings (range 0-1).",
    )
    parser.add_argument(
        "--y_prob",
        type=float,
        default=0.99,
        help="Probability threshold for y error in error settings (range 0-1).",
    )
    parser.add_argument(
        "--latent_prob",
        type=float,
        default=0.99,
        help="Probability threshold for latent error in error settings (range 0-1).",
    )    
    parser.add_argument(
        "--clean_prob",
        type=float,
        default=0.1,
        help="Probability threshold for latent error in error settings (range 0-1).",
    ) 
    parser.add_argument(
        "--use_rand_first_frame",
        action="store_true",
        default=False,
        help="Whether to use random frames as first_ref_frames instead of true first frames.",
    )
    parser.add_argument(
        "--remove_pose",
        action="store_true",
        default=False,
        help="Whether to remove pose information from the input.",
    )
    parser.add_argument(
        "--clean_buffer_update_prob",
        type=float,
        default=0.1,
        help="Probability threshold for latent error in error settings (range 0-1).",
    )  
    args = parser.parse_args()
    return args

class TextVideoDataset_onestage(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False, steps_per_epoch=1, args=None):
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.steps_per_epoch = steps_per_epoch
        self.args = args
        self.misc_size = [height, width]
        self.video_list = []

        # data_list = ['TikTok']
        self.sample_fps = frame_interval
        self.max_frames = max_num_frames
        self.use_rand_first_frame = args.use_rand_first_frame
        

        self.pose_dir = base_path
        file_list = os.listdir(self.pose_dir)
        print("!!! all dataset length: ", len(file_list))
        for iii_index in file_list:
            self.video_list.append(self.pose_dir + iii_index)
        
        random.shuffle(self.video_list)
        
        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        width, height = image.size
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_ref_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_ref_frame is None:
                first_ref_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_ref_frame
        else:
            return frames

    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    def load_video_and_pose(self, video_path, path_dir):
        """Load video frames and corresponding dwpose from mp4 files"""
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        stride = random.randint(1, self.sample_fps)
        cover_frame_num = stride * self.max_frames
        
        if total_frames < cover_frame_num + 1:
            start_frame = 0
            end_frame = total_frames - 1
            stride = max((total_frames // self.max_frames), 1)
            end_frame = min(stride * self.max_frames, total_frames - 1)
        else:
            max_start_frame = max(0, total_frames - cover_frame_num - 5)
            start_frame = random.randint(0, max_start_frame) if max_start_frame > 0 else 0
            end_frame = start_frame + cover_frame_num
        
        frame_list = []
        dwpose_list = []
        
        # Load dwpose video if exists
        dwpose_video_path = os.path.join(path_dir, 'dwpose.mp4')
        dwpose_cap = None
        if os.path.exists(dwpose_video_path):
            dwpose_cap = cv2.VideoCapture(dwpose_video_path)
        
        for i in range(start_frame, end_frame, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_list.append(frame)
                
                # Load corresponding dwpose frame
                if dwpose_cap is not None:
                    dwpose_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret_dwpose, dwpose_frame = dwpose_cap.read()
                    if ret_dwpose:
                        dwpose_frame = cv2.cvtColor(dwpose_frame, cv2.COLOR_BGR2RGB)
                        dwpose_frame = Image.fromarray(dwpose_frame)
                        dwpose_list.append(dwpose_frame)
                    else:
                        # Create a dummy dwpose frame if not available
                        dwpose_list.append(Image.new('RGB', frame.size, (0, 0, 0)))
                else:
                    # Create a dummy dwpose frame if dwpose video doesn't exist
                    dwpose_list.append(Image.new('RGB', frame.size, (0, 0, 0)))
        
        cap.release()
        if dwpose_cap is not None:
            dwpose_cap.release()
            
        return frame_list, dwpose_list
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, index):
        index = index % len(self.video_list)
        path_dir = self.video_list[index]

        # Load video frames directly from mp4
        video_path = os.path.join(path_dir, 'video.mp4')
        if not os.path.exists(video_path):
            # Fallback to old pkl format if mp4 doesn't exist
            frames_all = pickle.load(open(path_dir + '/frame_data.pkl', 'rb'))
            dwpose_all = pickle.load(open(path_dir + '/dw_pose_with_foot_wo_face.pkl', 'rb'))

            stride = random.randint(1, self.sample_fps)
            
            _total_frame_num = len(frames_all)
            cover_frame_num = (stride * self.max_frames)
            max_frames = self.max_frames
            if _total_frame_num < cover_frame_num + 1:
                start_frame = 0
                end_frame = _total_frame_num - 1
                stride = max((_total_frame_num // max_frames), 1)
                end_frame = min(stride * max_frames, _total_frame_num - 1)
            else:
                max_start_frame = max(0, _total_frame_num - cover_frame_num - 5)
                start_frame = random.randint(0, max_start_frame) if max_start_frame > 0 else 0
                end_frame = start_frame + cover_frame_num

            frame_list = []
            dwpose_list = []

            for i_index in range(start_frame, end_frame, stride):
                i_key = list(frames_all.keys())[i_index]
                i_frame = Image.open(BytesIO(frames_all[i_key]))
                if i_frame.mode != 'RGB':
                    i_frame = i_frame.convert('RGB')
                i_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                frame_list.append(i_frame)
                dwpose_list.append(i_dwpose)
        else:
            # Load video frames directly from mp4
            frame_list, dwpose_list = self.load_video_and_pose(video_path, path_dir)
            
        num_ref_frames = 12
        if self.use_rand_first_frame:
            ref_frame_indices = [random.randint(0, len(frame_list) - 1) for _ in range(min(num_ref_frames, len(frame_list)))]
        else:
            ref_frame_indices = list(range(min(num_ref_frames, len(frame_list))))

        first_ref_frames = [frame_list[idx].copy() for idx in ref_frame_indices]
        
        random_frame_idx = random.randint(0, len(frame_list) - 1)
        random_ref_frame = frame_list[random_frame_idx].copy()
        
        first_ref_frame = first_ref_frames[0]
        if os.path.exists(video_path):
            # For mp4 format, use the first dwpose frame as reference
            first_ref_dwpose = dwpose_list[0]
        else:
            # For pkl format
            i_key = list(frames_all.keys())[start_frame]
            first_ref_dwpose = Image.open(BytesIO(dwpose_all[i_key]))

        max_frames = self.max_frames
        if len(frame_list) < max_frames:
            for _ in range(max_frames - len(frame_list)):
                frame_list.append(frame_list[-1].copy())
                dwpose_list.append(dwpose_list[-1].copy())

        have_frames = len(frame_list) > 0
        middle_indix = 0

        if have_frames:

            target_aspect_ratio = self.misc_size[0] / self.misc_size[1]   # height / width

            l_height = first_ref_frames[0].size[1]  
            l_width = first_ref_frames[0].size[0]

            if l_width * target_aspect_ratio <= l_height:
                # Width is the limiting factor, crop height
                crop_width = random.randint(l_width - l_width//14, l_width)
                crop_height = int(crop_width * target_aspect_ratio)
            else:
                # Height is the limiting factor, crop width
                crop_height = random.randint(l_height - l_height//14, l_height)
                crop_width = int(crop_height / target_aspect_ratio)

            first_ref_frames = [frame for frame in first_ref_frames]
            random_ref_frame = random_ref_frame
            first_ref_dwpose = first_ref_dwpose

            first_ref_frames_tmp = [torch.from_numpy(np.array(self.resize(frame))) for frame in first_ref_frames]
            first_ref_frame_tmp = first_ref_frames_tmp[0] 
            random_ref_frame_tmp = torch.from_numpy(np.array(self.resize(random_ref_frame)))
            first_ref_dwpose_tmp = torch.from_numpy(np.array(self.resize(first_ref_dwpose))) # [3, 512, 320]

            video_data_tmp = torch.stack([self.frame_process(self.resize(ss)) for ss in frame_list], dim=0) # self.transforms(frames)
            dwpose_data_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss))).permute(2, 0, 1) for ss in dwpose_list], dim=0)

        video_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
        dwpose_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
        
        if have_frames:
            video_data[:len(frame_list), ...] = video_data_tmp      
            dwpose_data[:len(frame_list), ...] = dwpose_data_tmp
            
        video_data = video_data.permute(1, 0, 2, 3)
        dwpose_data = dwpose_data.permute(1, 0, 2, 3)

        # dwpose_data 
        # [3, 81, 480, 640]
                
        text = 'the person is dancing' 
        path = path_dir 
        if self.is_i2v:
            video, first_ref_frames, first_ref_frame, random_ref_frame = video_data, first_ref_frames_tmp, first_ref_frame_tmp, random_ref_frame_tmp

            data = {"text": text, "video": video, "path": path, "first_ref_frames": first_ref_frames, "first_ref_frame": first_ref_frame, "random_ref_frame": random_ref_frame, "dwpose_data": dwpose_data, "random_ref_data": first_ref_dwpose_tmp}

        else:
            data = {"text": text, "video": video, "path": path}
        return data
    
    def __len__(self):
        return len(self.video_list)
 

class LightningModelForTrain_onestage(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        model_VAE=None,
        args=None,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu", train_architecture=args.train_architecture)
        self.use_first_aug = args.use_first_aug
        self.add_error = args.add_error
        self.use_error_recycling = args.use_error_recycling if args else False
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])

        self.pipe = SVIDanceVideoPipeline.from_model_manager(model_manager, is_test=False)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.pipe_VAE = model_VAE.pipe.eval()
        self.tiler_kwargs = model_VAE.tiler_kwargs
        self.ref_pad_num = args.ref_pad_num
        self.y_error_num = args.y_error_num
        concat_dim = 4
        self.dwpose_embedding = nn.Sequential(
                    nn.Conv3d(3, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, 5120, (1, 2, 2), stride=(1, 2, 2), padding=0))

        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        elif train_architecture == "full":
            self.pipe.denoising_model().requires_grad_(True)
        else:
            raise ValueError(f"Unsupported training architecture: {train_architecture}")

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # Gradient clipping and monitoring settings
        self.gradient_clip_val = getattr(args, 'gradient_clip_val', 1.0)
        
        # Initialize noise buffer for storing individual noise samples with timestep grid
        self.error_buffer_size = getattr(args, 'error_buffer_k', 500) # Size per timestep grid
        self.buffer_replacement_strategy = getattr(args, 'buffer_replacement_strategy', 'fifo')
        self.buffer_warmup_iter = getattr(args, 'buffer_warmup_iter', 50)
        self.timestep_grid_size = getattr(args, 'timestep_grid_size', 25)  # Default grid size of 25
        
        # Initialize buffers as dictionaries with timestep grid as keys
        # Each grid will store samples for timesteps in that range
        num_grids = getattr(args, 'num_grids', 40)
        self.inferece_timesteps = self.pipe.scheduler.get_timesteps(num_inference_steps=num_grids, denoising_strength=1, shift=5.0)
        self.latent_error_buffer = {i: [] for i in range(num_grids)}
        self.y_error_buffer = {i: [] for i in range(num_grids)}
        
        self.iteration_count = 0
        self.error_modulate_factor = args.error_modulate_factor
        self.num_motion_frames = args.num_motion_frames
        self.p_motion_threshold = getattr(args, 'p_motion_threshold', 0.5)
        self.y_error_sample_from_all_grids = getattr(args, 'y_error_sample_from_all_grids', False)
        self.noise_prob = getattr(args, 'noise_prob', 0.99)
        self.y_prob = getattr(args, 'y_prob', 0.99)
        self.latent_prob = getattr(args, 'latent_prob', 0.99)
        self.remove_pose = getattr(args, 'remove_pose', False)
        self.clean_prob = getattr(args, 'clean_prob', 0.1)
        self.clean_buffer_update_prob = getattr(args, 'clean_buffer_update_prob', 0.5)
        # Parse y_error_sample_range
        self.y_error_sample_range = None
        if hasattr(args, 'y_error_sample_range') and args.y_error_sample_range:
            try:
                range_parts = args.y_error_sample_range.split(',')
                if len(range_parts) == 2:
                    start_ts, end_ts = int(range_parts[0]), int(range_parts[1])
                    # Convert timesteps to grid indices
                    start_grid = start_ts // self.timestep_grid_size
                    end_grid = end_ts // self.timestep_grid_size
                    self.y_error_sample_range = (start_grid, end_grid)
                    print(f"Y-error sampling range set to timesteps {start_ts}-{end_ts} (grids {start_grid}-{end_grid})")
                else:
                    print("Warning: Invalid y_error_sample_range format. Expected 'start,end'")
            except ValueError:
                print("Warning: Invalid y_error_sample_range values. Expected integers.")

    def _get_timestep_grid(self, timestep):
        """Get the grid index for a given timestep."""
        # Handle different timestep formats (scalar tensor, tensor with batch dim, etc.)
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                # Single timestep value
                timestep_val = timestep.item()
            else:
                # Tensor with batch dimension, take the first element
                timestep_val = timestep.flatten()[0].item()
        else:
            # Already a scalar value
            timestep_val = timestep
        

        timestep_val = max(0, min(timestep_val, 999))  # Clamp to [0, 999]
        grid_idx = torch.argmin((self.inferece_timesteps - timestep_val).abs()).item()

        
        # Ensure grid index is within valid range
        max_grid_idx = len(self.latent_error_buffer) - 1
        grid_idx = min(grid_idx, max_grid_idx)
        
        return grid_idx

    def _compute_l2_distance_batch(self, new_tensor, stored_tensors):
        """Compute L2 distances between new tensor and all stored tensors efficiently."""
        if not stored_tensors:
            return torch.tensor([])
        
        # Stack all stored tensors for batch computation
        stored_stack = torch.stack(stored_tensors)  # [num_stored, ...]
        new_flat = new_tensor.flatten()
        stored_flat = stored_stack.flatten(start_dim=1)  # [num_stored, flattened_size]
        
        # Compute L2 distances in batch
        distances = torch.norm(stored_flat - new_flat.unsqueeze(0), p=2, dim=1)
        return distances
    
    def _compute_l2_distance(self, tensor1, tensor2):
        """Compute L2 distance between two tensors"""
        # Flatten tensors
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        # Compute L2 distance (Euclidean distance)
        l2_distance = torch.norm(flat1 - flat2, p=2)
        return l2_distance.item()

    def _add_error_to_latent_buffer(self, error_sample, timestep):
        """Add error sample to buffer using specified replacement strategy based on timestep grid."""
        grid_idx = self._get_timestep_grid(timestep)
        error_cpu = error_sample.detach().cpu()
        
        if len(self.latent_error_buffer[grid_idx]) < self.error_buffer_size:
            # Buffer not full, simply add
            self.latent_error_buffer[grid_idx].append(error_cpu)
        else:
            # Buffer full, use specified replacement strategy
            if self.buffer_replacement_strategy == "random":
                # Random replacement - O(1), fastest
                replace_idx = random.randint(0, len(self.latent_error_buffer[grid_idx]) - 1)
                self.latent_error_buffer[grid_idx][replace_idx] = error_cpu
                
            elif self.buffer_replacement_strategy == "fifo":
                # First-in-first-out - O(1), simple queue behavior
                self.latent_error_buffer[grid_idx].pop(0)
                self.latent_error_buffer[grid_idx].append(error_cpu)
                
            elif self.buffer_replacement_strategy == "l2_batch":
                # Batch L2 computation - O(n) but vectorized, much faster than original
                distances = self._compute_l2_distance_batch(error_cpu, self.latent_error_buffer[grid_idx])
                most_similar_idx = torch.argmin(distances).item()
                self.latent_error_buffer[grid_idx][most_similar_idx] = error_cpu
                
            elif self.buffer_replacement_strategy == "l2_similarity":
                # Original L2 similarity method - O(n), slowest but most precise
                min_distance = float('inf')
                most_similar_idx = -1
                
                for i, stored_error in enumerate(self.latent_error_buffer[grid_idx]):
                    distance = self._compute_l2_distance(error_cpu, stored_error)
                    if distance < min_distance:
                        min_distance = distance
                        most_similar_idx = i
                
                if most_similar_idx != -1:
                    self.latent_error_buffer[grid_idx][most_similar_idx] = error_cpu

    def _add_error_to_y_buffer(self, error_sample, timestep):
        """Add error sample to buffer using specified replacement strategy based on timestep grid."""
        grid_idx = self._get_timestep_grid(timestep)
        error_cpu = error_sample.detach().cpu()
        
        if len(self.y_error_buffer[grid_idx]) < self.error_buffer_size:
            # Buffer not full, simply add
            self.y_error_buffer[grid_idx].append(error_cpu)
        else:
            # Buffer full, use specified replacement strategy
            if self.buffer_replacement_strategy == "random":
                # Random replacement - O(1), fastest
                replace_idx = random.randint(0, len(self.y_error_buffer[grid_idx]) - 1)
                self.y_error_buffer[grid_idx][replace_idx] = error_cpu
                
            elif self.buffer_replacement_strategy == "fifo":
                # First-in-first-out - O(1), simple queue behavior
                self.y_error_buffer[grid_idx].pop(0)
                self.y_error_buffer[grid_idx].append(error_cpu)
                
            elif self.buffer_replacement_strategy == "l2_batch":
                # Batch L2 computation - O(n) but vectorized, much faster than original
                distances = self._compute_l2_distance_batch(error_cpu, self.y_error_buffer[grid_idx])
                most_similar_idx = torch.argmin(distances).item()
                self.y_error_buffer[grid_idx][most_similar_idx] = error_cpu
                
            elif self.buffer_replacement_strategy == "l2_similarity":
                # Original L2 similarity method - O(n), slowest but most precise
                min_distance = float('inf')
                most_similar_idx = -1
                
                for i, stored_error in enumerate(self.y_error_buffer[grid_idx]):
                    distance = self._compute_l2_distance(error_cpu, stored_error)
                    if distance < min_distance:
                        min_distance = distance
                        most_similar_idx = i
                
                if most_similar_idx != -1:
                    self.y_error_buffer[grid_idx][most_similar_idx] = error_cpu

    def _sample_noise_error_from_noise_buffer(self, latents, timestep):
        """Randomly sample an error from the buffer based on timestep grid."""
        grid_idx = self._get_timestep_grid(timestep)
        
        if not self.latent_error_buffer[grid_idx]:
            return torch.zeros_like(latents)
        
        # Randomly select one sample from the corresponding grid
        selected_sample = random.choice(self.latent_error_buffer[grid_idx])
        error_sample = selected_sample.to(self.device)

        min_mod = 1.0 - self.error_modulate_factor
        max_mod = 1.0 + self.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)
        error_sample = error_sample * intensity_mod

        return error_sample

    def _sample_latent_error_from_latent_buffer(self, latents, timestep):
        """Randomly sample an error from the buffer based on timestep grid."""
        grid_idx = self._get_timestep_grid(timestep)
        
        if not self.y_error_buffer[grid_idx]:
            return torch.zeros_like(latents)
        
        # Randomly select one sample from the corresponding grid
        selected_sample = random.choice(self.y_error_buffer[grid_idx])
        error_sample = selected_sample.to(self.device)

        min_mod = 1.0 - self.error_modulate_factor
        max_mod = 1.0 + self.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)
        error_sample = error_sample * intensity_mod

        return error_sample
    
    def _sample_y_error_from_latent_buffer(self, latents, timestep):
        """Specially sample y_error from buffer - can be configured to sample from all grids or custom range."""
        if self.y_error_sample_range is not None:
            # Sample from custom timestep range
            start_grid, end_grid = self.y_error_sample_range
            all_samples = []
            for grid_idx in range(start_grid, min(end_grid + 1, len(self.y_error_buffer))):
                buffer = self.y_error_buffer[grid_idx]
                if buffer:  # Only add non-empty buffers
                    all_samples.extend(buffer)
            
            if not all_samples:
                return torch.zeros_like(latents)
            
            # Randomly select one sample from the custom range
            selected_sample = random.choice(all_samples)
            
        elif self.y_error_sample_from_all_grids:
            # Sample from all grids that have data
            all_samples = []
            for grid_idx, buffer in self.y_error_buffer.items():
                if buffer:  # Only add non-empty buffers
                    all_samples.extend(buffer)
            
            if not all_samples:
                return torch.zeros_like(latents)
            
            # Randomly select one sample from all available samples
            selected_sample = random.choice(all_samples)
        else:
            # Sample from current timestep grid only (original behavior)
            grid_idx = self._get_timestep_grid(timestep)
            
            if not self.y_error_buffer[grid_idx]:
                return torch.zeros_like(latents)
            
            # Randomly select one sample from the corresponding grid
            selected_sample = random.choice(self.y_error_buffer[grid_idx])
        
        error_sample = selected_sample.to(self.device)

        min_mod = 1.0 - self.error_modulate_factor
        max_mod = 1.0 + self.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)
        error_sample = error_sample * intensity_mod

        return error_sample

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.pipe_VAE.requires_grad_(False)
        self.pipe_VAE.eval()
        self.dwpose_embedding.train()
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):

        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)

        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            try:
                state_dict = load_state_dict(pretrained_lora_path)
            except:
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            
            state_dict_new = {}
            state_dict_new_module = {}
            for key in state_dict.keys():
                if 'pipe.dit.' in key:
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                if "dwpose_embedding" in key:
                    state_dict_new_module[key] = state_dict[key]
            state_dict = state_dict_new
            state_dict_new = {}

            for key in state_dict_new_module:
                if "dwpose_embedding" in key:
                    state_dict_new[key.split("dwpose_embedding.")[1]] = state_dict_new_module[key]
            self.dwpose_embedding.load_state_dict(state_dict_new, strict=True)

            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    def training_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        self.pipe_VAE.device = self.device
        dwpose_data = self.dwpose_embedding((torch.cat([batch["dwpose_data"][:, :, :1].repeat(1, 1, 3, 1, 1), batch["dwpose_data"]], dim=2)/255.).to(self.device))

        with torch.no_grad():
            if video is not None:
                # prompt
                prompt_emb = self.pipe_VAE.encode_prompt(text)
                # video
                video = video.to(dtype=self.pipe_VAE.torch_dtype, device=self.pipe_VAE.device)
                latents = self.pipe_VAE.encode_video(video, **self.tiler_kwargs)[0]
                # image
                if "first_ref_frames" in batch: # [1, 853, 480, 3]
                    first_ref_frames = [Image.fromarray(frame[0].cpu().to(torch.uint8).numpy()) for frame in batch["first_ref_frames"]]
                    _, _, num_frames, height, width = video.shape

                    # for padding
                    rand_ref_frame = Image.fromarray(batch["random_ref_frame"][0].cpu().to(torch.uint8).numpy())
                    num_condition = 1
                    image_emb = self.pipe_VAE.encode_images_adaptive(first_ref_frames[:num_condition], rand_ref_frame, num_frames, height, width, use_first_aug=self.use_first_aug, ref_pad_cfg=None, ref_pad_num=self.ref_pad_num)
                else:
                    image_emb = {}
                batch = {"latents": latents.unsqueeze(0), "prompt_emb": prompt_emb, "image_emb": image_emb}

        p1 = random.random()
        p2 = random.random()
        p = random.random()
        
        if p1 < 0.05:
            dwpose_data = torch.zeros_like(dwpose_data)
        if p2 < 0.1:
            dwpose_data = torch.zeros_like(dwpose_data)

        latents = batch["latents"].to(self.device)  # [1, 16, 21, 60, 104]
        prompt_emb = batch["prompt_emb"] # batch["prompt_emb"]["context"]:  [1, 1, 512, 4096]
        y_set_null=False
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device) # [1, 257, 1280]
            if p < 0.1:
                image_emb["clip_feature"] = torch.zeros_like(image_emb["clip_feature"]) # [1, 257, 1280]
        if "y" in image_emb:
            if p < 0.1:
                y_set_null=True
                image_emb["y"] = torch.zeros_like(image_emb["y"])
            image_emb["y"] = image_emb["y"].to(self.device)
        condition = rearrange(dwpose_data, 'b c f h w -> b (f h w) c').contiguous()
        if self.remove_pose:
            condition = None
        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(device=self.pipe.device)

        self.iteration_count += 1
        noise_w_error = noise 
        latents_w_error = latents 
        # Check if buffer has data for the current timestep grid
        current_grid_idx = self._get_timestep_grid(timestep)
        has_latent_buffer_data = len(self.latent_error_buffer[current_grid_idx]) > 0
        
        # For y_error, check based on sampling configuration
        if self.y_error_sample_range is not None:
            # Check custom range
            start_grid, end_grid = self.y_error_sample_range
            has_y_buffer_data = any(
                len(self.y_error_buffer[grid_idx]) > 0 
                for grid_idx in range(start_grid, min(end_grid + 1, len(self.y_error_buffer)))
            )
        elif self.y_error_sample_from_all_grids:
            # Check all grids
            has_y_buffer_data = any(len(buffer) > 0 for buffer in self.y_error_buffer.values())
        else:
            # Check current grid only
            has_y_buffer_data = len(self.y_error_buffer[current_grid_idx]) > 0

        add_error_latent = False
        add_error_noise = False
        add_error_y = False

        noise_random = random.random()
        y_random = random.random()
        latent_random = random.random()
        clean_random = random.random()

        if noise_random < self.noise_prob:
            add_error_noise = True
        if y_random < self.y_prob:
            add_error_y = True
        if latent_random < self.latent_prob:
            add_error_latent = True
        
        use_clean_input = False
        if clean_random < self.clean_prob:
            add_error_noise = False
            add_error_y = False
            add_error_latent = False
            use_clean_input = True

        if add_error_noise and has_latent_buffer_data:
            noise_error_sampled = self._sample_noise_error_from_noise_buffer(latents, timestep)
            noise_w_error = noise + noise_error_sampled.to(latents.dtype)

        if add_error_y and (not y_set_null) and has_y_buffer_data:  
            y_error_sampled = self._sample_y_error_from_latent_buffer(latents, timestep)
            max_start_idx = max(0, y_error_sampled.shape[2] - self.y_error_num)
            random_frame_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
            error_to_add = y_error_sampled[:, :, random_frame_idx:random_frame_idx+self.y_error_num, ...]
            image_emb["y"][:, 4:,:self.y_error_num,  ...] = image_emb["y"][:, 4:,:self.y_error_num,  ...] + error_to_add.to(image_emb["y"].dtype)


        if add_error_latent and has_latent_buffer_data:
            latent_error_sampled = self._sample_latent_error_from_latent_buffer(latents, timestep)
            latents_w_error = latents + latent_error_sampled.to(latents.dtype)

        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents_w_error, noise_w_error, timestep)  # error-corrupted X_t
        training_target = self.pipe.scheduler.training_target(latents, noise_w_error, timestep)  # self-corrected velocity pointing to clean latent

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            add_condition=condition,
        )
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        if self.use_error_recycling:
            with torch.no_grad():
                x_0_pred = self.pipe.scheduler.step(noise_pred, timestep, noisy_latents, to_final=True, self_corr=True)
                noise_corr_gt = self.pipe.scheduler.step(training_target, timestep, noisy_latents, to_final=True, self_corr=True)
                noise_error = x_0_pred - noise_corr_gt
                x_1_pred = self.pipe.scheduler.step(noise_pred, timestep, noisy_latents, to_final=True, self_corr=False)

                latent_corr_gt = self.pipe.scheduler.step(training_target, timestep, noisy_latents, to_final=True, self_corr=False)
                y_error = x_1_pred - latent_corr_gt

                # Check if we're in warmup phase
                if self.iteration_count <= self.buffer_warmup_iter:
                    # During warmup: gather errors and timesteps from all GPUs and update buffers
                    gathered_noise_errors = self.all_gather(noise_error)
                    gathered_y_errors = self.all_gather(y_error)
                    gathered_timesteps = self.all_gather(timestep)
                    if use_clean_input:
                        p = random.random()
                        if p < self.clean_buffer_update_prob:
                            self._update_error_buffers_distributed(gathered_noise_errors, gathered_y_errors, gathered_timesteps)
                    else:
                        self._update_error_buffers_distributed(gathered_noise_errors, gathered_y_errors, gathered_timesteps)
                else:
                    # After warmup: only use local GPU errors
                    # self._update_error_buffers_local(noise_error, y_error, timestep)
                    if use_clean_input:
                        p = random.random()
                        if p < self.clean_buffer_update_prob:
                            self._update_error_buffers_local(noise_error, y_error, timestep)
                    else:
                        self._update_error_buffers_local(noise_error, y_error, timestep)

        self.log("train_loss", loss, prog_bar=True)
        if self.use_error_recycling:
            # Log buffer stats for current timestep grid
            current_grid_idx = self._get_timestep_grid(timestep)
            total_latent_buffer_size = sum(len(buffer) for buffer in self.latent_error_buffer.values())
            total_y_buffer_size = sum(len(buffer) for buffer in self.y_error_buffer.values())
            
            self.log("latent_error_buffer_size", total_latent_buffer_size, prog_bar=False)
            self.log("y_error_buffer_size", total_y_buffer_size, prog_bar=False)
            self.log(f"latent_error_buffer_grid_{current_grid_idx}", len(self.latent_error_buffer[current_grid_idx]), prog_bar=False)
            self.log(f"y_error_buffer_grid_{current_grid_idx}", len(self.y_error_buffer[current_grid_idx]), prog_bar=False)
            self.log("current_timestep_grid", current_grid_idx, prog_bar=False)
            self.log("iteration_count", self.iteration_count, prog_bar=False)
            self.log("warmup_phase", int(self.iteration_count <= self.buffer_warmup_iter), prog_bar=False)
        return loss

    def _update_error_buffers_distributed(self, gathered_noise_errors, gathered_y_errors, gathered_timesteps):
        """Update error buffers with samples gathered from all processes."""
        # gathered_tensors have shape [num_gpus, batch_size, ...] for errors
        # gathered_timesteps have shape [num_gpus, batch_size] for timesteps
        # In this case, batch_size is 1, so shapes are [num_gpus, 1, ...] and [num_gpus, 1]
        num_gpus = gathered_noise_errors.shape[0]
        for i in range(num_gpus):
            noise_error_sample = gathered_noise_errors[i]
            y_error_sample = gathered_y_errors[i]
            timestep_sample = gathered_timesteps[i]  # Get the corresponding timestep for this GPU
            self._add_error_to_latent_buffer(noise_error_sample, timestep_sample)
            self._add_error_to_y_buffer(y_error_sample, timestep_sample)

    def _update_error_buffers_local(self, noise_error, y_error, timestep):
        """Update error buffers with samples from local GPU only (post-warmup)."""
        self._add_error_to_latent_buffer(noise_error, timestep)
        self._add_error_to_y_buffer(y_error, timestep)

    def get_noise_buffer_stats(self):
        """Get statistics of the noise buffer for debugging"""
        stats = {}
        
        # Calculate total buffer sizes across all grids
        total_latent_buffer_size = sum(len(buffer) for buffer in self.latent_error_buffer.values())
        total_y_buffer_size = sum(len(buffer) for buffer in self.y_error_buffer.values())
        
        stats["latent_buffer_size"] = total_latent_buffer_size
        stats["y_buffer_size"] = total_y_buffer_size
        stats["max_buffer_size_per_grid"] = self.error_buffer_size
        stats["timestep_grid_size"] = self.timestep_grid_size
        stats["num_grids"] = len(self.latent_error_buffer)
        stats["iteration_count"] = self.iteration_count
        stats["buffer_warmup_iter"] = self.buffer_warmup_iter
        stats["in_warmup_phase"] = self.iteration_count <= self.buffer_warmup_iter
        stats["buffer_replacement_strategy"] = self.buffer_replacement_strategy
        
        # Add per-grid statistics
        stats["per_grid_stats"] = {}
        for grid_idx in range(len(self.latent_error_buffer)):
            grid_stats = {
                "latent_buffer_size": len(self.latent_error_buffer[grid_idx]),
                "y_buffer_size": len(self.y_error_buffer[grid_idx]),
                "timestep_range": f"{grid_idx * self.timestep_grid_size}-{(grid_idx + 1) * self.timestep_grid_size - 1}"
            }
            
            if len(self.latent_error_buffer[grid_idx]) > 0:
                all_samples_tensor = torch.stack(self.latent_error_buffer[grid_idx])
                grid_stats["latent_buffer_stats"] = {
                    "mean": all_samples_tensor.mean().item(),
                    "std": all_samples_tensor.std().item(),
                    "max": all_samples_tensor.max().item(),
                    "min": all_samples_tensor.min().item(),
                }
            
            if len(self.y_error_buffer[grid_idx]) > 0:
                all_samples_tensor = torch.stack(self.y_error_buffer[grid_idx])
                grid_stats["y_buffer_stats"] = {
                    "mean": all_samples_tensor.mean().item(),
                    "std": all_samples_tensor.std().item(),
                    "max": all_samples_tensor.max().item(),
                    "min": all_samples_tensor.min().item(),
                }
            
            stats["per_grid_stats"][grid_idx] = grid_stats
        
        return stats
    
    def reset_noise_buffer(self):
        """Reset the noise buffer (useful for debugging)"""
        num_grids = (1000 + self.timestep_grid_size - 1) // self.timestep_grid_size
        self.latent_error_buffer = {i: [] for i in range(num_grids)}
        self.y_error_buffer = {i: [] for i in range(num_grids)}
        self.iteration_count = 0
        print("The noise buffers have been reset.")

    def configure_optimizers(self):
        trainable_modules = [
            {'params': filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())},
            {'params': self.dwpose_embedding.parameters()},
        ]
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        
        # Configure optimizer with gradient clipping
        optimizer_config = {
            "optimizer": optimizer,
        }
        # Add gradient clipping if enabled
        if self.gradient_clip_val > 0:
            optimizer_config["gradient_clip_val"] = self.gradient_clip_val
            optimizer_config["gradient_clip_algorithm"] = "norm"  # Use gradient norm clipping
        
        return optimizer_config

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), args=None):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu", train_architecture=args.train_architecture)
        model_manager.load_models(model_path)
        self.pipe = SVIDanceVideoPipeline.from_model_manager(model_manager, is_test=False)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_ref_frames" in batch:
                first_ref_frames = [Image.fromarray(frame.cpu().numpy()) for frame in batch["first_ref_frames"]]
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_ref_frames[0], num_frames, height, width)
            elif "first_ref_frame" in batch:
                first_ref_frame = Image.fromarray(batch["first_ref_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_ref_frame, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
            torch.save(data, path + ".tensors.pth")


def train_svi_dance(args):
    dataset = TextVideoDataset_onestage(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
        steps_per_epoch=args.steps_per_epoch,
        args=args,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model_VAE = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        args=args,
    )
    model = LightningModelForTrain_onestage(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        model_VAE=model_VAE,
        args=args,
    )
  
    logger = None
    
    # Configure trainer with gradient clipping support
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": "gpu",
        "devices": "auto",
        "num_nodes": args.num_nodes if hasattr(args, "num_nodes") else 1,
        "precision": "bf16",
        "strategy": args.training_strategy,
        "default_root_dir": args.output_path,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "callbacks": [pl.pytorch.callbacks.ModelCheckpoint(every_n_train_steps=100, save_top_k=1)],
        "logger": logger,
    }
    
    # Add gradient clipping if enabled
    if hasattr(args, 'gradient_clip_val') and args.gradient_clip_val > 0:
        trainer_kwargs["gradient_clip_val"] = args.gradient_clip_val
        trainer_kwargs["gradient_clip_algorithm"] = "norm"
    
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    args = parse_args()
    args = update_experiment_path(args, short=True)
    print_args(args)
    save_args_to_yaml(args, args.output_path)
    train_svi_dance(args)