import torch
from diffsynth import ModelManager, save_video, SVITalkVideoPipeline
import os
import numpy as np
import random
import pickle
import torchvision
from PIL import Image
import cv2
import subprocess
from tqdm import tqdm
import argparse
import glob
import random
from datetime import datetime
from utils.video_process import image_compose_width
from utils.image_process import *
from utils.project_utils import *

height = 480
width = 512
seed = None
# seed = 42
max_frames = 81
use_teacache = False

def parse_args():
    parser = argparse.ArgumentParser(description="SVI Talk Video Generation Script. Use --ref_image_path and --audio_path for custom inputs.")
    parser.add_argument(
        "--dit_root",
        default='weights/Wan2.1-I2V-14B-480P/',
        type=str,
        help="Root directory of the Wan2.1-I2 model.",
    )
    parser.add_argument(
        "--extra_module_root",
        default="weights/Stable-Video-Infinity/version-1.0/svi-talk.safetensors",
        type=str,
    )
    parser.add_argument(
        "--output",
        default="video_out/",
        type=str,
    )
    parser.add_argument(
        "--cfg_scale_audio",
        default=5.0,
        type=float,
        help="CFG scale for audio conditioning",
    )
    parser.add_argument(
        "--cfg_scale_text",
        default=2.0,
        type=float,
        help="CFG scale for text conditioning",
    )
    parser.add_argument(
        "--train_architecture",
        default='lora',
        type=str,
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--ref_pad_cfg",
        default=False,
        action="store_true",
        help="Whether to set mask with only 1-frame 1.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether to use tiled encoding.",
    )
    parser.add_argument(
        "--ref_pad_num",
        type=int,
        default=-1,  # 0 -> no padding k-> padding k ,-1 -> full padding
        help="Number of reference frames to pad with",
    )
    parser.add_argument(
        "--num_motion_frames",
        type=int,
        default=1,
        help="Number of motion frames to generate."
    )
    parser.add_argument(
        "--num_clips",
        type=int,
        default=50,
        help="Number of clips to generate."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of clips to generate."
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default=None,
        help="Path to reference image file (jpg/png). If not specified, will use default directories."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to audio file (wav/mp3/m4a/flac). If not specified, will use default directories."
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print_args(args)
    
    # Check if custom paths are provided
    if not os.path.exists(args.ref_image_path):
        raise FileNotFoundError(f"Reference image not found: {args.ref_image_path}")
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    
    # Create single test configuration with custom paths
    test_list_path = [{
        "sample_fps": 1,
        "ref_image_root": os.path.dirname(args.ref_image_path) + '/',
        "ref_image_file": os.path.basename(args.ref_image_path),
        "audio_path": args.audio_path,
        "prompt_name": "custom",
        "prompt": "a person is talking",
        "emotion_word": 'natural',
        "negative_prompt": "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    }]
    
    print(f"Using custom reference image: {args.ref_image_path}")
    print(f"Using custom audio file: {args.audio_path}")
    
    # random.seed(seed)   
    random.shuffle(test_list_path)
    # test_list_path.insert(0, test_first_sample)
    root = args.dit_root
    model_manager = ModelManager(device="cpu", train_architecture=args.train_architecture)
    model_manager.load_models(
        [root + "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32, # Image Encoder is loaded with float32
    )
    os.makedirs(args.output, exist_ok=True)
    model_manager.load_models(
        [
            [
                root + "diffusion_pytorch_model-00001-of-00007.safetensors",
                root + "diffusion_pytorch_model-00002-of-00007.safetensors",
                root + "diffusion_pytorch_model-00003-of-00007.safetensors",
                root + "diffusion_pytorch_model-00004-of-00007.safetensors",
                root + "diffusion_pytorch_model-00005-of-00007.safetensors",
                root + "diffusion_pytorch_model-00006-of-00007.safetensors",
                root + "diffusion_pytorch_model-00007-of-00007.safetensors",
                root + "multitalk.safetensors"

            ],
            root + "models_t5_umt5-xxl-enc-bf16.pth",
            root + "Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )

    extra_module_root = args.extra_module_root
    if extra_module_root.endswith('.safetensors'):
        safetensors_files = [extra_module_root]
    else:
        safetensors_files = glob.glob(os.path.join(extra_module_root, "*.safetensors"))
        safetensors_files.sort()

    model_manager.load_lora_v2(safetensors_files, lora_alpha=args.lora_alpha)

    pipe = SVITalkVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda", is_test=True)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    for path_dir_per in test_list_path:
        sample_fps = path_dir_per["sample_fps"]  # frame interval for sampling
        
        # Handle custom ref_image_file or use find_reference_image
        if "ref_image_file" in path_dir_per:
            ref_image_rgb_path = os.path.join(path_dir_per["ref_image_root"], path_dir_per["ref_image_file"])
        else:
            ref_image_rgb_path = find_reference_image(path_dir_per["ref_image_root"])

        emotion_word = path_dir_per.get("emotion_word")
        audio_path = path_dir_per["audio_path"]

        height, width = calculate_dimensions(ref_image_rgb_path, max_width=640)
        misc_size = [height,width]
        def resize(image):
            image = torchvision.transforms.functional.resize(
                image,
                (height, width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            return torch.from_numpy(np.array(image))

        frames_all = {}

        # Only load reference image for all frames
        ref_img = Image.fromarray(cv2.cvtColor(cv2.imread(ref_image_rgb_path), cv2.COLOR_BGR2RGB))
        for i in range(max_frames):
            frame_key = f"{i+1:05d}.jpg"
            frames_all[frame_key] = ref_img

        stride = sample_fps
        _total_frame_num = len(frames_all)
        cover_frame_num = (stride * max_frames)
        
        if _total_frame_num < cover_frame_num + 1:
            start_frame = 0
            end_frame = _total_frame_num-1
            stride = max((_total_frame_num//max_frames),1)
            end_frame = min(stride*max_frames, _total_frame_num)
        else:
            start_frame = 0
            end_frame = start_frame + cover_frame_num

        frame_list = []

        rand_ref_frame = frames_all[list(frames_all.keys())[0]]
        if rand_ref_frame.mode != 'RGB':
            rand_ref_frame = rand_ref_frame.convert('RGB')

        # sample sequence
        for i_index in range(start_frame, end_frame, stride):
            if i_index < len(frames_all):  # Check index within bounds
                i_key = list(frames_all.keys())[i_index]
                i_frame = frames_all[i_key]
                if i_frame.mode != 'RGB':
                    i_frame = i_frame.convert('RGB')
                frame_list.append(i_frame)
        
        # padding
        if (end_frame-start_frame) < max_frames:
            for _ in range(max_frames-(end_frame-start_frame)):
                i_key = list(frames_all.keys())[end_frame-1]
                i_frame = frames_all[i_key]
                if i_frame.mode != 'RGB':
                    i_frame = i_frame.convert('RGB')
                frame_list.append(i_frame)

        have_frames = len(frame_list)>0
        middle_indix = 0

        caption = path_dir_per["prompt"]
        output_img_video = []
        for ii in range(len(frame_list)):
            # Create visualization: just reference image
            output_img_video.append(tensor2pil(resize(rand_ref_frame)))

        rand_ref_frame_final = rand_ref_frame.resize((width, height))
        rand_ref_frame_final_gt = torch.from_numpy(np.array(rand_ref_frame_final)).clone()

        num_motion_frames = args.num_motion_frames
        num_clips = args.num_clips
    
        video_list = []
        # Generate a cleaner filename with timestamp for streaming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_name = ref_image_rgb_path.split('/')[-2]
        audio_name = audio_path.split('/')[-1].split('.')[0]
        prompt_name = emotion_word
        base_filename = f"{ref_name}_{audio_name}_{prompt_name}_cfga{args.cfg_scale_audio}_cfgt{args.cfg_scale_text}_step{args.num_steps}_{timestamp}"
        
        for chunk_idx in range(num_clips):
            audio_start_idx = chunk_idx * 81
            if audio_start_idx > 0:
                audio_start_idx = audio_start_idx - num_motion_frames

            video = pipe(
                prompt=path_dir_per["prompt"],
                negative_prompt=path_dir_per["negative_prompt"],
                input_image=rand_ref_frame_final,
                num_inference_steps=50,
                cfg_scale=dict(
                    audio=args.cfg_scale_audio,
                    text=args.cfg_scale_text),
                seed=seed, tiled=args.tiled,
                random_ref_frame=rand_ref_frame_final_gt,
                height=height,
                width=width,
                tea_cache_l1_thresh=0.3 if use_teacache else None,
                tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,
                audio_path=audio_path,
                use_controlnet=False,
                audio_start_idx=audio_start_idx,
                args=args
                
            )
            rand_ref_frame_final = video[-num_motion_frames:]

            if chunk_idx < num_clips - 1:
                # video_list += video[:-num_motion_frames]
                video_list += video
            else:
                video_list += video

            # Save intermediate video after each chunk
            video_out_chunk = []
            for ii in range(len(video_list)):
                ss = video_list[ii]
                video_out_chunk.append(image_compose_width(output_img_video[min(ii, len(output_img_video)-1)], ss))

            # Save intermediate video file
            chunk_filename = os.path.join(args.output, f"{base_filename}_chunk_{chunk_idx+1:02d}_of_{num_clips:02d}.mp4")
            print(f"Saving chunk {chunk_idx+1}/{num_clips}: {chunk_filename}")
            save_video(video_out_chunk, chunk_filename, fps=25, quality=8)
            
            # Add audio to intermediate video if available
            if audio_path and os.path.exists(audio_path):
                video_with_audio_filename = chunk_filename.replace('.mp4', '_with_audio.mp4')
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', chunk_filename,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-shortest',
                    '-y',
                    video_with_audio_filename
                ]
                try:
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                    print(f"Video with audio saved: {video_with_audio_filename}")
                    os.remove(chunk_filename)  # Remove video-only file
                except subprocess.CalledProcessError as e:
                    print(f"ffmpeg merge failed for chunk {chunk_idx+1}: {e}")

        video = video_list
