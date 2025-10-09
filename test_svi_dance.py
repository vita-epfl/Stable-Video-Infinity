import torch
from diffsynth import ModelManager, save_video, SVIDanceVideoPipeline
import os
import pickle
import numpy as np
import random
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import  torch.nn  as nn
import subprocess
from tqdm import tqdm
import torch, os, argparse
from datetime import datetime
from utils.video_process import *
from utils.image_process import *
from utils.project_utils import *
import glob
import random
import io

height = 832
width = 480
seed = None
max_frames = 81
use_teacache = False

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--dit_root",
        default='./weights/Wan2.1-I2V-14B-480P/',
        type=str,
        help="Root directory of the Wan2.1-I2V-14B-480P model.",
    )
    parser.add_argument(
        "--extra_module_root",
        default="weights/Stable-Video-Infinity/version-1.0/svi-dance.safetensors",
        type=str,
    )
    parser.add_argument(
        "--output",
        default="videos/",
        type=str,
    )
    parser.add_argument(
        "--cfg_scale_audio",
        default=1.0,
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
        "--ref_pad_cfg",
        default=False,
        action="store_true",
        help="Whether to set mask with only 1-frame 1.",
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
        default=10,
        help="Number of clips to generate."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of steps to generate."
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="Number of reference frames to use."
    )
    parser.add_argument(
        "--remove_pose",
        action="store_true",
        default=False,
        help="Whether to remove pose data. Default is False."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Image path. Default is False."
    )
    parser.add_argument(
        "--pose_path",
        type=str,
        required=True,
        help="Pose path. Default is False."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print_args(args)
    # Base directories

    test_list_path = [{
        "sample_fps": 1,
        "ref_image_root": args.image_path,
        "cond_video_root": args.pose_path,
        "audio_path": "",
        "prompt_name": "default",
        "prompt": "a person is path_dir_per",
        "emotion_word": 'natural',
        "negative_prompt": "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    }]

    # load models
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
# 
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
    pipe = SVIDanceVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda", is_test=True)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    # inference
    for path_dir_per in test_list_path:
        sample_fps = path_dir_per["sample_fps"]  # frame interval for sampling

        first_frame_path = path_dir_per["ref_image_root"]
        cond_pose_path = path_dir_per["cond_video_root"]
        emotion_word = path_dir_per.get("emotion_word")
        audio_path = path_dir_per["audio_path"]

        rand_ref_frame = Image.open(first_frame_path)
        video_reader = imageio.get_reader(cond_pose_path)
        pose = []
        for frame in video_reader:
            pose.append(frame)
        humanpose_data_torch = torch.stack([torch.from_numpy(np.array(frame).transpose(2,0,1)) for frame in pose], dim=0)

        # set image size
        original_width, original_height = rand_ref_frame.size
        max_width = 640
        if original_width <= max_width:
            width = original_width
            height = original_height
        else:
            aspect_ratio = original_height / original_width
            width = max_width
            height = int(width * aspect_ratio)
        
        width = (width // 16) * 16
        height = (height // 16) * 16
    
        misc_size = [height,width]
        def resize(image):
            image = torchvision.transforms.functional.resize(
                image,
                (height, width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            return torch.from_numpy(np.array(image))

        first_ref_frame_final = rand_ref_frame.resize((width, height))
        humanpose_data_all= resize_and_pad_to_target(humanpose_data_torch, (misc_size[0],  misc_size[1] ), pad_value=0).permute(1,0,2,3)  # [3, N, H, W]
        rand_ref_frame = torch.from_numpy(np.array(rand_ref_frame))

        # sample pose sequence
        stride = sample_fps
        sampled_idx = 0
        if humanpose_data_all.shape[1] < max_frames:
            print(f"Humanpose data is less than max_frames, repeated padding")
            repeat_times = (max_frames // humanpose_data_all.shape[1]) + 1
            humanpose_data_all = torch.cat([humanpose_data_all for _ in range(repeat_times)], dim=1)

        humanpose_data = humanpose_data_all[:, :max_frames, ...]
        sampled_idx = max_frames - 1

        num_motion_frames = args.num_motion_frames
        num_clips = args.num_clips

        video_list = []
        video_out_chunk = []
        # Generate a cleaner filename with timestamp for streaming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"output_cfgt{args.cfg_scale_text}_step{args.num_steps}_{timestamp}"
        
        for chunk_idx in range(num_clips):
            audio_start_idx = chunk_idx * 81
            if audio_start_idx > 0:
                audio_start_idx = audio_start_idx - num_motion_frames

            video = pipe(
                prompt='the person is dancing',
                negative_prompt=path_dir_per["negative_prompt"],
                input_image=first_ref_frame_final,
                num_inference_steps=args.num_steps,
                cfg_scale=dict(
                    audio=args.cfg_scale_audio,
                    text=args.cfg_scale_text),
                seed=seed, tiled=True,
                humanpose_data=None if args.remove_pose else humanpose_data,
                random_ref_frame=rand_ref_frame,
                height=height,
                width=width,
                tea_cache_l1_thresh=0.3 if use_teacache else None,
                tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,
                use_controlnet=False,
                cond_wo_pose=True,
                args=args
            )
            first_ref_frame_final = video[-num_motion_frames:]

            if chunk_idx < num_clips - 1:
                video_list = video[:-num_motion_frames]
            else:
                video_list = video


            for ii in range(len(video_list)):
                ss = video_list[ii]
                humanpose_frame = tensor2pil(humanpose_data[:, ii].permute(1, 2, 0).to(torch.uint8))
                frame_with_pose = image_compose_width(tensor2pil(resize(rand_ref_frame.permute(2, 0, 1))), humanpose_frame)
                video_out_chunk.append(image_compose_width(frame_with_pose, ss))

            # Save intermediate video file
            chunk_filename = os.path.join(args.output, f"{base_filename}_chunk_{chunk_idx+1:02d}_of_{num_clips:02d}.mp4")
            print(f"Saving chunk {chunk_idx+1}/{num_clips}: {chunk_filename}")
            save_video(video_out_chunk, chunk_filename, fps=25, quality=5)

            # sample pose sequence
            humanpose_data_new = torch.zeros(3, max_frames, misc_size[0], misc_size[1], device=humanpose_data.device)
            humanpose_data_new[:, :num_motion_frames] = humanpose_data[:, -num_motion_frames:]
            for i in range(num_motion_frames, max_frames):
                sampled_idx = sampled_idx % humanpose_data_all.shape[1]
                humanpose_data_new[:, i] = humanpose_data_all[:, sampled_idx]
                sampled_idx += 1
            humanpose_data = humanpose_data_new
            

        video = video_list

    