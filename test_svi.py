import torch
from diffsynth import ModelManager, save_video, SVIVideoPipeline
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
import torch.nn as nn
import cv2
import sys  
import subprocess
from tqdm import tqdm
import torch, os, argparse
from datetime import datetime
from utils.image_process import *
import glob
from utils.project_utils import print_args

height = 480
width = 832
seed = None
# seed = 42
max_frames = 81
use_teacache = False

def load_prompts_from_file(prompt_file_path):
    """Load prompts from a prompt.txt file"""
    if not os.path.exists(prompt_file_path):
        print(f"Warning: prompt file not found at {prompt_file_path}")
        return ["Default prompt: the subject is moving naturally"]
    
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Extract prompts from the file content
        # The format is: prompts = ["prompt1", "prompt2", ...]
        if 'prompts = [' in content:
            # Find the prompts list and extract it
            start_idx = content.find('prompts = [')
            if start_idx != -1:
                # Execute the assignment to extract prompts
                local_vars = {}
                exec(content[start_idx:], {}, local_vars)
                return local_vars.get('prompts', ["Default prompt: the subject is moving naturally"])
        
        # If not in expected format, treat each line as a prompt
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
        if lines:
            return lines
        else:
            return ["Default prompt: the subject is moving naturally"]
            
    except Exception as e:
        print(f"Error reading prompts from {prompt_file_path}: {e}")
        return ["Default prompt: the subject is moving naturally"]


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--dit_root",
        default='weights/Wan2.1-I2V-14B-480P/',
        type=str,
        help="Root directory of the Wan2.1-I2 model.",
    )
    parser.add_argument(
        "--extra_module_root",
        default="weights/Stable-Video-Infinity/version-1.0/svi-shot.safetensors",
        type=str,
    )
    parser.add_argument(
        "--output",
        default="videos/",
        type=str,
    )
    parser.add_argument(
        "--cfg_scale_text",
        default=5.0,
        type=float,
    )
    parser.add_argument(
        "--lora_alpha",
        default=1.0,
        type=float,
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
        help="Number of clips to generate."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data_inference/wan_i2v/",
        help="Root directory containing test samples."
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default=None,
        help="Direct path to reference image. If specified, will override data_root scanning."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Direct path to prompt file. If specified, will be used instead of prompt.txt in data_root."
    )
    parser.add_argument(
        "--test_samples",
        type=str,
        nargs='*',
        help="Specific test sample directories to process. If not specified, all samples will be processed."
    )
    parser.add_argument(
        "--max_prompts_per_sample",
        type=int,
        default=None,
        help="Maximum number of prompts to use per sample. If not specified, all prompts will be used."
    )
    parser.add_argument(
        "--ref_pad_num",
        type=int,
        default=0,  # 0 -> no padding k-> padding k , -1 -> full padding
        help="Number of reference frames to pad with",
    )
    parser.add_argument(
        "--use_first_prompt_only",
        default=False,
        action="store_true",
        help="Whether to use only the first prompt for all clips instead of cycling through different prompts.",
    )
    parser.add_argument(
        "--use_first_aug",
        default=False,
        action="store_true",
        help="Whether to use only the first prompt for all clips instead of cycling through different prompts.",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=832,
        help="Maximum width of the output video."
    )
    parser.add_argument(
        "--seed_times",
        type=int,
        default=42,
        help="Number of times to seed the random number generator."
    )
    parser.add_argument(
        "--repeat_first_clip",
        default=False,
        action="store_true",
        help="Whether to repeat the first frames."
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether to use tiled encoding.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs='+',
        default=[30, 52],
        help="Tile size.",
    )
    parser.add_argument(
        "--tile_stride",
        type=int,
        nargs='+',
        default=[15, 26],
        help="Tile stride.",
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        # default="Camera moves dynamically accordingly",
        # default="Camera slightly moves accordingly",
        default="none",
        help="Prefix to add before each prompt. Default is 'Camera moves dynamically accordingly'.",
    )
    parser.add_argument(
        "--prompt_repeat_times",
        type=int,
        default=1,
        help="Number of times to repeat each prompt before moving to the next. Default is 1 (no repeat).",
    )
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=6*10**9,
        help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print_args(args)
    
    # Define common negative prompt for all scenarios
    common_negative_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    # Check if direct paths are provided
    if args.ref_image_path and args.prompt_path:
        # Use direct paths
        print("Using direct paths for reference image and prompt file")
        print(f"Reference image: {args.ref_image_path}")
        print(f"Prompt file: {args.prompt_path}")
        
        if not os.path.exists(args.ref_image_path):
            print(f"Error: Reference image not found at {args.ref_image_path}")
            exit(1)
        if not os.path.exists(args.prompt_path):
            print(f"Error: Prompt file not found at {args.prompt_path}")
            exit(1)
        
        # Load prompts from specified file
        prompts = load_prompts_from_file(args.prompt_path)
        
        # Get directory and name from reference image path
        ref_image_dir = os.path.dirname(args.ref_image_path)
        ref_image_name = os.path.splitext(os.path.basename(args.ref_image_path))[0]
        
        test_list_path = [{
            "ref_image_root": ref_image_dir + '/',
            "prompt_name": ref_image_name,
            "prompts": prompts,
            "emotion_word": 'natural',
            "negative_prompt": common_negative_prompt,
            "direct_image_path": args.ref_image_path  # Store direct path
        }]
        
        print(f"Generated 1 test scenario with {len(prompts)} prompts")
    else:
        # Use original data_root scanning logic
        # Base directories - only reference images needed
        ref_image_base = args.data_root
        
        # Get all subdirectories automatically
        all_ref_image_dirs = [d for d in os.listdir(ref_image_base) if os.path.isdir(os.path.join(ref_image_base, d))]
        
        # Filter directories based on command line arguments
        if args.test_samples:
            # Use specified test samples
            ref_image_dirs = []
            for sample in args.test_samples:
                if sample in all_ref_image_dirs:
                    ref_image_dirs.append(sample)
                else:
                    print(f"Warning: Test sample '{sample}' not found in {ref_image_base}")
            if not ref_image_dirs:
                print("No valid test samples found. Exiting.")
                exit(1)
        else:
            # Use all available directories
            ref_image_dirs = all_ref_image_dirs
        
        print(f"Available test samples: {all_ref_image_dirs}")
        print(f"Processing {len(ref_image_dirs)} test samples: {ref_image_dirs}")
        
        # Generate test list for image-to-video generation
        test_list_path = []
        for ref_dir in ref_image_dirs:
            ref_dir_path = os.path.join(ref_image_base, ref_dir)
            prompt_file_path = os.path.join(ref_dir_path, 'prompt.txt')
            
            # Load prompts from file or use default
            prompts = load_prompts_from_file(prompt_file_path)
            
            test_list_path.append({
                "ref_image_root": ref_dir_path + '/',
                "prompt_name": ref_dir,
                "prompts": prompts,  # Use the loaded prompts list
                "emotion_word": 'natural',
                "negative_prompt": common_negative_prompt
            })
        
        print(f"Generated {len(test_list_path)} test scenarios with their corresponding prompts")

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

    pipe = SVIVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda", is_test=True)
    pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    for sample_idx, path_dir_per in enumerate(test_list_path):
        print(f"\n{'#'*100}")
        print(f"STARTING SAMPLE {sample_idx + 1}/{len(test_list_path)}: {path_dir_per['prompt_name']}")
        print(f"{'#'*100}")
        
        # Use direct image path if provided, otherwise search for reference image
        if "direct_image_path" in path_dir_per:
            ref_image_rgb_path = path_dir_per["direct_image_path"]
        else:
            ref_image_rgb_path = find_reference_image(path_dir_per["ref_image_root"])
        
        emotion_word = path_dir_per.get("emotion_word")
        
        print(f"Reference image: {ref_image_rgb_path}")
        print(f"Available prompts: {len(path_dir_per['prompts'])}")

        height, width = calculate_dimensions(ref_image_rgb_path, max_width=args.max_width)
        # height, width = calculate_dimensions(ref_image_rgb_path, max_width=832)
        print(f"Video dimensions: {width}x{height}")
        
        # Load reference image
        rand_ref_frame = Image.fromarray(cv2.cvtColor(cv2.imread(ref_image_rgb_path), cv2.COLOR_BGR2RGB))
        if rand_ref_frame.mode != 'RGB':
            rand_ref_frame = rand_ref_frame.convert('RGB')

        rand_ref_frame_final_single = rand_ref_frame.resize((width, height))
        if args.repeat_first_clip:
            rand_ref_frame_final = [rand_ref_frame_final_single] * args.num_motion_frames
        else:
            rand_ref_frame_final = rand_ref_frame_final_single 


        rand_ref_frame_final_gt = torch.from_numpy(np.array(rand_ref_frame_final_single)).clone()

        # Get prompts from the loaded prompts list
        loaded_prompts = path_dir_per["prompts"]
        
        print(f"\nProcessing {path_dir_per['prompt_name']} with {len(loaded_prompts)} prompts")

        # Pure image-to-video generation
        num_motion_frames = args.num_motion_frames
        
        # Calculate num_clips based on use_first_prompt_only setting
        if args.use_first_prompt_only:
            num_clips = args.num_clips  # Use the specified number of clips
        else:
            # Calculate effective prompts considering repeat times
            effective_prompts = len(loaded_prompts) * args.prompt_repeat_times
            num_clips = min(args.num_clips, effective_prompts)
        
        if args.use_first_prompt_only:
            print(f"Generating {num_clips} clips using the first prompt repeatedly")
        else:
            if args.prompt_repeat_times > 1:
                print(f"Generating {num_clips} clips with each prompt repeated {args.prompt_repeat_times} times")
            else:
                print(f"Generating {num_clips} clips using the first {min(num_clips, len(loaded_prompts))} prompts from the loaded sequence")
        
        seeds = range(0, 10000)
        video_list = []
        # Generate filename for streaming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_name = path_dir_per['prompt_name']
        prompt_name = emotion_word
        base_filename = f"i2v_{ref_name}_{prompt_name}_cfgt{args.cfg_scale_text}_step{args.num_steps}_{timestamp}"
        
        # Create sample-specific output directory
        sample_output_dir = os.path.join(args.output, f"{ref_name}_{timestamp}")
        os.makedirs(sample_output_dir, exist_ok=True)
        print(f"Created output directory for sample: {sample_output_dir}")

        for chunk_idx in range(num_clips):
            seed = int(seeds[chunk_idx] * args.seed_times)

            if args.seed_times== -1:
                seed= None

            # Choose prompt based on args.use_first_prompt_only and prompt_repeat_times
            if args.use_first_prompt_only:
                current_prompt = loaded_prompts[0]  # Always use first prompt
            else:
                # Calculate which prompt to use considering repeat times
                prompt_index = chunk_idx // args.prompt_repeat_times
                # Ensure we don't exceed available prompts
                prompt_index = prompt_index % len(loaded_prompts)
                current_prompt = loaded_prompts[prompt_index]
            
            # Add prompt prefix
            if args.prompt_prefix != "none":
                current_prompt = f"{args.prompt_prefix}, {current_prompt}"
            
            print(f"\n{'='*80}")
            print(f"PROCESSING SAMPLE: {ref_name}")
            print(f"CHUNK: {chunk_idx+1}/{num_clips}")
            if not args.use_first_prompt_only and args.prompt_repeat_times > 1:
                prompt_index = chunk_idx // args.prompt_repeat_times
                repeat_index = chunk_idx % args.prompt_repeat_times
                print(f"PROMPT INDEX: {prompt_index + 1} (Repeat: {repeat_index + 1}/{args.prompt_repeat_times})")
            print(f"PROMPT: {current_prompt}")
            if args.use_first_prompt_only:
                print(f"NOTE: Using first prompt only (use_first_prompt_only=True)")
            print(f"{'='*80}")
            print(f"Starting video generation...")
            
            video = pipe(
                prompt=current_prompt,
                negative_prompt=path_dir_per["negative_prompt"],
                input_image=rand_ref_frame_final,
                num_inference_steps=args.num_steps,
                cfg_scale=dict(text=args.cfg_scale_text),
                seed=seed, tiled=args.tiled,
                random_ref_frame=rand_ref_frame_final_gt,
                height=height,
                width=width,
                tea_cache_l1_thresh=0.3 if use_teacache else None,
                tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,
                args=args
            )
            print(f"Video generation completed!")
            rand_ref_frame_final = video[-num_motion_frames:]
            if chunk_idx < num_clips - 1:
                video_list += video[:-num_motion_frames]
            else:
                video_list += video

            chunk_filename = os.path.join(sample_output_dir, f"{base_filename}_chunk_{chunk_idx+1:02d}_of_{num_clips:02d}.mp4")
            chunk_filename_vis = os.path.join(sample_output_dir, f"{base_filename}.mp4")
            print(f"Saving chunk {chunk_idx+1}/{num_clips}: {chunk_filename}")

            # save_video(video_list, chunk_filename, fps=16, quality=8)
            save_video(video_list, chunk_filename_vis, fps=24, quality=8)              

            print(f"Chunk {chunk_idx+1}/{num_clips} saved successfully!")

        video = video_list
        print(f"\n{'#'*100}")
        print(f"COMPLETED SAMPLE {sample_idx + 1}/{len(test_list_path)}: {path_dir_per['prompt_name']}")
        print(f"{'#'*100}")

    print(f"\nCompleted processing all {len(test_list_path)} scenarios!")