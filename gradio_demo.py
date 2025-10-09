#!/usr/bin/env python3
"""
Stable Video Infinity - Gradio Demo
Interactive video generation demo based on streaming generation
"""

import gradio as gr
import torch
import os
import numpy as np
import cv2
from PIL import Image
import tempfile
import shutil
from datetime import datetime
import json
import argparse
from pathlib import Path
import base64

from diffsynth import ModelManager, save_video, SVIVideoPipeline
from utils.image_process import calculate_dimensions

class SVIGradioDemo:
    def __init__(self, model_config):
        self.model_config = model_config
        self.model_manager = None
        self.pipe = None
        self.is_initialized = False
        
        # Define model mode configurations
        self.model_modes = {
            "film": {
                "name": "SVI-Film (1 text prompt stream)",
                "description": "Suitable for cinematic narratives, storylines and long continuous shots",
                "lora_path": "weights/Stable-Video-Infinity/version-1.0/svi-film.safetensors",
                "demo_image": "data/toy_test/film/frame.jpg",
                "demo_prompts": "data/toy_test/film/prompt.txt",
                "num_motion_frames": 5,
                "ref_pad_num": 0,
                "use_first_prompt_only": False
            },
            "shot": {
                "name": "SVI-Shot (1 text prompt)",
                "description": "Suitable for camera movements, dynamic shooting effects",
                "lora_path": "weights/Stable-Video-Infinity/version-1.0/svi-shot.safetensors",
                "demo_image": "data/toy_test/shot/frame.jpg",
                "demo_prompts": "data/toy_test/shot/prompt.txt",
                "num_motion_frames": 1,
                "ref_pad_num": -1,
                "use_first_prompt_only": True
            }
        }
        
        self.current_mode = "film"  # Default mode
    
    def switch_model_mode(self, mode):
        """Switch model mode and reinitialize"""
        if mode not in self.model_modes:
            return f"❌ Unknown mode: {mode}"
        
        print(f"🔄 Switching to mode: {self.model_modes[mode]['name']}")
        self.current_mode = mode
        
        # Update model configuration
        mode_config = self.model_modes[mode]
        self.model_config["extra_module_root"] = mode_config["lora_path"]
        
        # Reset initialization state to force model reload
        self.is_initialized = False
        self.model_manager = None
        self.pipe = None
        
        return f"✅ Switched to {mode_config['name']} mode\\n📝 {mode_config['description']}\\n⚠️ Please click 'Initialize Models' to reload"
    
    def get_demo_content_for_mode(self, mode):
        """Get demo content for specified mode"""
        if mode not in self.model_modes:
            return None, ""
        
        mode_config = self.model_modes[mode]
        demo_image = None
        demo_prompts = ""
        
        # Load demo image
        if os.path.exists(mode_config["demo_image"]):
            try:
                demo_image = Image.open(mode_config["demo_image"])
                print(f"✅ Loaded demo image: {mode_config['demo_image']}")
            except Exception as e:
                print(f"⚠️ Could not load demo image: {e}")
        
        # Load demo prompts
        if os.path.exists(mode_config["demo_prompts"]):
            try:
                with open(mode_config["demo_prompts"], "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    # Extract prompts list content
                    if content.startswith("prompts = [") and content.endswith("]"):
                        demo_prompts = content[len("prompts = "):]
                    else:
                        demo_prompts = content
                print(f"✅ Loaded demo prompts: {mode_config['demo_prompts']}")
            except Exception as e:
                print(f"⚠️ Could not load demo prompts: {e}")
        
        return demo_image, demo_prompts
        
    def get_current_mode_config(self):
        """Get current mode configuration"""
        return self.model_modes[self.current_mode]
        
    def initialize_models(self):
        """Initialize models"""
        if self.is_initialized:
            return "Models already initialized!"
            
        try:
            print("Initializing models...")
            
            # Initialize model manager
            self.model_manager = ModelManager(
                device="cpu", 
                train_architecture=self.model_config["train_architecture"]
            )
            
            # Load CLIP model
            clip_model_path = os.path.join(
                self.model_config["dit_root"], 
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            )
            self.model_manager.load_models(
                [clip_model_path],
                torch_dtype=torch.float32,
            )
            
            # Load main models
            dit_models = [
                os.path.join(self.model_config["dit_root"], f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors")
                for i in range(1, 8)
            ]
            
            other_models = [
                os.path.join(self.model_config["dit_root"], "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(self.model_config["dit_root"], "Wan2.1_VAE.pth"),
            ]
            
            self.model_manager.load_models(
                [dit_models] + other_models,
                torch_dtype=torch.bfloat16,
            )
            
            # Load LoRA (if specified)
            if self.model_config["extra_module_root"] and os.path.exists(self.model_config["extra_module_root"]):
                # import glob
                # safetensors_files = glob.glob(
                #     os.path.join(self.model_config["extra_module_root"], "*.safetensors")
                # )
                # safetensors_files.sort()

                self.model_manager.load_lora_v2(
                    self.model_config["extra_module_root"], 
                    lora_alpha=self.model_config["lora_alpha"]
                )
            else:
                raise FileNotFoundError(f"LoRA path not found: {self.model_config['extra_module_root']}")
            
            # Initialize pipeline
            self.pipe = SVIVideoPipeline.from_model_manager(
                self.model_manager, 
                torch_dtype=torch.bfloat16, 
                device="cuda", 
                is_test=True
            )
            self.pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)
            
            self.is_initialized = True
            return "✅ Models initialized successfully!"
            
        except Exception as e:
            return f"❌ Error initializing models: {str(e)}"
    
    def generate_video(self, 
                      input_image, 
                      prompts_text, 
                      negative_prompt,
                      num_clips,
                      num_steps,
                      cfg_scale_text,
                      seed):
        """Main video generation function"""
        
        if not self.is_initialized:
            yield None, None
            return
        
        if input_image is None:
            yield None, None  
            return
        
        if not prompts_text.strip():
            yield None, None
            return
        
        # Get num_motion_frames from current mode configuration
        num_motion_frames = self.get_current_mode_config()["num_motion_frames"]
        
        try:
            # Create videos output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            videos_dir = os.path.join(os.getcwd(), "videos")
            os.makedirs(videos_dir, exist_ok=True)
            
            # Create subdirectory for this generation
            session_dir = os.path.join(videos_dir, f"session_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)
            
            # Process input image - following test_svi.py approach
            if hasattr(input_image, 'save'):  # PIL Image object
                image = input_image
            else:  # file path string
                image = Image.open(input_image)
                
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Calculate video dimensions
            height, width = calculate_dimensions(image, max_width=832)
            rand_ref_frame_final_single = image.resize((width, height))
            rand_ref_frame_final = rand_ref_frame_final_single
            
            # Convert to tensor - following test_svi.py
            rand_ref_frame_final_gt = torch.from_numpy(np.array(rand_ref_frame_final_single)).clone()
            
            # Parse prompts - supports multiple formats
            prompts_list = []
            prompts_text_stripped = prompts_text.strip()
            
            # Detect Python list format
            if prompts_text_stripped.startswith('[') and prompts_text_stripped.endswith(']'):
                try:
                    # Try to parse as Python list
                    import ast
                    parsed_list = ast.literal_eval(prompts_text_stripped)
                    if isinstance(parsed_list, list):
                        prompts_list = [str(prompt).strip() for prompt in parsed_list if str(prompt).strip()]
                        print(f"Parsed as Python list: {len(prompts_list)} prompts")
                    else:
                        raise ValueError("Not a valid list")
                except Exception as e:
                    print(f"Failed to parse as Python list: {e}")
                    # Fall back to other parsing methods
                    prompts_list = []
            
            # If list format parsing failed or not in list format, try other methods
            if not prompts_list:
                if '\\n' in prompts_text_stripped:
                    # Multi-line mode
                    prompts_list = [line.strip() for line in prompts_text_stripped.split('\\n') if line.strip()]
                    print(f"Parsed as multi-line: {len(prompts_list)} prompts")
                elif ',' in prompts_text_stripped:
                    # Comma-separated mode
                    prompts_list = [prompt.strip() for prompt in prompts_text_stripped.split(',') if prompt.strip()]
                    print(f"Parsed as comma-separated: {len(prompts_list)} prompts")
                else:
                    # Single prompt
                    prompts_list = [prompts_text_stripped]
                    print("Parsed as single prompt")
            
            if not prompts_list:
                yield None, None
                return
            
            print(f"Final prompts list ({len(prompts_list)} prompts): {prompts_list[:3]}{'...' if len(prompts_list) > 3 else ''}")
            
            # Generate video stream
            video_list = []

            
            # Use same seed calculation as test_svi.py
            seeds = range(0, 10000)
            
            for chunk_idx in range(num_clips):
                
                # Following test_svi.py seed calculation
                current_seed = int(seeds[chunk_idx] * 42)
                
                # Cycle through prompts, following test_svi.py streaming approach
                current_prompt = prompts_list[chunk_idx % len(prompts_list)]
                
                print(f"Chunk {chunk_idx+1}/{num_clips}: Using prompt '{current_prompt}' with seed {current_seed}")
                
                # Generate video clip - following test_svi.py parameters
                video = self.pipe(
                    prompt=current_prompt,
                    negative_prompt=negative_prompt,
                    input_image=rand_ref_frame_final,
                    num_inference_steps=num_steps,
                    cfg_scale=dict(text=cfg_scale_text),
                    seed=current_seed,
                    tiled=True,
                    random_ref_frame=rand_ref_frame_final_gt,
                    height=height,
                    width=width,
                    tea_cache_l1_thresh=None,  # Not using teacache
                    tea_cache_model_id=None,
                    args=argparse.Namespace(
                        ref_pad_cfg=False,
                        ref_pad_num=self.get_current_mode_config()["ref_pad_num"],
                        num_motion_frames=num_motion_frames
                    )
                )
                
                # Save current clip preview
                clip_preview_path = os.path.join(session_dir, f"clip_{chunk_idx+1:03d}_preview.mp4")
                clip_preview_path = os.path.abspath(clip_preview_path)  # Ensure absolute path
                
                # Save video directly
                save_video(video, clip_preview_path, fps=16, quality=8)
                
                print(f"Video generation for chunk {chunk_idx+1} completed! Preview saved to {clip_preview_path}")
                print(f"📁 File exists: {os.path.exists(clip_preview_path)}")
                print(f"📏 File size: {os.path.getsize(clip_preview_path) if os.path.exists(clip_preview_path) else 'N/A'} bytes")
                
                # Update reference frame - following test_svi.py
                rand_ref_frame_final = video[-num_motion_frames:]
                
                # Add to video list - following test_svi.py
                if chunk_idx < num_clips - 1:
                    video_list += video[:-num_motion_frames]
                else:
                    video_list += video
                
                # Save cumulative video (from clip 1 to current clip)
                cumulative_video_path = os.path.join(session_dir, f"cumulative_clips_1_to_{chunk_idx+1}.mp4")
                cumulative_video_path = os.path.abspath(cumulative_video_path)
                
                # Save cumulative video directly
                save_video(video_list, cumulative_video_path, fps=16, quality=8)
                
                print(f"✅ Cumulative video (clips 1-{chunk_idx+1}) saved to: {cumulative_video_path}")
                print(f"📁 Cumulative file exists: {os.path.exists(cumulative_video_path)}")
                print(f"📏 Cumulative file size: {os.path.getsize(cumulative_video_path) if os.path.exists(cumulative_video_path) else 'N/A'} bytes")
                
                # Real-time playback of latest cumulative video
                if os.path.exists(cumulative_video_path) and os.path.getsize(cumulative_video_path) > 0:
                    yield cumulative_video_path, None
                    print(f"✅ Yielded cumulative video (clips 1-{chunk_idx+1}) for playback")
                else:
                    print(f"❌ ERROR: Cumulative video file not found or empty: {cumulative_video_path}")
                    yield None, None
                
            # Final video is the last cumulative video
            final_cumulative_video = os.path.join(session_dir, f"cumulative_clips_1_to_{num_clips}.mp4")
            final_cumulative_video = os.path.abspath(final_cumulative_video)
            
            print(f"✅ Final cumulative video: {final_cumulative_video}")
            print(f"📁 Final file exists: {os.path.exists(final_cumulative_video)}")
            print(f"📏 Final file size: {os.path.getsize(final_cumulative_video) if os.path.exists(final_cumulative_video) else 'N/A'} bytes")
            
            # Save generation info
            info_path = os.path.join(session_dir, "generation_info.json")
            generation_info = {
                "timestamp": timestamp,
                "num_clips": num_clips,
                "num_prompts": len(prompts_list),
                "prompts": prompts_list,
                "seed": seed,
                "num_steps": num_steps,
                "cfg_scale": cfg_scale_text,
                "num_motion_frames": num_motion_frames,
                "video_dimensions": f"{width}x{height}",
                "cumulative_videos": [f"cumulative_clips_1_to_{i+1}.mp4" for i in range(num_clips)]
            }
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(generation_info, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Generation completed! Files saved in: {session_dir}")
            print(f"📂 Final cumulative video: {final_cumulative_video}")
            print(f"📋 Generation info: {info_path}")
            
            yield final_cumulative_video, final_cumulative_video
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield None, None

def create_demo(model_config):
    """Create Gradio interface"""
    
    demo_app = SVIGradioDemo(model_config)
    
    # Get current mode demo content
    current_demo_image, current_demo_prompts = demo_app.get_demo_content_for_mode(demo_app.current_mode)
    
    # Read logo and convert to base64
    logo_base64 = ""
    logo_path = "assets/logo_white.png"
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                logo_data = f.read()
                logo_base64 = base64.b64encode(logo_data).decode()
        except Exception as e:
            print(f"Warning: Could not load logo: {e}")
    
    # Use current mode demo content
    demo_image = current_demo_image
    demo_prompts = current_demo_prompts
    
    # Predefined prompts
    example_prompts = [
        "The sun hangs low over the horizon, casting a golden path across the water.",
        "Gentle waves rhythmically wash upon the wet, reflective sand.",
        "A majestic eagle soars through mountain peaks covered in morning mist.",
        "Cherry blossoms fall like snow in a peaceful Japanese garden.",
        "City lights twinkle as evening transitions to night.",
        "A train moves through a vast landscape under a starry sky."
    ]
    
    # Predefined multi-prompt examples
    streaming_examples = [
        "Gentle waves wash over the shore,\\nSeagulls fly across the horizon,\\nClouds drift slowly in the sky",
        "Fire dances in a cozy fireplace,\\nSparks rise and fade away,\\nWarmth radiates from the flames",
        "Leaves rustle in a gentle breeze,\\nSunlight filters through branches,\\nNature comes alive with motion",
        '["A cat sits peacefully in a sunny window", "The cat notices a bird outside and becomes alert", "The cat stretches and jumps down from the window", "The cat walks across the room with graceful steps", "The cat curls up in a cozy corner for a nap"]',
        '["A Siamese kitten rests snugly inside a straw hat", "The kitten decides to explore and jumps out of the hat", "The kitten sees a feather toy and pounces on it", "The kitten chases the toy around the living room", "The kitten tires itself out and falls asleep on the floor"]'
    ]
    
    # Add demo prompts to examples if available
    if demo_prompts:
        streaming_examples.append(demo_prompts)
    
    # Predefined negative prompt
    default_negative_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    with gr.Blocks(
        title="Stable Video Infinity Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto !important;
            padding: 20px !important;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white !important;
        }
        .video-container {
            max-height: 600px;
        }
        .main-header {
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            color: white !important;
        }
        .control-panel {
            background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
            border: 2px solid #475569;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            color: white !important;
        }
        .output-panel {
            background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
            border: 2px solid #475569;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            color: white !important;
        }
        .status-good {
            background-color: #22c55e !important;
            color: white !important;
            border-radius: 8px !important;
        }
        .status-error {
            background-color: #ef4444 !important;
            color: white !important;
            border-radius: 8px !important;
        }
        .custom-button {
            background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 12px !important;
            padding: 15px 30px !important;
            font-size: 16px !important;
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        .custom-button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 8px rgba(59, 130, 246, 0.6) !important;
        }
        .example-box {
            background: linear-gradient(135deg, #475569 0%, #334155 100%);
            border: 1px solid #64748b;
            border-radius: 12px;
            padding: 18px;
            margin: 12px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            color: white !important;
        }
        .section-header {
            color: #e2e8f0 !important;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #475569;
        }
        /* Dark theme labels and text */
        .gr-block label, .gr-form label {
            color: #e2e8f0 !important;
        }
        .gr-block p, .gr-form p {
            color: #cbd5e1 !important;
        }
        /* Beautify sliders */
        .gr-slider input[type="range"] {
            background: linear-gradient(to right, #3b82f6, #1e40af) !important;
        }
        /* Beautify text boxes */
        .gr-textbox textarea, .gr-textbox input {
            background-color: #475569 !important;
            border: 2px solid #64748b !important;
            border-radius: 8px !important;
            color: white !important;
            transition: border-color 0.3s ease !important;
        }
        .gr-textbox textarea:focus, .gr-textbox input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        .gr-textbox textarea::placeholder, .gr-textbox input::placeholder {
            color: #94a3b8 !important;
        }
        /* Dropdown dark theme */
        .gr-dropdown select {
            background-color: #475569 !important;
            border: 2px solid #64748b !important;
            color: white !important;
        }
        /* Number input dark theme */
        .gr-number input {
            background-color: #475569 !important;
            border: 2px solid #64748b !important;
            color: white !important;
        }
        /* Slider labels and values */
        .gr-slider .gr-slider-label {
            color: #e2e8f0 !important;
        }
        /* File upload area */
        .gr-file-upload {
            background-color: #475569 !important;
            border: 2px dashed #64748b !important;
            color: white !important;
        }
        /* Markdown content */
        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
            color: #e2e8f0 !important;
        }
        .markdown-content p, .markdown-content li {
            color: #cbd5e1 !important;
        }
        /* Overall background force override */
        body, .gradio-app, .app {
            background: #0f172a !important;
            color: white !important;
        }
        /* Preview container styles */
        .preview-container {
            margin: 15px 0;
            border-radius: 12px;
            overflow: hidden;
        }
        /* Real-time preview animation */
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
            100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
        }
        .gr-video[data-testid*="Latest Clip"] {
            animation: pulse 2s infinite;
            border: 2px solid #3b82f6;
        }
        
        /* Auto-play video styles */
        .auto-play-video {
            border: 3px solid #10b981 !important;
            border-radius: 12px !important;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.6) !important;
        }
        
        .auto-play-video video {
            border-radius: 8px !important;
        }
        
        /* Status display styles */
        .status-display {
            margin: 15px 0;
            border-radius: 12px;
            overflow: hidden;
        }
        """
    ) as demo:
        
        # Add logo and title
        with gr.Row(elem_classes=["main-header"]):
            with gr.Column(scale=1):
                if logo_base64:
                    gr.HTML(f"""
                    <div style="text-align: center; margin: 30px 0; color: white;">
                        <img src="data:image/png;base64,{logo_base64}" alt="Stable Video Infinity" 
                             style="max-height: 200px; max-width: 600px; margin: 0 auto 20px auto; 
                                    display: block; filter: brightness(1.2) contrast(1.1);">
                        <p style="font-size: 20px; color: #e2e8f0; margin-bottom: 12px; font-weight: 500;">
                            Upload an image and describe how you want it to move to generate a long video!
                        </p>
                        <p style="font-size: 18px; color: #cbd5e1; font-weight: 400;">
                            ✨ Supports long prompt stream for short-filming ✨
                        </p>
                    </div>
                    """)
                else:
                    gr.HTML("""
                    <div style="text-align: center; margin: 30px 0; color: white;">
                        <h1 style="color: #e2e8f0; margin-bottom: 20px; font-size: 2.5rem;">🎬 Stable Video Infinity</h1>
                        <p style="font-size: 20px; color: #e2e8f0; margin-bottom: 12px; font-weight: 500;">
                            Upload an image and describe how you want it to move to generate a long video!
                        </p>
                        <p style="font-size: 18px; color: #cbd5e1; font-weight: 400;">
                            ✨ Supports long prompt stream for short-filming ✨
                        </p>
                    </div>
                    """)
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-panel"]):
                gr.Markdown("## 🎮 Controls", elem_classes=["section-header"])
                
                # Mode selection
                with gr.Row():
                    model_mode = gr.Dropdown(
                        choices=[("SVI-Film (Long Prompt Stream)", "film"), ("SVI-Shot (Single Prompt)", "shot")],
                        value="film",
                        label="🎬 Select Model Mode",
                        info="Different modes are suitable for different types of video generation"
                    )
                    mode_switch_btn = gr.Button("🔄 Switch Mode", variant="secondary", size="sm")
                
                mode_status = gr.Textbox(
                    label="Mode Status",
                    value="Current Mode: SVI-Film (Cinema Style) - Suitable for cinematic narratives, storylines and long continuous shots",
                    interactive=False,
                    lines=2
                )
                
                gr.Markdown("---")
                
                # Model initialization
                init_btn = gr.Button("🚀 Initialize Models", variant="primary", size="lg", elem_classes=["custom-button"])
                init_status = gr.Textbox(
                    label="Model Status", 
                    value="Click 'Initialize Models' to start",
                    interactive=False
                )
                
                gr.Markdown("---")
                
                # Input image
                input_image = gr.Image(
                    label="📸 Input Image",
                    type="pil",
                    height=350,
                    value=demo_image  # Set default demo image
                )
                
                # Demo example button
                if demo_image or demo_prompts:
                    with gr.Row():
                        load_demo_btn = gr.Button(
                            "🎭 Load Demo Example",
                            variant="secondary",
                            size="sm"
                        )
                
                # Prompts (supports multiple)
                prompts = gr.Textbox(
                    label="✨ Prompts (Multi-format support)",
                    placeholder="""Multiple formats supported:

1. Multi-line format:
Line 1: First motion description
Line 2: Second motion description
Line 3: Third motion description

2. Python list format:
["First prompt", "Second prompt", "Third prompt"]

3. Comma-separated format:
First prompt, Second prompt, Third prompt""",
                    lines=8,
                    info="💡 Supports newlines, Python lists, or comma-separated prompts",
                    value=demo_prompts if demo_prompts else ""
                )
                
                # Example prompt selector
                streaming_example_dropdown = gr.Dropdown(
                    choices=streaming_examples,
                    label="🎞️ Streaming Prompt Examples",
                    value=None
                )
                
                # Negative prompt
                negative_prompt = gr.Textbox(
                    label="🚫 Negative Prompt",
                    value=default_negative_prompt,
                    lines=2
                )
                
                gr.Markdown("## ⚙️ Generation Settings", elem_classes=["section-header"])
                
                with gr.Row():
                    num_clips = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="🎞️ Number of Clips",
                        info="More clips = longer video"
                    )
                    num_steps = gr.Slider(
                        minimum=1, maximum=100, value=50, step=5,
                        label="🔧 Inference Steps",
                        info="Higher = better quality"
                    )
                
                with gr.Row():
                    cfg_scale_text = gr.Slider(
                        minimum=4.0, maximum=10.0, value=5.0, step=0.5,
                        label="🎯 CFG Scale",
                        info="How closely to follow prompts"
                    )
                
                with gr.Row():
                    seed = gr.Number(
                        label="🎲 Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🎬 Generate Streaming Video", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["custom-button"]
                )
                
            with gr.Column(scale=1, elem_classes=["output-panel"]):
                gr.Markdown("## 🎥 Output", elem_classes=["section-header"])
                
                # Combined video output window
                output_video = gr.Video(
                    label="🎬 Video Output (Shows each clip as it's generated, then final video)",
                    height=500,
                    autoplay=True,
                    show_download_button=True,
                    interactive=False,
                    elem_classes=["auto-play-video"],
                    format="mp4"  # Explicitly specify format
                )
                
                # Video status display
                video_status = gr.HTML(
                    value="<div style='text-align: center; color: #cbd5e1; font-size: 16px;'>📺 Ready to generate video</div>",
                    elem_classes=["status-display"]
                )
                
                # Usage instructions
                gr.Markdown("""
                ## 💡 How to Use
                
                <div class="example-box">
                <strong style="color: #e2e8f0;">🎬 Model Mode Selection:</strong><br>
                <span style="color: #cbd5e1;">
                <strong>📽️ SVI-Film (Cinema Style):</strong> Suitable for cinematic narratives, storylines and long continuous shots<br>
                - Supports multi-prompt continuous generation<br>
                - Each clip uses 5 motion frames<br>
                - Ideal for complex story scenarios<br><br>
                <strong>🎥 SVI-Shot (Camera Movement):</strong> Suitable for camera movements and dynamic shooting effects<br>
                - Repeats using the first prompt<br>
                - Each clip uses 1 motion frame<br>
                - Ideal for camera movements and shooting techniques
                </span>
                </div>
                
                <div class="example-box">
                <strong style="color: #e2e8f0;">🎯 Multi-format Prompt Support:</strong><br>
                <span style="color: #cbd5e1;">
                <strong>1. Multi-line format:</strong><br>
                Line 1: First motion<br>
                Line 2: Second motion<br><br>
                <strong>2. Python list format:</strong><br>
                ["First prompt", "Second prompt", "Third prompt"]<br><br>
                <strong>3. Comma-separated:</strong><br>
                First prompt, Second prompt, Third prompt
                </span>
                </div>
                
                
                ## 📋 Tips
                <div style="color: #cbd5e1;">
                - <strong style="color: #e2e8f0;">🔄 Mode Switch</strong>: Select mode and click "Switch Mode" to auto-load corresponding demo<br>
                - <strong style="color: #e2e8f0;">🎬 Instant Playback</strong>: Each clip plays immediately when generated and saved!<br>
                - <strong style="color: #e2e8f0;">📁 File Access</strong>: All generated MP4 files are saved and can be downloaded<br>
                - <strong style="color: #e2e8f0;">Demo Example</strong>: Click "Load Demo Example" to try the current mode's demo<br>
                - <strong style="color: #e2e8f0;">Input Image</strong>: Upload a clear image with good lighting<br>
                - <strong style="color: #e2e8f0;">Multiple Prompts</strong>: Use any format - list, lines, or commas<br>
                - <strong style="color: #e2e8f0;">Number of Clips</strong>: Will cycle through your prompts automatically<br>
                - <strong style="color: #e2e8f0;">CFG Scale</strong>: Higher values follow prompts more closely<br>
                - <strong style="color: #e2e8f0;">Motion Frames</strong>: Auto-set based on mode (Film=5, Shot=1)<br>
                - <strong style="color: #e2e8f0;">Python Lists</strong>: Perfect for long sequences like stories<br>
                - <strong style="color: #e2e8f0;">File Storage</strong>: All videos saved in ./videos/session_YYYYMMDD_HHMMSS/
                </div>
                """, elem_classes=["example-box"])
        
        # Event handling
        def update_prompt_from_streaming_example(example):
            return example if example else ""
        
        def load_demo_example():
            """Load demo example"""
            current_image, current_prompts = demo_app.get_demo_content_for_mode(demo_app.current_mode)
            return current_image, current_prompts if current_prompts else ""
        
        def switch_mode_and_load_demo(mode):
            """Switch mode and load corresponding demo"""
            # Switch mode
            status = demo_app.switch_model_mode(mode)
            
            # Get new mode demo content
            new_image, new_prompts = demo_app.get_demo_content_for_mode(mode)
            
            return (
                status,  # mode_status
                new_image,  # input_image
                new_prompts if new_prompts else "",  # prompts
            )
        
        # Mode switch event
        mode_switch_btn.click(
            fn=switch_mode_and_load_demo,
            inputs=[model_mode],
            outputs=[mode_status, input_image, prompts]
        )
        
        streaming_example_dropdown.change(
            fn=update_prompt_from_streaming_example,
            inputs=[streaming_example_dropdown],
            outputs=[prompts]
        )
        
        # Add demo loading event
        if demo_image or demo_prompts:
            load_demo_btn.click(
                fn=load_demo_example,
                outputs=[input_image, prompts]
            )
        
        init_btn.click(
            fn=demo_app.initialize_models,
            outputs=[init_status]
        )
        
        def generate_video_with_preview(input_image, prompts, negative_prompt, num_clips, num_steps, cfg_scale_text, seed):
            """Wrapper generation function to handle preview display"""
            print("🚀 Starting video generation with debug info...")
            current_clip_count = 0
            
            # Initial state
            yield None, "<div style='text-align: center; color: #cbd5e1; font-size: 16px;'>🚀 Starting generation...</div>"
            
            try:
                # Use generator for real-time preview updates
                for clip_preview, final_video in demo_app.generate_video(
                    input_image, prompts, negative_prompt, num_clips, num_steps, 
                    cfg_scale_text, seed
                ):
                    print(f"🔍 DEBUG: Generator yielded - clip_preview={clip_preview}, final_video={final_video}")
                    
                    if clip_preview and not final_video:
                        # Update output video for each completed clip in real-time
                        current_clip_count += 1
                        print(f"🎬 Processing clip {current_clip_count}: {clip_preview}")
                        
                        # Verify file exists
                        if not os.path.exists(clip_preview):
                            print(f"❌ ERROR: Clip file does not exist: {clip_preview}")
                            error_html = f"""
                            <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #ef4444, #dc2626); 
                                        border-radius: 10px; color: white; font-weight: bold; margin: 10px 0;'>
                                <div style='font-size: 18px; margin-bottom: 10px;'>
                                    ❌ Error: Clip {current_clip_count} file not found
                                </div>
                                <div style='font-size: 14px; opacity: 0.9;'>
                                    File path: {clip_preview}
                                </div>
                            </div>
                            """
                            yield None, error_html
                            continue
                            
                        file_size = os.path.getsize(clip_preview)
                        print(f"✅ Clip file verified: {clip_preview} (size: {file_size} bytes)")
                        
                        if file_size == 0:
                            print(f"❌ ERROR: Clip file is empty: {clip_preview}")
                            continue
                        
                        status_html = f"""
                        <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #10b981, #059669); 
                                    border-radius: 10px; color: white; font-weight: bold; margin: 10px 0;'>
                            <div style='font-size: 18px; margin-bottom: 10px;'>
                                🎬 Clip {current_clip_count}/{num_clips} Generated & Playing!
                            </div>
                            <div style='background: rgba(255,255,255,0.2); border-radius: 20px; height: 8px; margin: 10px 0;'>
                                <div style='background: white; height: 8px; border-radius: 20px; width: {(current_clip_count/num_clips)*100}%;'></div>
                            </div>
                            <div style='font-size: 14px; opacity: 0.9;'>
                                📁 File: {os.path.basename(clip_preview)} ({file_size:,} bytes)
                            </div>
                        </div>
                        """
                        
                        print(f"🎯 About to yield clip to frontend: {clip_preview}")
                        
                        # Add small delay to ensure file write completion
                        import time
                        time.sleep(0.5)
                        
                        yield clip_preview, status_html
                        print(f"✅ Successfully yielded clip {current_clip_count}")
                        
                    elif final_video:
                        # Final result, show complete video
                        print(f"🎞️ Processing final video: {final_video}")
                        
                        # Verify file exists
                        if not os.path.exists(final_video):
                            print(f"❌ ERROR: Final video file does not exist: {final_video}")
                            continue
                            
                        file_size = os.path.getsize(final_video)
                        print(f"✅ Final video file verified: {final_video} (size: {file_size} bytes)")
                        
                        final_status_html = f"""
                        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #3b82f6, #1e40af); 
                                    border-radius: 10px; color: white; font-weight: bold; margin: 10px 0;'>
                            <div style='font-size: 20px; margin-bottom: 15px;'>
                                🎉 Final Video Complete!
                            </div>
                            <div style='background: rgba(255,255,255,0.2); border-radius: 20px; height: 10px; margin: 10px 0;'>
                                <div style='background: white; height: 10px; border-radius: 20px; width: 100%;'></div>
                            </div>
                            <div style='font-size: 16px; opacity: 0.9;'>
                                📁 File: {os.path.basename(final_video)} ({file_size:,} bytes) | 🎬 {num_clips} clips
                            </div>
                        </div>
                        """
                        
                        print(f"🎯 About to yield final video to frontend: {final_video}")
                        
                        # Add small delay to ensure file write completion
                        import time
                        time.sleep(0.5)
                        
                        yield final_video, final_status_html
                        print(f"✅ Successfully yielded final video")
                        break
                    else:
                        print("⚠️ Generator yielded None values, continuing...")
                        yield None, "<div style='text-align: center; color: #cbd5e1; font-size: 16px;'>⏳ Processing...</div>"
                        
            except Exception as e:
                print(f"❌ ERROR in generate_video_with_preview: {str(e)}")
                import traceback
                traceback.print_exc()
                error_html = f"""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #ef4444, #dc2626); 
                            border-radius: 10px; color: white; font-weight: bold; margin: 10px 0;'>
                    <div style='font-size: 18px; margin-bottom: 10px;'>
                        ❌ Generation Error
                    </div>
                    <div style='font-size: 14px; opacity: 0.9;'>
                        {str(e)}
                    </div>
                </div>
                """
                yield None, error_html
        
        generate_btn.click(
            fn=generate_video_with_preview,
            inputs=[
                input_image, prompts, negative_prompt,
                num_clips, num_steps, cfg_scale_text,
                seed
            ],
            outputs=[output_video, video_status],
            show_progress=True,
            queue=True
        )
    
    return demo.queue()

def get_model_config():
    """Get model configuration"""
    parser = argparse.ArgumentParser(description="SVI Gradio Demo")
    parser.add_argument(
        "--dit_root",
        default='./weights/Wan2.1-I2V-14B-480P/',
        type=str,
        help="Root directory of the Wan2.1-I2 model.",
    )
    parser.add_argument(
        "--extra_module_root",
        default="weights/Stable-Video-Infinity/version-1.0/svi-film.safetensors",
        type=str,
        help="Root directory of extra modules (LoRA).",
    )
    parser.add_argument(
        "--train_architecture",
        default='lora',
        type=str,
        help="Training architecture."
    )
    parser.add_argument(
        "--lora_alpha",
        default=1.0,
        type=float,
        help="LoRA alpha value."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="Host to bind the server to."
    )
    parser.add_argument(
        "--port",
        default=7860,
        type=int,
        help="Port to bind the server to."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=True,
        help="Enable Gradio sharing."
    )
    
    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    config = get_model_config()
    
    print("=" * 80)
    print("STABLE VIDEO INFINITY - GRADIO DEMO")
    print("=" * 80)
    print("Model Configuration:")
    for key, value in config.items():
        if key not in ['host', 'port', 'share']:
            print(f"  {key}: {value}")
    print("=" * 80)
    
    demo = create_demo(config)
    
    # Configure static file paths to allow Gradio access to generated videos
    videos_dir = os.path.join(os.getcwd(), "videos")
    if os.path.exists(videos_dir):
        print(f"📁 Videos directory: {videos_dir}")
    else:
        print("📁 Creating videos directory...")
        os.makedirs(videos_dir, exist_ok=True)
    
    demo.launch(
        server_name=config['host'],
        server_port=config['port'],
        share=config['share'],
        show_error=True,
        debug=True,
        # Allow Gradio to access videos directory
        allowed_paths=[videos_dir]
    )