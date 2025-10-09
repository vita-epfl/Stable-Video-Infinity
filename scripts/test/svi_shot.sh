#!/bin/bash

python test_svi.py \
--output videos/svi_shot/ \
--dit_root ./weights/Wan2.1-I2V-14B-480P/ \
--ref_pad_num -1 \
--cfg_scale_text 5.0 \
--num_motion_frames 1 \
--ref_image_path data/toy_test/shot/frame.jpg \
--prompt_path data/toy_test/shot/prompt.txt \
--use_first_prompt_only \
--extra_module_root weights/Stable-Video-Infinity/version-1.0/svi-shot.safetensors 
