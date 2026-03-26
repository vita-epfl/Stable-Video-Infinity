#!/bin/bash

python test_svi.py \
--output videos/svi_2.0/ \
--dit_root ./weights/Wan2.1-I2V-14B-480P/ \
--ref_pad_num -1 \
--cfg_scale_text 5.0 \
--num_motion_frames 5 \
--num_clips 999 \
--ref_image_path data/toy_test/svi_2.0/frame.jpg \
--prompt_path data/toy_test/svi_2.0/prompt.txt \
--prompt_repeat_times 2 \
--extra_module_root weights/Stable-Video-Infinity/version-2.0/SVI_Wan2.1-I2V-14B_lora_v2.0.safetensors
