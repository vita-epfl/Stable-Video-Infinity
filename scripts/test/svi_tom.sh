#!/bin/bash

python test_svi.py \
--output videos/svi_tom/ \
--dit_root ./weights/Wan2.1-I2V-14B-480P/ \
--ref_pad_num 0 \
--cfg_scale_text 5.0 \
--num_motion_frames 1 \
--ref_image_path data/toy_test/tom/frame.png \
--prompt_path data/toy_test/tom/prompt.txt \
--extra_module_root weights/Stable-Video-Infinity/version-1.0/svi-tom.safetensors 
