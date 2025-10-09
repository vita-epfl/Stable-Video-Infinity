#!/bin/bash

python test_svi_dance.py \
--output videos/svi_dance \
--dit_root ./weights/Wan2.1-I2V-14B-480P/ \
--cfg_scale_text 2.0 \
--ref_pad_num -1 \
--num_clips 10 \
--num_steps 50 \
--num_motion_frames 1 \
--image_path data/toy_test/dance/image.png \
--pose_path data/toy_test/dance/pose.mp4 \
--extra_module_root weights/Stable-Video-Infinity/version-1.0/svi-dance.safetensors 