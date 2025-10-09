#!/bin/bash

python test_svi_talk.py \
--output videos/svi_talk/ \
--dit_root ./weights/Wan2.1-I2V-14B-480P/ \
--ref_pad_num -1 \
--num_clips 50 \
--num_motion_frames 1 \
--ref_image_path data/toy_test/talk/obama.png \
--audio_path data/toy_test/talk/obama_5min.wav \
--extra_module_root weights/Stable-Video-Infinity/version-1.0/svi-talk.safetensors \