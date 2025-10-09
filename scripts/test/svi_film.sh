#!/bin/bash

python test_svi.py \
--output videos/svi_film/ \
--dit_root ./weights/Wan2.1-I2V-14B-480P/ \
--ref_pad_num 0 \
--cfg_scale_text 5.0 \
--num_motion_frames 5 \
--ref_image_path data/toy_test/film/frame.jpg \
--prompt_path data/toy_test/film/prompt.txt \
--extra_module_root weights/Stable-Video-Infinity/version-1.0/svi-film.safetensors 
