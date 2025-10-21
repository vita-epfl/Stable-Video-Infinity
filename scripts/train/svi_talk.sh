python train_svi_talk.py \
--learning_rate 1e-4 \
--lora_rank 128 \
--lora_alpha 128 \
--dataset_path  ./data/toy_train/svi-talk/preprocessed/ \
--dit_path "./weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors,./weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors,./weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors,./weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors,./weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors,./weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors,./weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors,./weights/Wan2.1-I2V-14B-480P/multitalk.safetensors" \
--vae_path "./weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
--text_encoder_path "./weights/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
--image_encoder_path "./weights/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
--max_epochs 10 \
--train_architecture lora \
--use_gradient_checkpointing \
--use_gradient_checkpointing_offload \
--training_strategy "deepspeed_stage_2" \
--output_path "./experiments/train/svi-talk" \
--use_error_recycling \
--error_buffer_k 500 \
--y_error_num 3 \
--num_motion_frames 1 \
--buffer_warmup_iter 50 \
--buffer_replacement_strategy l2_batch \
--y_error_sample_from_all_grids \
--num_grids 50 \
--ref_pad_num -1 \
--noise_prob 0.01 \
--y_prob 0.9 \
--latent_prob 0.9 \
--clean_prob 0.2 \
--clean_buffer_update_prob 0.1 \
--exp_prefix train-svi-talk


# To check:
# enable_multitalk is automatically set to True in terminal output:  "This model is initialized with extra kwargs: {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'enable_multitalk': True}"

