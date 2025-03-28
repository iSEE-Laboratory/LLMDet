#!/bin/bash

WANDB__SERVICE_WAIT=300 deepspeed train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /home/shenghao2/myLMM/huggingface/llava-onevision-qwen2-7b-ov \
    --version qwen_1_5 \
    --data_path /mnt/Datasets/LLaVA/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder /mnt/Datasets/LLaVA/LLaVA-Instruct-150K/data \
    --vision_tower siglip \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --repeat_num 1 \
    --load_full_model True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/llava-onevision-qwen2-7b-ov-lora-2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
