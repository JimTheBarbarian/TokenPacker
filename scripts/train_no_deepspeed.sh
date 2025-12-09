#!/bin/bash


# Single GPU training
#python llava/train/train_no_deepspeed.py \
#    --model_name_or_path /path/to/base/model \
#    --version qwen \
#    --data_path /path/to/training/data.json \
#    --image_folder /path/to/images \
#    --vision_tower openai/clip-vit-large-patch14-336 \
#    --mm_projector_type mlp2x_gelu \
#    --mm_vision_select_layer -2 \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --image_aspect_ratio pad \
#    --bf16 True \
#    --output_dir ./checkpoints/llava-training \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 4 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 500 \
#    --save_total_limit 2 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 2048 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 4 \
#    --lazy_preprocess True \
#    --report_to wandb

# Multi-GPU training with PyTorch DDP
 torchrun --nproc_per_node=4 --master_port=25001 llava/train/train_no_deepspeed.py \
    --model_name_or_path Qwen3/Qwen3-0.6b \
    --version plain \
    --data_path ../../../../../ssss/Datasets/llava-pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ../../../../../ssss/Datasets/llava-pretrain/llava_pretrain_558k \
    --vision_tower google/siglip2-base-patch16-384 \
    --mm_projector_type tokenpacker \
    --scale_factor 2 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ../../../../../ssss/Datasets/llave-pretrain/checkpoints/llava-tokenpacker-pretrain/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    #--fp32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "none"

