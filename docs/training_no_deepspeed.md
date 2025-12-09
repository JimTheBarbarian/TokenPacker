# Training Without DeepSpeed

This directory contains a modified training script (`train_no_deepspeed.py`) that removes all DeepSpeed dependencies and uses native PyTorch distributed training instead.

## Key Changes from Original `train.py`

### Removed Features
1. **DeepSpeed Integration**: All DeepSpeed-specific code has been removed
2. **LoRA Support**: Removed all LoRA/PEFT-related code (as requested)
3. **Quantization**: Removed 4-bit and 8-bit quantization support
4. **Zero-3 Functions**: Removed `maybe_zero_3()` and related DeepSpeed checkpoint gathering

### Simplified Features
1. **Distributed Training**: Now uses only PyTorch's native DDP (DistributedDataParallel)
2. **Model Saving**: Simplified checkpoint saving without DeepSpeed-specific logic
3. **Trainer Class**: Created `SimpleLLaVATrainer` that extends HuggingFace Trainer without DeepSpeed dependencies

### Preserved Features
1. **Vision Tower Integration**: Full support for multimodal training
2. **MM Projector**: Support for training only the multimodal projector (`tune_mm_mlp_adapter`)
3. **Custom Learning Rates**: Support for different learning rates for mm_projector
4. **Gradient Checkpointing**: Memory-efficient training
5. **All Preprocessing**: Image processing, conversation templates, and data collation

## Usage

### Single GPU Training

```bash
python llava/train/train_no_deepspeed.py \
    --model_name_or_path /path/to/base/model \
    --version qwen \
    --data_path /path/to/training/data.json \
    --image_folder /path/to/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --bf16 True \
    --output_dir ./checkpoints/llava-training \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --model_max_length 2048 \
    --gradient_checkpointing True
```

### Multi-GPU Training with PyTorch DDP

```bash
torchrun --nproc_per_node=4 --master_port=25001 llava/train/train_no_deepspeed.py \
    --model_name_or_path /path/to/base/model \
    --version qwen \
    --data_path /path/to/training/data.json \
    --image_folder /path/to/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --bf16 True \
    --output_dir ./checkpoints/llava-training \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --model_max_length 2048 \
    --gradient_checkpointing True
```

Replace `--nproc_per_node=4` with the number of GPUs you want to use.

## Important Arguments

### Model Arguments
- `--model_name_or_path`: Path to the base language model
- `--version`: Conversation template version (e.g., "qwen", "v1", "llama_2")
- `--vision_tower`: Vision encoder model
- `--mm_projector_type`: Type of multimodal projector (e.g., "mlp2x_gelu")
- `--tune_mm_mlp_adapter`: Only train the multimodal projector (freeze LLM)
- `--freeze_backbone`: Freeze the language model backbone

### Data Arguments
- `--data_path`: Path to training data JSON file
- `--image_folder`: Directory containing training images
- `--lazy_preprocess`: Load and process data on-the-fly (memory efficient)
- `--image_aspect_ratio`: How to handle image aspect ratios ("pad", "square")

### Training Arguments
- `--bf16`: Use bfloat16 precision (recommended for modern GPUs)
- `--fp16`: Use float16 precision (alternative to bf16)
- `--gradient_checkpointing`: Enable gradient checkpointing for memory efficiency
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Accumulate gradients over multiple steps
- `--learning_rate`: Learning rate
- `--mm_projector_lr`: Separate learning rate for mm_projector (optional)
- `--model_max_length`: Maximum sequence length
- `--num_train_epochs`: Number of training epochs

## Removed Training Arguments

The following arguments from the original script are **no longer supported**:

- `--bits`: Quantization bits (4/8/16)
- `--lora_enable`: Enable LoRA
- `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA hyperparameters
- `--double_quant`, `--quant_type`: Quantization settings
- `--group_by_modality_length`: Custom sampler for grouping by modality (simplified)

## Memory Optimization Tips

Without DeepSpeed's Zero optimization, you'll need to manage memory more carefully:

1. **Use Gradient Checkpointing**: `--gradient_checkpointing True`
2. **Reduce Batch Size**: Lower `--per_device_train_batch_size`
3. **Increase Gradient Accumulation**: `--gradient_accumulation_steps` to maintain effective batch size
4. **Use Mixed Precision**: `--bf16 True` (or `--fp16 True`)
5. **Reduce Sequence Length**: Lower `--model_max_length` if possible
6. **Freeze Some Parameters**: Use `--tune_mm_mlp_adapter True` to only train projector

## Example: Training Only MM Projector (Pretraining)

```bash
python llava/train/train_no_deepspeed.py \
    --model_name_or_path Qwen/Qwen2-7B \
    --version qwen \
    --data_path ./data/pretrain.json \
    --image_folder ./data/pretrain_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
```

## Example: Full Model Fine-tuning

```bash
torchrun --nproc_per_node=2 llava/train/train_no_deepspeed.py \
    --model_name_or_path ./checkpoints/llava-pretrain \
    --version qwen \
    --data_path ./data/finetune.json \
    --image_folder ./data/finetune_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
```

## Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing`
4. Reduce `model_max_length`
5. Use `tune_mm_mlp_adapter` mode

### Slow Training
1. Increase `dataloader_num_workers`
2. Use `lazy_preprocess True`
3. Ensure `bf16` or `fp16` is enabled
4. Check that all GPUs are being utilized with `torchrun`

### Model Not Found
Ensure the `model_name_or_path` points to a valid model checkpoint that includes:
- Model weights
- Configuration files
- Tokenizer files

## Original Training Script

The original DeepSpeed-enabled training script remains available at `llava/train/train.py` and can be used if you gain access to DeepSpeed in the future.
