#!/bin/bash
# Activate environment
source /home/chenx/miniconda3/bin/activate neurostorm

# Set visible GPUs (change 0,1 to the GPU IDs you want to use)
export CUDA_VISIBLE_DEVICES=0,1
# export PYTHONNOUSERSITE=1

# Run training
python -m torch.distributed.run --nproc_per_node=2 train_mae.py \
    --train_list /public/home/wangmo/swinunet/pretrain/train.txt \
    --val_list /public/home/wangmo/swinunet/pretrain/val.txt \
    --output_dir checkpoints_mae \
    --time_window 480 \
    --batch_size 1 \
    --epochs 10 \
    --mask_ratio 0.75 \
    --feature_size 48 \
    --use_checkpoint
