#!/bin/bash
# Activate environment
source /home/chenx/miniconda3/bin/activate neurostorm

# Set visible GPUs (change 0,1 to the GPU IDs you want to use)
export CUDA_VISIBLE_DEVICES=0,1
# export PYTHONNOUSERSITE=1

# Run training
python -m torch.distributed.run --nproc_per_node=2 code/swin_unet_ccbd/train_mae.py \
    --data_root /public/home/wangmo/BIDS_results/20251114_npz/results \
    --output_dir code/swin_unet_ccbd/checkpoints_mae \
    --time_window 480 \
    --batch_size 1 \
    --epochs 10 \
    --mask_ratio 0.75 \
    --feature_size 48 \
    --use_checkpoint
