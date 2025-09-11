#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python -m main.py \
    --input_dir '/data/umang/materials/fz_reduced/Open_718_Z_Upsampling' \
    --hr_data_dir 'Train/HR_Images/preprocessed_imgs_1D' \
    --val_lr_data_dir 'Val/LR_Images/X4/preprocessed_imgs_1D' \
    --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_1D' \
    --model 'reynolds_qsr' \
    --lr 2e-7 \
    --weight_decay 0 \
    --n_resblocks 0 \
    --n_resgroups 0 \
    --n_feats 128 \
    --n_colors 4 \
    --save 'reynolds_2D_All_positive_min_angle_include_L1'\
    --loss '1*MisOrientation' \
    --dist_type "minimum_angle_transformation"\
    --patch_size 64 \
    --batch_size 1 \
    --scale 4 \
    --val_freq 5 \
    --save_model_freq 50 \
    --syms_type 'FCC' \
    --syms_req \
    --epoch 1000 \
    --include_consistency_loss False \
    --prog_patch 