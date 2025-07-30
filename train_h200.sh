#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python -m main.py \
    --input_dir '/data/umang/materials/fz_reduced/Open_718_Z_Upsampling' \
    --hr_data_dir 'Train/HR_Images/preprocessed_imgs_1D' \
    --val_lr_data_dir 'Val/LR_Images/X4/preprocessed_imgs_1D' \
    --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_1D' \
    --model 'so3reynolds_qrbsa_1d' \
    --lr 2e-7 \
    --weight_decay 0 \
    --n_resblocks 10 \
    --n_resgroups 10 \
    --n_feats 128 \
    --n_colors 4 \
    --save 'reynolds_wrapper_original_qrbsa_1d'\
    --loss '1*MisOrientation' \
    --dist_type "minimum_angle_transformation"\
    --patch_size 64 \
    --batch_size 1 \
    --scale 4 \
    --val_freq 1 \
    --save_model_freq 100 \
    --syms_type 'FCC' \
    --syms_req \
    --epoch 1000 \
    --include_consistency_loss False \
    --prog_patch 