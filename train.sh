#!/bin/sh

python -m cProfile main.py \
    --input_dir '/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718_Z_Upsampling' \
    --hr_data_dir 'Train/HR_Images/preprocessed_imgs_1D' \
    --val_lr_data_dir 'Val/LR_Images/X4/preprocessed_imgs_1D' \
    --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_1D' \
    --model 'qrbsa_1d' \
    --lr 2e-5 \
    --weight_decay 1e-5 \
    --n_resblocks 10 \
    --n_resgroups 10 \
    --n_feats 256 \
    --n_colors 4 \
    --save '/qrbsa_loss_mean_consistency_pick_max' \
    --loss '1*MisOrientation' \
    --dist_type 'minimum_angle_transformation' \
    --patch_size 64 \
    --batch_size 4 \
    --scale 4 \
    --val_freq 2 \
    --save_model_freq 1 \
    --syms_type 'FCC' \
    --syms_req \
    --epoch 500 \
    --include_consistency_loss True \
    --prog_patch
 