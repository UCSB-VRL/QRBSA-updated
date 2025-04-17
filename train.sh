#!/bin/sh

python -m cProfile main.py \
    --input_dir '/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718_Z_Upsampling' \
    --hr_data_dir 'Train/HR_Images/preprocessed_imgs_1D' \
    --val_lr_data_dir 'Val/LR_Images/X4/preprocessed_imgs_1D' \
    --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_1D' \
    --model 'qrbsa_1d' \
    --lr 2e-6 \
    --weight_decay 0 \
    --n_resblocks 10 \
    --n_resgroups 10 \
    --n_feats 4 \
    --n_colors 4 \
    --save '/qrbsa_validHR_expand' \
    --loss '1*MisOrientation' \
    --dist_type 'valid_symmHR_expand' \
    --patch_size 64 \
    --batch_size 4 \
    --scale 4 \
    --val_freq 5 \
    --save_model_freq 1 \
    --syms_type 'FCC' \
    --syms_req \
    --epoch 500 \
    --include_consistency_loss False \
    --prog_patch \
    
 