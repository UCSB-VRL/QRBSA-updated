#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python -m main.py \
    --input_dir '/data/home/umang/Materials/Materials_data_mount/materials/fz_reduced/Open_718_Z_Upsampling' \
    --hr_data_dir 'Train/HR_Images/preprocessed_imgs_1D' \
    --val_lr_data_dir 'Val/LR_Images/X4/preprocessed_imgs_1D' \
    --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_1D' \
    --syms_np_path './model/reynolds_utils/fcc_symmetry_group.npy' \
    --syms_inv_np_path './model/reynolds_utils/fcc_symmetry_group_inv.npy' \
    --model 'reynolds_qsr' \
    --lr 2e-6 \
    --weight_decay 0 \
    --n_resblocks 0 \
    --n_resgroups 10 \
    --n_feats 64 \
    --n_channels 4 \
    --n_scale 4 \
    --save 'equivariant_reynolds' \
    --loss '1*MisOrientation' \
    --dist_type "minimum_angle_transformation"\
    --patch_size 64 \
    --batch_size 1 \
    --scale 4 \
    --val_freq 5 \
    --save_model_freq 100 \
    --syms_type 'FCC' \
    --syms_req \
    --epoch 1000 \
    --include_consistency_loss False \
    --prog_patch 