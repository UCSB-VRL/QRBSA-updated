#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python -m main.py \
    --input_dir '/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718_Z_Upsampling' \
    --hr_data_dir 'Train/HR_Images/preprocessed_imgs_1D' \
    --val_lr_data_dir 'Val/LR_Images/X4/preprocessed_imgs_1D' \
    --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_1D' \
    --syms_np_path './model/reynolds_utils/fcc_symmetry_group.npy' \
    --syms_inv_np_path './model/reynolds_utils/fcc_symmetry_group_inv.npy' \
    --upsample_2d \
    --model 'reynolds_qsr' \
    --lr 2e-6 \
    --weight_decay 0 \
    --n_resblocks 0 \
    --n_resgroups 10 \
    --n_feats 64 \
    --n_channels 4 \
    --save 'reynolds_1layer_pixel_shuffle' \
    --loss '1*MisOrientation' \
    --dist_type "minimum_angle_transformation"\
    --patch_size 64 \
    --batch_size 10 \
    --scale 4 \
    --val_freq 5 \
    --save_model_freq 100 \
    --syms_type 'FCC' \
    --syms_req \
    --epoch 1000 \
    --include_consistency_loss False \
    --prog_patch 


# "--dist_type", "minimum_angle_transformation",
# "--input_dir", "/data/home/umang//Materials/Materials_data_mount/fz_reduced/Open_718_Z_Upsampling",
# "--hr_data_dir", "Train/HR_Images/preprocessed_imgs_1D",
# "--val_lr_data_dir", "Val/LR_Images/X4/preprocessed_imgs_1D",
# "--val_hr_data_dir", "Val/HR_Images/preprocessed_imgs_1D",
# "--syms_np_path", "./model/reynolds_utils/fcc_symmetry_group.npy",
# "--syms_inv_np_path", "./model/reynolds_utils/fcc_symmetry_group_inv.npy",
# "--upsample_2d",
# "--model", "reynolds_qsr",
# "--n_resblocks", "0",
# "--n_resgroups", "10",
# "--n_feats", "64",
# "--n_channels", "4",
# "--save", "DEBUG_SESSION",
# "--loss", "1*MisOrientation",
# "--patch_size", "64",
# "--batch_size", "1",
# "--scale", "4",
# "--val_freq", "1",
# "--save_model_freq", "100",
# "--syms_type", "FCC",
# "--syms_req",
# "--prog_patch",
# "--lr", "2e-5",
# "--include_consistency_loss", "False", 
# "--weight_decay", "1e-5",
