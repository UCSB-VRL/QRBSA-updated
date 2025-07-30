#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python test.py \
    --input_dir '/data/umang/materials/fz_reduced/Open_718_Z_Upsampling' \
    --model 'so3reynolds_qrbsa_1d' \
    --patch_size 256 \
    --n_resblocks 10 \
    --n_resgroups 10 \
    --n_feats 128 \
    --n_colors 4 \
    --save 'reynolds_wrapper_original_qrbsa_1d' \
    --resume -1 \
    --model_to_load 'model_best' \
    --test_dataset_type 'Test' \
    --test_only \
    --dist_type 'minimum_angle_transformation' \
    --scale 4 \
    --syms_type 'FCC'