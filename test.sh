#!/bin/sh

CUDA_VISIBLE_DEVICES=6 python test.py \
    --input_dir '/data/home/umang/Materials/Materials_data_mount/fz_reduced/Open_718_Z_Upsampling' \
    --model 'qrbsa_1d' \
    --patch_size 256 \
    --n_resblocks 10 \
    --n_resgroups 10 \
    --n_feats 256 \
    --n_colors 4 \
    --save 'transposeConv_1pass_bs4_outerprod' \
    --resume -1 \
    --model_to_load 'model_best' \
    --test_dataset_type 'Test' \
    --test_only \
    --dist_type 'minimum_angle_transformation' \
    --scale 4 \
    --syms_type 'FCC'
