#!/bin/sh

python test.py \
    --input_dir '/data/home/umang/Materials/data/fz_reduced/Open_718_Z_Upsampling' \
    --model 'qrbsa_1d' \
    --patch_size 128 \
    --n_resblocks 10 \
    --n_resgroups 10 \
    --n_feats 256 \
    --n_colors 4 \
    --save 'qrbsa_consistency_includes_symmetry' \
    --resume -1 \
    --model_to_load 'model_best' \
    --test_dataset_type 'Test' \
    --test_only \
    --dist_type 'valid_symmHR_expand' \
    --scale 4 \
    --syms_type 'FCC'
