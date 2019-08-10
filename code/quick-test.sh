#!/usr/bin/env bash

python MVSNet_main.py --mode test --regularize 3DCNN \
    --load /data3/lyf/MVSNET/mvsnet_train_log/192-1.06-b1-dtu_training-unet-0514-2302-no-refine/model-33870 \
    --data /data3/lyf/MVSNET/mvsnet_test/standard_dataset/illumination_part6_adaptives/ \
    --out /data3/lyf/MVSNET/mvsnet_test/results/illumination_part6_adaptives/ \
    --batch 1 \
    --max_d 184 --view_num 5 --max_w 1152 --max_h 864 --interval_scale 1 --gpu 3 --threshold 0.5

