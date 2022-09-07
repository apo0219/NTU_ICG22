#!/bin/bash
python3 train.py \
    --epoch 20 \
    --style_dir style \
    --output_dir paper_model \
    --train_size 10000 \
    --test_size 1000 \
    --batch_size 4 \
    --save_epoch 2 \
    --eval_epoch 2 \
    --gpu 0 \
    -t 2 \
    --alpha 1 \
    --beta 1000000 \
    --gamma 0.00001 \
    --learning_rate 0.001