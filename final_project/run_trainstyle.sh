#!/bin/bash
python3 train.py \
    --epoch 10 \
    --style_dir style \
    --output_dir my_style_13 \
    --train_size 10000 \
    --test_size 1000 \
    --batch_size 2 \
    --save_epoch 1 \
    --gpu 1 \
    --alpha 1 \
    --beta 300000 \
    --gamma 0.00001 \
    --learning_rate 0.0005 \
    --from_check_no_optimizer model.pth \
    --do_my_train_do_my_style \
    --train_style_image style/style_13.jpg
