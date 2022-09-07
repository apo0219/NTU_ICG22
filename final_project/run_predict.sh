#!/bin/bash
python3 train.py \
    --style_dir style \
    --output_dir predict \
    --gpu 1 \
    --batch_size 1 \
    --from_checkpoint model.pth \
    --do_predict \
    --predict_dir yahcreeper_content
