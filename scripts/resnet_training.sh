#!/bin/bash

dataset_path="/data/wuxilei/watch_action_recognizer/"
check_point_path="/data/wuxilei/watch_action_recognizer/log/"


python ../training.py --dataset_path ${dataset_path} --preprocess_strategy "shuffle" --seq_len 224 \
--train_batch_size 64 --test_batch_size 64 --num_epoch 1000 --opt_method "adamw" --lr_rate 1e-4 --weight_decay 1e-4 \
--save_epoch 50 --eval_epoch 1 --check_point_path ${check_point_path} --use_gpu true --gpu_device "0" \
--model_name "resnet101" --head_name "span_cls" --strategy_name "span_cls" --n_classes 18
