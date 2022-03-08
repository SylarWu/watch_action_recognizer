#!/bin/bash

cd ..

dataset_path="/data/wuxilei/watch_action_recognizer/"
check_point_path="/data/wuxilei/watch_action_recognizer/log/"

model_name="vit-b-16"
preprocess_strategy="user_1"

python training.py --dataset_path ${dataset_path} --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
--train_batch_size 64 --eval_batch_size 64 --num_epoch 100 --opt_method "adamw" --lr_rate 1e-4 --weight_decay 1e-4 \
--save_epoch 50 --eval_epoch 1 --check_point_path ${check_point_path} --use_gpu true --gpu_device "0" \
--model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls" --test_batch_size 64 --n_classes 18

python testing.py --dataset_path ${dataset_path} --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
--train_batch_size 64 --eval_batch_size 64 --num_epoch 100 --opt_method "adamw" --lr_rate 1e-4 --weight_decay 1e-4 \
--save_epoch 50 --eval_epoch 1 --check_point_path ${check_point_path} --use_gpu true --gpu_device "0" \
--model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls" --test_batch_size 64 --n_classes 18