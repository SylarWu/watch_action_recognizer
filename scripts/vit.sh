#!/bin/bash

cd ..

dataset_path="/data/wuxilei/watch_action_recognizer/"
check_point_path="/data/wuxilei/watch_action_recognizer/log/"

cuda=0

for model_name in vit_es_7 vit_es_14 vit_es_16 vit_es_32
do
  for preprocess_strategy in normal_0 user_1 shuffle_0 shuffle_1 shuffle_2
  do
    CUDA_VISIBLE_DEVICES=${cuda} python training.py --dataset_path ${dataset_path} \
    --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
    --train_batch_size 64 --eval_batch_size 64 --num_epoch 100 --opt_method "adamw" --lr_rate 1e-4 --weight_decay 1e-4 \
    --save_epoch 50 --eval_epoch 1 \
    --check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
    --model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls"

    CUDA_VISIBLE_DEVICES=${cuda} python testing.py --dataset_path ${dataset_path} \
    --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
    --check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
    --model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls" \
    --test_batch_size 64
  done
done
