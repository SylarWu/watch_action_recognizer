#!/bin/bash

cd ..

dataset_path="/data/wuxilei/watch_action_recognizer/"
check_point_path="/data/wuxilei/watch_action_recognizer/log/"

cuda=0

for model_name in resnet18 resnet34 resnet50 resnet101
do

  for ((i = 0; i < 5; ++i));
  do
    preprocess_strategy=normal_${i}
    echo "===============================================基于normal_${i}策略对模型进行训练==============================================="
    CUDA_VISIBLE_DEVICES=${cuda} python training.py --dataset_path ${dataset_path} \
    --preprocess_method ${preprocess_method} --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
    --train_batch_size 256 --eval_batch_size 256 --num_epoch 1000 --opt_method "adamw" \
    --lr_rate 5e-4 --lr_rate_adjust_epoch 50 --lr_rate_adjust_factor 0.5 --weight_decay 1e-4 \
    --save_epoch 50 --eval_epoch 1 --patience 20 \
    --check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
    --model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls"
    echo "===============================================基于normal_${i}策略对模型进行测试==============================================="
    CUDA_VISIBLE_DEVICES=${cuda} python testing.py --dataset_path ${dataset_path} \
    --preprocess_method ${preprocess_method} --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
    --check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
    --model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls" \
    --test_batch_size 256
  done

  for ((i = 1; i <= 10; ++i));
  do
    preprocess_strategy=user_${i}
    echo "===============================================基于user_${i}策略对模型进行训练==============================================="
    CUDA_VISIBLE_DEVICES=${cuda} python training.py --dataset_path ${dataset_path} \
    --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
    --train_batch_size 64 --eval_batch_size 64 --num_epoch 100 --opt_method "adamw" --lr_rate 1e-4 --weight_decay 1e-4 \
    --save_epoch 50 --eval_epoch 1 \
    --check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
    --model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls"
    echo "===============================================基于user_${i}策略对模型进行测试==============================================="
    CUDA_VISIBLE_DEVICES=${cuda} python testing.py --dataset_path ${dataset_path} \
    --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
    --check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
    --model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls" \
    --test_batch_size 64
  done

  for ((i = 0; i < 10; ++i));
  do
    preprocess_strategy=shuffle_${i}
    echo "===============================================基于shuffle_${i}策略对模型进行训练==============================================="
    CUDA_VISIBLE_DEVICES=${cuda} python training.py --dataset_path ${dataset_path} \
    --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
    --train_batch_size 64 --eval_batch_size 64 --num_epoch 100 --opt_method "adamw" --lr_rate 1e-4 --weight_decay 1e-4 \
    --save_epoch 50 --eval_epoch 1 \
    --check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
    --model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls"
    echo "===============================================基于shuffle_${i}策略对模型进行测试==============================================="
    CUDA_VISIBLE_DEVICES=${cuda} python testing.py --dataset_path ${dataset_path} \
    --preprocess_strategy ${preprocess_strategy} --seq_len 224 \
    --check_point_path ${check_point_path} --use_gpu true --gpu_device ${cuda} \
    --model_name ${model_name} --head_name "span_cls" --strategy_name "span_cls" \
    --test_batch_size 64
  done

done
