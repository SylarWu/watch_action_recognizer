#!/bin/bash

datasource_path="/data/wuxilei/watch_action_recognizer/transform_source"
output_path="/data/wuxilei/watch_action_recognizer"
seq_len=224

cd ../data_process

# 进行归一化处理
# 基于normal策略对数据切分
for ((i = 0; i < 5; ++i));
do
  echo "==============================================================================================================="
  echo "基于normal_${i}策略对数据源进行切分"
  python sensor_data_preprocess.py --input ${datasource_path} --output ${output_path} \
  --strategy normal_${i} --length ${seq_len} --normalize true
  echo "==============================================================================================================="
done

# 基于user_independent策略对数据切分
for ((i = 1; i <= 10; ++i));
do
  echo "==============================================================================================================="
  echo "基于user_${i}策略对数据源进行切分"
  python sensor_data_preprocess.py --input ${datasource_path} --output ${output_path} \
  --strategy user_${i} --length ${seq_len} --normalize true
  echo "==============================================================================================================="
done

# 基于shuffle策略对数据切分
for ((i = 0; i < 10; ++i));
do
  echo "==============================================================================================================="
  echo "基于shuffle_${i}策略对数据源进行切分"
    python sensor_data_preprocess.py --input ${datasource_path} --output ${output_path} \
    --strategy shuffle_${i} --length ${seq_len} --normalize true
  echo "==============================================================================================================="
done

# 不进行归一化处理
# 基于normal策略对数据切分
for ((i = 0; i < 5; ++i));
do
  echo "==============================================================================================================="
  echo "基于normal_${i}策略对数据源进行切分"
  python sensor_data_preprocess.py --input ${datasource_path} --output ${output_path} \
  --strategy normal_${i} --length ${seq_len}
  echo "==============================================================================================================="
done

# 基于user_independent策略对数据切分
for ((i = 1; i <= 10; ++i));
do
  echo "==============================================================================================================="
  echo "基于user_${i}策略对数据源进行切分"
  python sensor_data_preprocess.py --input ${datasource_path} --output ${output_path} \
  --strategy user_${i} --length ${seq_len}
  echo "==============================================================================================================="
done

# 基于shuffle策略对数据切分
for ((i = 0; i < 10; ++i));
do
  echo "==============================================================================================================="
  echo "基于shuffle_${i}策略对数据源进行切分"
    python sensor_data_preprocess.py --input ${datasource_path} --output ${output_path} \
    --strategy shuffle_${i} --length ${seq_len}
  echo "==============================================================================================================="
done