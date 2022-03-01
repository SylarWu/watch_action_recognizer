# 基于normal策略对数据切分
for ((i = 0; i < 5; ++i));
do
  echo "==============================================================================================================="
  echo "基于normal_$i 策略对数据源进行切分"
  python ../data_process/sensor_data_preprocess.py --input '/data/wuxilei/watch_action_recognizer/transform_source' \
  --output '/data/wuxilei/watch_action_recognizer' --strategy normal_$i --length 224 --normalize true
  echo "==============================================================================================================="
done

# 基于user_independent策略对数据切分
for ((i = 0; i < 10; ++i));
do
  echo "==============================================================================================================="
  echo "基于user_$i 策略对数据源进行切分"
  python ../data_process/sensor_data_preprocess.py --input '/data/wuxilei/watch_action_recognizer/transform_source' \
  --output '/data/wuxilei/watch_action_recognizer' --strategy user_$i --length 224 --normalize true
  echo "==============================================================================================================="
done

# 基于shuffle策略对数据切分
echo "==============================================================================================================="
echo "基于shuffle策略对数据源进行切分"
python ../data_process/sensor_data_preprocess.py --input '/data/wuxilei/watch_action_recognizer/transform_source' \
--output '/data/wuxilei/watch_action_recognizer' --strategy shuffle --length 224 --normalize true
echo "==============================================================================================================="