import os


class BasicConfig(object):
    def __init__(self):
        # 数据预处理脚本config
        self.datasource_path = os.path.join('F:/CODE/Python/watch_action_recognizer/data_source/transform_source')
        self.dataset_path = os.path.join('F:/CODE/Python/watch_action_recognizer/data_source')
        self.preprocess_strategy = 'normal'
        self.seq_len = 224
        self.train_test_ratio = [0.8, 0.2]

        self.batch_size = 64
