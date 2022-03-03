import os


class BasicConfig(object):
    def __init__(self):
        # 数据预处理脚本config
        self.datasource_path = os.path.join('/data/wuxilei/watch_action_recognizer', 'data_source', 'transform_source')
        self.dataset_path = os.path.join('/data/wuxilei/watch_action_recognizer', 'data_source')
        self.preprocess_strategy = 'user_1'
        self.seq_len = 224
        self.train_test_ratio = [0.8, 0.2]

        # 训练超参
        self.train_batch_size = 64
        self.num_epoch = 100
        self.opt_method = 'adamw'
        self.lr_rate = 5e-4
        self.weight_decay = 1e-4
        self.save_epoch = 10
        self.eval_epoch = 1
        self.check_point_path = os.path.join('/data/wuxilei/watch_action_recognizer', 'log')
        self.use_gpu = True
        self.gpu_device = "0"
        self.eval_batch_size = 16

        # model/head/strategy
        self.model_name = 'resnet101'
        self.head_name = 'span_cls'
        self.strategy_name = 'span_cls'

        # 测试超参
        self.test_batch_size = 16
        self.model_path = os.path.join('/data/wuxilei/watch_action_recognizer', 'log', 'ResNet1D_SpanClassifier_final')

        self.n_classes = 18
