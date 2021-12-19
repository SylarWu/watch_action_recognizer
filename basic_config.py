import os


class BasicConfig(object):
    def __init__(self):
        # 数据预处理脚本config
        self.datasource_path = os.path.join(os.getcwd(), 'data_source', 'transform_source')
        self.dataset_path = os.path.join(os.getcwd(), 'data_source')
        self.preprocess_strategy = 'normal_0'
        self.seq_len = 224
        self.train_test_ratio = [0.8, 0.2]

        # 训练超参
        self.batch_size = 64
        self.train_batch_size = 64
        self.num_epoch = 1000
        self.opt_method = 'adamw'
        self.lr_rate = 5e-4
        self.weight_decay = 1e-4
        self.save_epoch = 100
        self.eval_epoch = 20
        self.check_point_path = os.path.join(os.path.join(os.getcwd(), 'log'))
        self.use_gpu = True

        self.eval_batch_size = 16

        # model/head/strategy
        self.model_name = 'unet'
        self.head_name = 'seq_cls'
        self.strategy_name = 'seq_cls'

        self.hidden_dim = 16 * 4
        self.n_classes = 18
