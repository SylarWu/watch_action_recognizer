import os
from model import ResNet1D, SpanClassifier


class BasicConfig(object):
    def __init__(self):
        self.model_mapping = {
            'resnet18': ResNet1D.__name__,
            'resnet34': ResNet1D.__name__,
            'resnet50': ResNet1D.__name__,
            'resnet101': ResNet1D.__name__,
        }
        self.head_mapping = {
            'span_cls': SpanClassifier.__name__,
        }

        # model/head/strategy
        self.model_name = 'resnet101'
        self.head_name = 'span_cls'
        self.strategy_name = 'span_cls'

        # 数据预处理脚本config
        self.datasource_path = os.path.join('/data/wuxilei/watch_action_recognizer', 'transform_source')
        self.dataset_path = os.path.join('/data/wuxilei/watch_action_recognizer')
        self.preprocess_strategy = 'shuffle_0'
        self.seq_len = 224
        self.train_test_ratio = [0.8, 0.2]

        # 训练超参
        self.train_batch_size = 64
        self.eval_batch_size = 64
        self.num_epoch = 1000
        self.opt_method = 'adamw'
        self.lr_rate = 1e-4
        self.weight_decay = 1e-4
        self.save_epoch = 10
        self.eval_epoch = 1
        self.check_point_path = os.path.join('/data/wuxilei/watch_action_recognizer/log', '%s-%s' % (
            self.model_name, self.preprocess_strategy
        ))
        self.use_gpu = True
        self.gpu_device = "0"

        # 测试超参
        self.test_batch_size = 64
        self.model_path = os.path.join('/data/wuxilei/watch_action_recognizer/log', '%s-%s' % (
            self.model_name, self.preprocess_strategy
        ), '%s_%s_final' % (self.model_mapping.get(self.model_name), self.head_mapping.get(self.head_name)))

        self.n_classes = 18
