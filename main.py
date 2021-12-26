import logging
import os

import scipy.io as scio
from torch.utils.data.dataloader import DataLoader

from basic_config import BasicConfig
from data_process import preprocess_with_upsampling, SensorDataset
from pipeline import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

def _init_model(model_name):
    logger.info('初试化模型')
    if model_name.startswith('resnet'):
        from model import resnet, ResNetConfig
        return resnet(model_name, ResNetConfig())


def _init_head(head_name, hidden_dim, n_classes):
    logger.info('初试化预测头')
    if head_name == 'span_cls':
        from model import SpanClassifier
        return SpanClassifier(hidden_dim, n_classes)


def _init_strategy(config: BasicConfig):
    logger.info('初试化训练策略')
    model = _init_model(config.model_name)
    head = _init_head(config.head_name, model.get_output_size(), config.n_classes)
    if config.strategy_name == 'span_cls':
        from strategy import SpanCLSStrategy
        return SpanCLSStrategy(model, head)


if __name__ == '__main__':
    basic_config = BasicConfig()
    preprocess_with_upsampling(basic_config.datasource_path,
                               basic_config.dataset_path,
                               basic_config.preprocess_strategy,
                               basic_config.train_test_ratio,
                               basic_config.seq_len)

    train_mat = scio.loadmat(os.path.join(basic_config.dataset_path,
                                          '%s_upsampling_%d' % (basic_config.preprocess_strategy, basic_config.seq_len),
                                          'train.mat'))
    test_mat = scio.loadmat(os.path.join(basic_config.dataset_path,
                                         '%s_upsampling_%d' % (basic_config.preprocess_strategy, basic_config.seq_len),
                                         'test.mat'))
    train_dataset = SensorDataset(train_mat)
    test_dataset = SensorDataset(test_mat)

    strategy = _init_strategy(basic_config)

    trainer = Trainer(
        strategy=strategy,
        train_data_loader=DataLoader(train_dataset, batch_size=basic_config.train_batch_size, shuffle=True,
                                     drop_last=True),
        eval_data_loader=DataLoader(test_dataset, batch_size=basic_config.eval_batch_size, shuffle=False),
        num_epoch=basic_config.num_epoch,
        opt_method=basic_config.opt_method,
        lr_rate=basic_config.lr_rate,
        weight_decay=basic_config.weight_decay,
        save_epoch=basic_config.save_epoch,
        eval_epoch=basic_config.eval_epoch,
        check_point_path=basic_config.check_point_path,
        use_gpu=basic_config.use_gpu,
    )

    trainer.training()
