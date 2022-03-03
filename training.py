import os
import logging
import argparse

import scipy.io as scio
import torch.cuda
from torch.utils.data.dataloader import DataLoader

from basic_config import BasicConfig
from data_process import SensorDataset
from pipeline import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def _init_configs() -> BasicConfig:
    parser = argparse.ArgumentParser(description="模型训练")

    parser.add_argument('--dataset_path', dest="dataset_path", required=True, type=str,
                        help="包含train.mat/test.mat数据集路径")

    parser.add_argument('--preprocess_strategy', dest="preprocess_strategy", required=True, type=str,
                        help="预处理数据集策略，基于此加载相应的数据集：normal_i(0-4)/user_i(1-10)/shuffle")

    parser.add_argument('--seq_len', dest="seq_len", required=False, type=int, default=224,
                        help="数据集经过处理后序列长度")

    parser.add_argument('--train_batch_size', dest="train_batch_size", required=False, type=int, default=64,
                        help="训练使用batch_size")

    parser.add_argument('--test_batch_size', dest="test_batch_size", required=False, type=int, default=64,
                        help="测试/验证使用batch_size")

    parser.add_argument('--num_epoch', dest="num_epoch", required=False, type=int, default=1000,
                        help="训练epoch")

    parser.add_argument('--opt_method', dest="opt_method", required=False, type=str, default="adam",
                        help="训练模型使用优化器")

    parser.add_argument('--lr_rate', dest="lr_rate", required=False, type=float, default=2e-5,
                        help="训练学习率")

    parser.add_argument('--weight_decay', dest="weight_decay", required=False, type=float, default=1e-4,
                        help="训练正则化系数")

    parser.add_argument('--save_epoch', dest="save_epoch", required=False, type=int, default=50,
                        help="训练中途每隔一定epoch数后对模型进行保存")

    parser.add_argument('--eval_epoch', dest="eval_epoch", required=False, type=int, default=1,
                        help="训练中途每隔一定epoch数后使用模型在验证集上验证")

    parser.add_argument('--check_point_path', dest="check_point_path", required=True, type=str,
                        help="训练中途临时保存路径")

    parser.add_argument('--use_gpu', dest="use_gpu", required=False, type=bool, default=torch.cuda.is_available(),
                        help="训练是否使用GPU：0-3")

    parser.add_argument('--gpu_device', dest="gpu_device", required=False, type=str, default="0",
                        help="训练使用的GPU编号")

    parser.add_argument('--model_name', dest="model_name", required=False, type=str, default="resnet101",
                        help="训练使用的模型名")

    parser.add_argument('--head_name', dest="head_name", required=False, type=str, default="span_cls",
                        help="训练使用的预测头")

    parser.add_argument('--strategy_name', dest="strategy_name", required=False, type=str, default="span_cls",
                        help="使用的训练策略")

    parser.add_argument('--n_classes', dest="n_classes", required=False, type=int, default=18,
                        help="分类个数")

    configs = BasicConfig()
    args = parser.parse_args()
    configs.dataset_path        = args.dataset_path
    configs.preprocess_strategy = args.preprocess_strategy
    configs.seq_len             = args.seq_len
    configs.train_batch_size    = args.train_batch_size
    configs.test_batch_size     = args.test_batch_size
    configs.num_epoch           = args.num_epoch
    configs.opt_method          = args.opt_method
    configs.lr_rate             = args.lr_rate
    configs.weight_decay        = args.weight_decay
    configs.save_epoch          = args.save_epoch
    configs.eval_epoch          = args.eval_epoch
    configs.use_gpu             = args.use_gpu
    configs.gpu_device          = args.gpu_device
    configs.model_name          = args.model_name
    configs.head_name           = args.head_name
    configs.strategy_name       = args.strategy_name
    configs.n_classes           = args.n_classes
    configs.check_point_path    = os.path.join(args.check_point_path, "%s_%s" % (
        configs.model_name, configs.preprocess_strategy
    ))
    return configs


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
    basic_config = _init_configs()
    os.environ["CUDA_VISIBLE_DEVICES"] = basic_config.gpu_device

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
