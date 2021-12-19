import os
import scipy.io as scio
from torch.utils.data.dataloader import DataLoader

from basic_config import BasicConfig
from data_process import preprocess_with_upsampling, SensorDataset
from pipeline import Trainer


def _init_model(config: BasicConfig):
    if config.model_name == 'unet':
        from model import SPPUNet, UNetConfig
        return SPPUNet(UNetConfig())


def _init_head(config: BasicConfig):
    if config.head_name == 'seq_cls':
        from model import SequenceClassifier
        return SequenceClassifier(hidden_dim=config.hidden_dim,
                                  n_classes=config.n_classes)


def _init_strategy(config: BasicConfig):
    model = _init_model(config)
    head = _init_head(config)
    if config.strategy_name == 'seq_cls':
        from strategy import SequenceCLSStrategy
        return SequenceCLSStrategy(model, head)


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
        train_data_loader=DataLoader(train_dataset, batch_size=basic_config.train_batch_size, shuffle=True, drop_last=True),
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
