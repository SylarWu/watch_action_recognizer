import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# TODO: 不支持从断点重新开始训练，后续可添加
class Trainer(object):
    def __init__(self,
                 strategy: nn.Module,
                 train_data_loader: DataLoader,
                 eval_data_loader: DataLoader,
                 num_epoch=1000,
                 opt_method='adam',
                 lr_rate=2e-5,
                 weight_decay=1e-4,
                 save_epoch=100,
                 eval_epoch=100,
                 check_point_path=None,
                 use_gpu=True):
        super(Trainer, self).__init__()

        self.strategy = strategy

        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader

        self.num_epoch = num_epoch

        self.opt_method = opt_method
        self.lr_rate = lr_rate
        self.weight_decay = weight_decay

        self.save_epoch = save_epoch
        self.eval_epoch = eval_epoch

        self.check_point_path = check_point_path

        self.use_gpu = use_gpu

    def _init_optimizer(self):
        params = [
            {'params': self.strategy.model.parameters()},
            {'params': self.strategy.head.parameters()},
        ]
        if self.opt_method == 'adam':
            self.optimizer = torch.optim.Adam(params=params,
                                              lr=self.lr_rate,
                                              weight_decay=self.weight_decay)
        elif self.opt_method == 'adamw':
            self.optimizer = torch.optim.AdamW(params=params,
                                               lr=self.lr_rate,
                                               weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params=params,
                                             lr=self.lr_rate,
                                             weight_decay=self.weight_decay)

    def training(self):
        if self.use_gpu:
            self.strategy = self.strategy.cuda()

        self._init_optimizer()

        logger.info('开始训练'.center(100, '='))

        for epoch in range(self.num_epoch):
            self.strategy.train()
            log_info = 'Epoch: %d. ' % (epoch + 1)
            loss_res = 0
            for data in self.train_data_loader:
                data = to_var(data, self.use_gpu)
                loss_res += train_one_step(data[0], data[2:], self.optimizer, self.strategy)
            log_info += 'Loss: %f. ' % loss_res
            self.strategy.eval()
            if (epoch + 1) % self.eval_epoch == 0:
                metrics = evaluating_model(self.eval_data_loader,
                                           self.strategy,
                                           self.use_gpu)
                for k, v in metrics.items():
                    log_info += '%s: %.2f%%. ' % (k, v)
            if (epoch + 1) % self.save_epoch == 0:
                # TODO: 保存模型函数
                pass
            logger.info(log_info)
        # TODO: 保存最后一次训练后的模型
