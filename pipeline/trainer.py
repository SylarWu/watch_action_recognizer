import os.path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self,
                 strategy: nn.Module,
                 train_data_loader: DataLoader,
                 eval_data_loader: DataLoader,
                 num_epoch: int,
                 opt_method: str,
                 lr_rate: float,
                 weight_decay: float,
                 save_epoch: int,
                 eval_epoch: int,
                 check_point_path: os.path,
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

    def _to_var(self, data: tuple):
        if self.use_gpu:
            return [Variable(x.cuda()) for x in data]
        else:
            return [Variable(x) for x in data]

    def _train_one_step(self, data):
        self.optimizer.zero_grad()

        accData, gyrData, label = data

        loss = self.strategy(accData, gyrData, label)

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def training(self):
        if self.use_gpu:
            self.strategy = self.strategy.cuda()

        self._init_optimizer()

        for epoch in range(self.num_epoch):
            self.strategy.train()
            log_info = 'Epoch: %d. ' % (epoch + 1)
            train_loss = 0
            for data in tqdm(self.train_data_loader):
                data = self._to_var(data)
                train_loss += self._train_one_step(data)
            log_info += 'Train Loss: %f. ' % train_loss

            if (epoch + 1) % self.eval_epoch == 0:
                self.strategy.eval()
                with torch.no_grad():
                    eval_loss = 0
                    for data in tqdm(self.eval_data_loader):
                        data = self._to_var(data)
                        eval_loss += self.strategy(data[0], data[1], data[2])
                log_info += 'Eval Loss: %f.' % eval_loss
            if (epoch + 1) % self.save_epoch == 0:
                torch.save(self.strategy.state_dict(),
                           os.path.join(self.check_point_path, '%s-%s-%d' % (self.strategy.model.__class__.__name__,
                                                                             self.strategy.head.__class__.__name__,
                                                                             epoch + 1)))
            print(log_info)
        torch.save(self.strategy.state_dict(),
                   os.path.join(self.check_point_path, '%s-%s-final' % (self.strategy.model.__class__.__name__,
                                                                        self.strategy.head.__class__.__name__)))
