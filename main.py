import os

import scipy.io as scio

from basic_config import BasicConfig
from data_process import preprocess_with_upsampling, SensorDataset

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
