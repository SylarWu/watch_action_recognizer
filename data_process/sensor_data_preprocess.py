import os
import random

import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F

from data_process.sensor_data import SensorData


def _init_config(datasource_path: os.path):
    # user_id: 1 - 10
    # action_id: 1 - 18
    # attempt_id: 0 - 9
    file_path_list = os.listdir(datasource_path)
    users = set()
    actions = set()
    attempts = set()
    for file_path in file_path_list:
        user_id, action_id, attempt_id = file_path.split('.')[0].split('_')
        users.add(user_id)
        actions.add(action_id)
        attempts.add(attempt_id)
    return sorted(users), sorted(actions), sorted(attempts), sorted(file_path_list)


def preprocess_with_upsampling(datasource_path: os.path,
                               output_dir: os.path,
                               strategy: str = 'normal',
                               ratio: list = [0.8, 0.2],
                               seq_len: int = 224):
    """
    在序列轴上进行上采样到固定长度，依据不同策略切分数据成训练/测试集
    :param datasource_path: 数据源路径
    :param output_dir: 输出路径
    :param strategy: 制作数据策略: normal/user_independent/shuffle/
        normal_i: test第i*2 ~ i*2 + 1次尝试，其余尝试数据进行训练，i：0~4
        user_i: test第i人，其余人数据进行train，i：1~10
        shuffle: 随机shuffle，上面两种的折衷
    :param ratio: train/test比例
    :param seq_len: 固定长度
    :return: 在output_dir下生成train.mat/test.mat
    """
    if not os.path.exists(os.path.join(output_dir, '%s_upsampling_%d' % (strategy, seq_len))):
        os.makedirs(os.path.join(output_dir, '%s_upsampling_%d' % (strategy, seq_len)))
    # 初试化源数据参数
    users, actions, attempts, file_path_list = _init_config(datasource_path)
    # 首先对所有数据进行上采样到固定长度，超过固定长度的上采样到固定长度的倍数
    merge_by_user_id = [list() for _ in users]
    merge_by_attempt_id = [list() for _ in attempts]
    for file_path in file_path_list:
        user_id, action_id, attempt_id = file_path.split('.')[0].split('_')
        origin_mat = scio.loadmat(os.path.join(datasource_path, file_path))

        length = origin_mat['label'].shape[1]
        factor = length // seq_len + 1
        assert origin_mat['label'][0][0] == int(action_id)

        acc = F.interpolate(torch.from_numpy(origin_mat['accData']), size=(factor * seq_len), mode='linear')
        gyr = F.interpolate(torch.from_numpy(origin_mat['gyrData']), size=(factor * seq_len), mode='linear')
        label = origin_mat['label'][0][0] * torch.ones((1, factor * seq_len))

        for i in range(factor):
            sensor_data = SensorData(user_id=int(user_id),
                                     action_id=int(action_id),
                                     attempt_id=int(attempt_id),
                                     accData=acc[0, :, i * seq_len:(i + 1) * seq_len],
                                     gyrData=gyr[0, :, i * seq_len:(i + 1) * seq_len],
                                     label=label[0, i * seq_len:(i + 1) * seq_len])
            merge_by_user_id[sensor_data.user_id - 1].append(sensor_data)
            merge_by_attempt_id[sensor_data.attempt_id].append(sensor_data)
    # 上采样完成，开始基于策略对数据进行划分
    train_data = {
        'accData': list(),
        'gyrData': list(),
        'label': list(),
    }
    test_data = {
        'accData': list(),
        'gyrData': list(),
        'label': list(),
    }
    if strategy.startswith("normal_"):
        _, target_attempt = strategy.split('_')
        target_attempt = int(target_attempt)
        for attempt_id, data in enumerate(merge_by_attempt_id):
            if attempt_id // 2 == target_attempt:
                # 加入测试集
                for instance in data:
                    assert instance.attempt_id == attempt_id
                    test_data['accData'].append(instance.accData.numpy())
                    test_data['gyrData'].append(instance.gyrData.numpy())
                    test_data['label'].append(instance.label.numpy())
            else:
                # 加入训练集
                for instance in data:
                    assert instance.attempt_id == attempt_id
                    train_data['accData'].append(instance.accData.numpy())
                    train_data['gyrData'].append(instance.gyrData.numpy())
                    train_data['label'].append(instance.label.numpy())

    elif strategy == 'shuffle':
        for attempt_id, data in enumerate(merge_by_attempt_id):
            # 先shuffle
            random.shuffle(data)
            # 前ratio[0]加入训练集
            for instance in data[:int(len(data) * ratio[0])]:
                assert instance.attempt_id == attempt_id
                train_data['accData'].append(instance.accData.numpy())
                train_data['gyrData'].append(instance.gyrData.numpy())
                train_data['label'].append(instance.label.numpy())
            # 后ratio[1]加入测试集
            for instance in data[int(len(data) * ratio[0]):]:
                assert instance.attempt_id == attempt_id
                test_data['accData'].append(instance.accData.numpy())
                test_data['gyrData'].append(instance.gyrData.numpy())
                test_data['label'].append(instance.label.numpy())
    else:
        _, target_user = strategy.split('_')
        target_user = int(target_user)
        for user_id, data in enumerate(merge_by_user_id):
            if user_id + 1 == target_user:
                # 加入测试集
                for instance in data:
                    assert instance.user_id == user_id + 1
                    test_data['accData'].append(instance.accData.numpy())
                    test_data['gyrData'].append(instance.gyrData.numpy())
                    test_data['label'].append(instance.label.numpy())
            else:
                # 加入训练集
                for instance in data:
                    assert instance.user_id == user_id + 1
                    train_data['accData'].append(instance.accData.numpy())
                    train_data['gyrData'].append(instance.gyrData.numpy())
                    train_data['label'].append(instance.label.numpy())

    for key, value in train_data.items():
        train_data[key] = np.array(value)
    for key, value in test_data.items():
        test_data[key] = np.array(value)

    scio.savemat(os.path.join(output_dir, '%s_upsampling_%d' % (strategy, seq_len), 'train.mat'), train_data)
    scio.savemat(os.path.join(output_dir, '%s_upsampling_%d' % (strategy, seq_len), 'test.mat'), test_data)


if __name__ == '__main__':
    strategy = 'normal_4'
    datasource_path = os.path.join('F:/CODE/Python/watch_action_recognizer/data_source/transform_source')
    output_dir = os.path.join('F:/CODE/Python/watch_action_recognizer/data_source/')
    preprocess_with_upsampling(datasource_path, output_dir, strategy, seq_len=224)

    train_data = scio.loadmat(os.path.join(output_dir, '%s_upsampling_%d' % (strategy, 224), 'train.mat'))
    test_data = scio.loadmat(os.path.join(output_dir, '%s_upsampling_%d' % (strategy, 224), 'test.mat'))

    print(train_data['accData'].shape)
    print(train_data['gyrData'].shape)
    print(train_data['label'].shape)

    print(test_data['accData'].shape)
    print(test_data['gyrData'].shape)
    print(test_data['label'].shape)