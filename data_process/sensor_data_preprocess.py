import argparse
import logging
import os
import random
import time

import numpy
import scipy.io as scio
import torch
import torch.nn.functional as F

from sensor_data import SensorData

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="数据预处理：在序列轴上进行上采样到固定长度，依据不同策略切分数据成训练/测试集")

parser.add_argument("-i", '--input', dest="input_path", required=True, type=str,
                    help="数据源路径")

parser.add_argument("-o", '--output', dest="output_path", required=True, type=str,
                    help="数据经过转换后输出路径")

parser.add_argument("-s", '--strategy', dest="strategy", required=True, type=str,
                    help="制作数据策略：normal_i(0-4)/user_i(1-10)/shuffle")

parser.add_argument("-l", '--length', dest="seq_len", required=True, type=int,
                    help="经过处理后序列长度")

parser.add_argument("-n", '--normalize', dest="is_normalize", required=True, type=bool,
                    help="是否对数据进行预归一化处理")


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
                               strategy: str = 'normal_0',
                               ratio: list = [0.8, 0.2],
                               seq_len: int = 224,
                               is_nomalize: bool = False):
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

    logger.info("初试化源数据参数")

    users, actions, attempts, file_path_list = _init_config(datasource_path)

    logger.info("开始对数据进行上采样到固定长度: %d" % seq_len)

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
        label = (origin_mat['label'][0][0] - 1) * torch.ones((1, factor * seq_len))

        for i in range(factor):
            sensor_data = SensorData(user_id=int(user_id),
                                     action_id=int(action_id),
                                     attempt_id=int(attempt_id),
                                     accData=acc[:, :, i * seq_len:(i + 1) * seq_len],
                                     gyrData=gyr[:, :, i * seq_len:(i + 1) * seq_len],
                                     label=label[:, i * seq_len:(i + 1) * seq_len])
            merge_by_user_id[sensor_data.user_id - 1].append(sensor_data)
            merge_by_attempt_id[sensor_data.attempt_id].append(sensor_data)

    logger.info("上采样完成，开始基于策略对数据进行划分: %s" % strategy)

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
                    test_data['accData'].append(instance.accData)
                    test_data['gyrData'].append(instance.gyrData)
                    test_data['label'].append(instance.label)
            else:
                # 加入训练集
                for instance in data:
                    assert instance.attempt_id == attempt_id
                    train_data['accData'].append(instance.accData)
                    train_data['gyrData'].append(instance.gyrData)
                    train_data['label'].append(instance.label)
    elif strategy.startswith("shuffle_"):
        assert len(ratio) == 2
        for attempt_id, data in enumerate(merge_by_attempt_id):
            # 先shuffle
            random.seed(time.time())
            random.shuffle(data)
            # 前ratio[0]加入训练集
            for instance in data[:int(len(data) * ratio[0])]:
                assert instance.attempt_id == attempt_id
                train_data['accData'].append(instance.accData)
                train_data['gyrData'].append(instance.gyrData)
                train_data['label'].append(instance.label)
            # 后ratio[1]加入测试集
            for instance in data[int(len(data) * ratio[0]):]:
                assert instance.attempt_id == attempt_id
                test_data['accData'].append(instance.accData)
                test_data['gyrData'].append(instance.gyrData)
                test_data['label'].append(instance.label)
    else:
        _, target_user = strategy.split('_')
        target_user = int(target_user)
        for user_id, data in enumerate(merge_by_user_id):
            if user_id + 1 == target_user:
                # 加入测试集
                for instance in data:
                    assert instance.user_id == user_id + 1
                    test_data['accData'].append(instance.accData)
                    test_data['gyrData'].append(instance.gyrData)
                    test_data['label'].append(instance.label)
            else:
                # 加入训练集
                for instance in data:
                    assert instance.user_id == user_id + 1
                    train_data['accData'].append(instance.accData)
                    train_data['gyrData'].append(instance.gyrData)
                    train_data['label'].append(instance.label)
    for key, value in train_data.items():
        train_data[key] = torch.vstack(value)
    for key, value in test_data.items():
        test_data[key] = torch.vstack(value)

    if is_nomalize:
        logger.info("对数据进行归一化")
        _normalize(train_data, test_data)

    for key, value in train_data.items():
        train_data[key] = numpy.array(value)
    for key, value in test_data.items():
        test_data[key] = numpy.array(value)

    logger.info("数据集生成完成，保存")
    scio.savemat(os.path.join(output_dir, '%s_upsampling_%d' % (strategy, seq_len), 'train.mat'), train_data)
    scio.savemat(os.path.join(output_dir, '%s_upsampling_%d' % (strategy, seq_len), 'test.mat'), test_data)


def _normalize(train_data, test_data):
    train_len = train_data['label'].size(0)
    test_len = test_data['label'].size(0)

    accData = torch.vstack([train_data['accData'], test_data['accData']])
    assert accData.size(0) == train_len + test_len
    accData = (accData - torch.mean(accData, dim=-1, keepdim=True)) / torch.std(accData, dim=-1, keepdim=True)
    train_data['accData'] = accData[:train_len, :, :]
    test_data['accData'] = accData[train_len:, :, :]

    gyrData = torch.vstack([train_data['gyrData'], test_data['gyrData']])
    assert gyrData.size(0) == train_len + test_len
    gyrData = (gyrData - torch.mean(gyrData, dim=-1, keepdim=True)) / torch.std(gyrData, dim=-1, keepdim=True)
    train_data['gyrData'] = gyrData[:train_len, :, :]
    test_data['gyrData'] = gyrData[train_len:, :, :]


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess_with_upsampling(datasource_path=args.input_path,
                               output_dir=args.output_path,
                               strategy=args.strategy,
                               ratio=[0.8, 0.2],
                               seq_len=args.seq_len,
                               is_nomalize=args.is_normalize)
