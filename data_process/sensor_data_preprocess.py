import os
import random
import numpy as np
import scipy.io as scio


def _init_config(datasource_path: os.path):
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


def preprocess_with_padding(datasource_path: os.path,
                            output_dir: os.path,
                            strategy: str = 'shuffle',
                            ratio: list = [0.8, 0.1, 0.1],
                            max_seq_len: int = 224):
    """
    使用padding预处理数据，将其制作成train/dev/test
    :param datasource_path: 数据源路径
    :param output_dir: 输出路径
    :param strategy: 制作数据策略: normal/user_independent/shuffle/
        normal: train到所有人的动作
        user_independent: train部分人的动作，dev和test都是没见过的人
        shuffle: 随机shuffle，上面两种的折衷
    :param ratio: train/dev/test比例
    :param max_seq_len: padding的最大长度
    :return:
    """
    if not os.path.exists(os.path.join(output_dir, 'after_padding_%d' % max_seq_len)):
        os.makedirs(os.path.join(output_dir, 'after_padding_%d' % max_seq_len))

    users, actions, attempts, file_path_list = _init_config(datasource_path)
    if strategy == 'shuffle':
        save_data = {
            'acc': list(),
            'gyr': list(),
            'label': list(),
            'length': list(),
        }
        for file_path in file_path_list:
            temp_data = scio.loadmat(os.path.join(datasource_path, file_path))
            length = temp_data['label'].shape[1]
            for i in range(length // max_seq_len + 1):
                # 初试化
                temp_acc = np.zeros((3, max_seq_len), dtype=np.float32)
                temp_gyr = np.zeros((3, max_seq_len), dtype=np.float32)
                temp_label = np.zeros(max_seq_len, dtype=np.int32)

                if (i + 1) * max_seq_len > length:
                    # 长度过短
                    temp_acc[:, :length - i * max_seq_len] = temp_data['accData'][:, :, i * max_seq_len:]
                    temp_gyr[:, :length - i * max_seq_len] = temp_data['gyrData'][:, :, i * max_seq_len:]
                    temp_label[:length - i * max_seq_len] = temp_data['label'][:, i * max_seq_len:]
                else:
                    # 长度足够max_seq_len
                    temp_acc[:, :] = temp_data['accData'][:, :, i * max_seq_len: (i + 1) * max_seq_len]
                    temp_gyr[:, :] = temp_data['gyrData'][:, :, i * max_seq_len: (i + 1) * max_seq_len]
                    temp_label = temp_data['label'][:, i * max_seq_len: (i + 1) * max_seq_len]
                save_data['acc'].append(temp_acc)
                save_data['gyr'].append(temp_gyr)
                save_data['label'].append(temp_label)
                save_data['length'].append(max_seq_len if (i + 1) * max_seq_len > length else length - i * max_seq_len)
        indexes = list(range(len(save_data['length'])))
        random.shuffle(indexes)
        acc = np.zeros((len(indexes), 3, max_seq_len), np.float32)
        gyr = np.zeros((len(indexes), 3, max_seq_len), np.float32)
        label = np.zeros((len(indexes), max_seq_len), np.int32)
        length = np.zeros((len(indexes)))
        for i, index in enumerate(indexes):
            acc[i, :, :] = save_data['acc'][index][:, :]
            gyr[i, :, :] = save_data['gyr'][index][:, :]
            label[i, :] = save_data['label'][index][:]
            length[i] = save_data['length'][index]
        scio.savemat(os.path.join(output_dir, 'after_padding_%d' % max_seq_len, 'train.mat'), {
            'accData': acc[:int(len(indexes) * ratio[0])],
            'gyrData': gyr[:int(len(indexes) * ratio[0])],
            'label': label[:int(len(indexes) * ratio[0])],
            'length': length[:int(len(indexes) * ratio[0])],
        })

        scio.savemat(os.path.join(output_dir, 'after_padding_%d' % max_seq_len, 'dev.mat'), {
            'accData': acc[int(len(indexes) * ratio[0]):int(len(indexes) * (ratio[0] + ratio[1]))],
            'gyrData': gyr[int(len(indexes) * ratio[0]):int(len(indexes) * (ratio[0] + ratio[1]))],
            'label': label[int(len(indexes) * ratio[0]):int(len(indexes) * (ratio[0] + ratio[1]))],
            'length': length[int(len(indexes) * ratio[0]):int(len(indexes) * (ratio[0] + ratio[1]))],
        })

        scio.savemat(os.path.join(output_dir, 'after_padding_%d' % max_seq_len, 'test.mat'), {
            'accData': acc[int(len(indexes) * (ratio[0] + ratio[1])):],
            'gyrData': gyr[int(len(indexes) * (ratio[0] + ratio[1])):],
            'label': label[int(len(indexes) * (ratio[0] + ratio[1])):],
            'length': length[int(len(indexes) * (ratio[0] + ratio[1])):],
        })


def preprocess_with_overlap(seq_len: int = 224):
    pass


if __name__ == '__main__':
    datasource_path = os.path.join('F:/CODE/Python/watch_action_recognizer/data_source/transform_source')
    output_dir = os.path.join('F:/CODE/Python/watch_action_recognizer/data_source/')
    preprocess_with_padding(datasource_path, output_dir)
