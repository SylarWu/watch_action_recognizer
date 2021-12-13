import os
import scipy.io as scio
import numpy as np

class SensorDataSource(object):
    def __init__(self, datasource_path: os.path):
        file_path_list = os.listdir(datasource_path)
        max_seq_len = 0
        for file_path in file_path_list:
            data = scio.loadmat(os.path.join(datasource_path, file_path))
            print(data['label'])




if __name__ == '__main__':
    datasource_path = os.path.join('F:/CODE/Python/watch_action_recognizer/data_source/transform_source')

    datasource = SensorDataSource(datasource_path)