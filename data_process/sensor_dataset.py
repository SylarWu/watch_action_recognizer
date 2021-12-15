import torch
from torch.utils.data.dataset import Dataset


class SensorDataset(Dataset):
    def __init__(self, mat_data):
        super(SensorDataset, self).__init__()
        self.accData = torch.from_numpy(mat_data['accData'])
        self.gyrData = torch.from_numpy(mat_data['gyrData'])
        self.label = torch.from_numpy(mat_data['label'])

        self.num_samples = self.accData.size(0)
        assert self.num_samples == self.gyrData.size(0)
        assert self.num_samples == self.label.size(0)

        self._transform()

    def _transform(self):
        # 加速度数据中心化
        self.accData = (self.accData - torch.mean(self.accData, dim=-1, keepdim=True)) / torch.std(self.accData, dim=-1,
                                                                                                   keepdim=True)
        # 陀螺仪数据中心化
        self.gyrData = (self.gyrData - torch.mean(self.gyrData, dim=-1, keepdim=True)) / torch.std(self.gyrData, dim=-1,
                                                                                                   keepdim=True)

    def __getitem__(self, index):
        return (self.accData[index], self.gyrData[index], self.label[index])

    def __len__(self):
        return self.num_samples
