from torch.utils.data import Dataset
import numpy as np
import torch


class TomoDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        image = sample["image"]  # np.array, shape: (D, H, W)
        label = sample["label"]  # np.array, shape: (D, H, W)

        # 필요시 transform 적용 (예: normalize, augmentation 등)
        if self.transform:
            sample = self.transform(sample)
            image = sample["image"]
            label = sample["label"]

        # 채널 차원 추가: (D, H, W) -> (1, D, H, W)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # torch.Tensor로 변환
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label