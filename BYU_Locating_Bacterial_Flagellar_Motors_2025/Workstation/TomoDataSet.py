import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def random_crop(volume, label, patch_size):
    D, H, W = volume.shape
    pd_d, pd_h, pd_w = patch_size
    z = np.random.randint(0, max(D - pd_d + 1, 1))
    y = np.random.randint(0, max(H - pd_h + 1, 1))
    x = np.random.randint(0, max(W - pd_w + 1, 1))
    cropped_vol = volume[z:z+pd_d, y:y+pd_h, x:x+pd_w]
    cropped_label = label[z:z+pd_d, y:y+pd_h, x:x+pd_w]
    return cropped_vol, cropped_label

class TomoDataset(Dataset):
    def __init__(self, csv_path, base_dir, patch_size=(64, 64, 64), transform=None):
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.transform = transform
        self.patch_size = patch_size
        self.df = pd.read_csv(csv_path)
        self.motor_coords = {}
        for _, row in self.df.iterrows():
            tomo_id = row["tomo_id"]
            if row["Number of motors"] == 0 or row["Motor axis 0"] == -1:
                continue
            coord = (int(row["Motor axis 0"]), int(row["Motor axis 1"]), int(row["Motor axis 2"]))
            self.motor_coords.setdefault(tomo_id, []).append(coord)
        self.tomo_ids = self.df["tomo_id"].unique()

    def __len__(self):
        return len(self.tomo_ids)

    def load_volume(self, tomo_dir):
        slice_paths = sorted(glob.glob(os.path.join(tomo_dir, "slice_*.jpg")))
        slices = []
        for sp in slice_paths:
            img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            slices.append(img)
        if len(slices) == 0:
            return None
        volume = np.stack(slices, axis=0)
        return volume

    def create_label(self, tomo_id, volume_shape):
        label = np.zeros(volume_shape, dtype=np.uint8)
        if tomo_id in self.motor_coords:
            for coord in self.motor_coords[tomo_id]:
                z, y, x = coord
                if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
                    label[z, y, x] = 1
        return label

    def __getitem__(self, idx):
        tomo_id = self.tomo_ids[idx]
        tomo_dir = os.path.join(self.base_dir, tomo_id)
        volume = self.load_volume(tomo_dir)
        if volume is None:
            raise ValueError(f"Volume for {tomo_id} could not be loaded.")
        label = self.create_label(tomo_id, volume.shape)
        volume, label = random_crop(volume, label, self.patch_size)
        volume = np.expand_dims(volume, axis=0)
        label = np.expand_dims(label, axis=0)
        sample = {"image": volume, "label": label, "tomo_id": tomo_id}
        if self.transform:
            sample = self.transform(sample)
        image = torch.tensor(sample["image"], dtype=torch.float32)
        label = torch.tensor(sample["label"].squeeze(0), dtype=torch.long)
        return image, label
