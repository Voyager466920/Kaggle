import os
import glob
import cv2
import math
import torch
import numpy as np
import pandas as pd
from Models.UNET3D import UNet3D

def load_volume(tomo_dir):
    slice_paths = sorted(glob.glob(os.path.join(tomo_dir, "slice_*.jpg")))
    slices = []
    for sp in slice_paths:
        img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            slices.append(img)
    if len(slices) == 0:
        return None
    volume = np.stack(slices, axis=0)
    return volume

def infer_volume_by_patches(model, volume, device, patch_size=(64,64,64)):
    D, H, W = volume.shape
    pd_, ph_, pw_ = patch_size
    pad_D = math.ceil(D / pd_) * pd_
    pad_H = math.ceil(H / ph_) * ph_
    pad_W = math.ceil(W / pw_) * pw_
    padded_volume = np.zeros((pad_D, pad_H, pad_W), dtype=volume.dtype)
    padded_volume[:D, :H, :W] = volume
    pred_volume = np.zeros((pad_D, pad_H, pad_W), dtype=np.uint8)
    for d in range(0, pad_D, pd_):
        for h in range(0, pad_H, ph_):
            for w in range(0, pad_W, pw_):
                patch = padded_volume[d:d+pd_, h:h+ph_, w:w+pw_]
                input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                pred_patch = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                pred_volume[d:d+pd_, h:h+ph_, w:w+pw_] = pred_patch
    pred_volume = pred_volume[:D, :H, :W]
    return pred_volume

def find_motor_coordinate(pred_mask):
    coords = np.argwhere(pred_mask == 1)
    if len(coords) == 0:
        return None
    return coords[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load("/home/junha/Kaggle/BYU/CheckPoints/best_model.pth", map_location=device))
    model.eval()
    test_dir = "/home/junha/Kaggle/BYU/Datasets/test"
    tomo_ids = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    tomo_ids.sort()
    results = []
    for tomo_id in tomo_ids:
        tomo_path = os.path.join(test_dir, tomo_id)
        volume = load_volume(tomo_path)
        if volume is None:
            results.append([tomo_id, 0, -1, -1, -1])
            continue
        pred_mask = infer_volume_by_patches(model, volume, device, patch_size=(64,64,64))
        coord = find_motor_coordinate(pred_mask)
        if coord is None:
            results.append([tomo_id, 0, -1, -1, -1])
        else:
            z, y, x = coord
            results.append([tomo_id, 1, z, y, x])
    submission_df = pd.DataFrame(results, columns=["tomo_id", "Number of motors", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
    submission_df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
