import os
import glob
import random
import pandas as pd
import shutil

TRAIN_ROOT = "C:/junha/Datasets/Kaggle/byu-locating-bacterial-flagellar-motors-2025/train"
VAL_ROOT = "C:/junha/Datasets/Kaggle/byu-locating-bacterial-flagellar-motors-2025/val"

if not os.path.exists(VAL_ROOT):
    os.makedirs(VAL_ROOT)

tomo_dirs = glob.glob(os.path.join(TRAIN_ROOT, "tomo_*"))
tomo_dirs = sorted(tomo_dirs)
tomo_ids = [os.path.basename(d) for d in tomo_dirs]

random.shuffle(tomo_ids)
n_total = len(tomo_ids)
n_val = int(n_total * 0.2)  # 전체의 20%
val_ids = tomo_ids[:n_val]
train_ids = tomo_ids[n_val:]

print(f"총 {n_total}개 디렉토리 중, Train: {len(train_ids)}개, Val: {len(val_ids)}개")

for t_id in val_ids:
    src = os.path.join(TRAIN_ROOT, t_id)
    dst = os.path.join(VAL_ROOT, t_id)
    print(f"Moving {src} -> {dst}")
    shutil.move(src, dst)

csv_path = 'C:/junha/Datasets/Kaggle/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv'
df = pd.read_csv(csv_path)

df_train = df[df["tomo_id"].isin(train_ids)].copy()
df_val   = df[df["tomo_id"].isin(val_ids)].copy()

df_train.to_csv("train_labels_train.csv", index=False)
df_val.to_csv("train_labels_val.csv", index=False)

print("Train/Val CSV 파일 생성 완료!")
