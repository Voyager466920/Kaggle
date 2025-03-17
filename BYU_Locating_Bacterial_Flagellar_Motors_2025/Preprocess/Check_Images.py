import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


CSV_PATH = 'C:/junha/Datasets/Kaggle/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv'
TRAIN_DIR = 'C:/junha/Datasets/Kaggle/byu-locating-bacterial-flagellar-motors-2025/train'

df = pd.read_csv(CSV_PATH)
df_has_motor = df[df['Number of motors'] > 0].copy()
df_has_motor = df_has_motor[(df_has_motor['Motor axis 0'] >= 0) & (df_has_motor['Motor axis 1'] >= 0) & (df_has_motor['Motor axis 2'] >= 0)]
grouped = df_has_motor.groupby(['tomo_id', 'Motor axis 0'])
for (tomo_id, slice_idx), group in grouped:
    tomo_folder = os.path.join(TRAIN_DIR, tomo_id)
    slice_idx_int = int(slice_idx)
    slice_filename = f"slice_{slice_idx_int:04d}.jpg"
    slice_path = os.path.join(tomo_folder, slice_filename)
    if not os.path.exists(slice_path):
        continue
    img = cv2.imread(slice_path)
    if img is None:
        continue
    for _, row in group.iterrows():
        y = int(row['Motor axis 1'])
        x = int(row['Motor axis 2'])
        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
            continue
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    plt.figure(figsize=(10,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'{tomo_id} - slice {slice_idx_int:04d}')
    plt.show()
