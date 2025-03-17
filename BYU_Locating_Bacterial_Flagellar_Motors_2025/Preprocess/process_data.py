import os
import glob
import cv2
import numpy as np
import pandas as pd


def load_volume_from_dir(tomo_dir):
    slice_paths = sorted(glob.glob(os.path.join(tomo_dir, "slice_*.jpg")))
    slices = []
    for sp in slice_paths:
        img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[경고] {sp} 이미지 로드 실패")
            continue
        slices.append(img)
    if len(slices) == 0:
        return None
    volume = np.stack(slices, axis=0)  # (D, H, W)
    return volume


def build_data_list(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    data_list = []
    motor_coords = {}
    for _, row in df.iterrows():
        tomo_id = row["tomo_id"]
        if row["Number of motors"] == 0 or row["Motor axis 0"] == -1:
            continue
        coord = (int(row["Motor axis 0"]), int(row["Motor axis 1"]), int(row["Motor axis 2"]))
        motor_coords.setdefault(tomo_id, []).append(coord)

    # CSV에 있는 모든 unique tomo_id에 대해 볼륨과 라벨 생성
    tomo_ids = df["tomo_id"].unique()
    for tomo_id in tomo_ids:
        tomo_dir = os.path.join(base_dir, tomo_id)
        if not os.path.isdir(tomo_dir):
            print(f"[경고] {tomo_dir} 디렉토리가 존재하지 않습니다.")
            continue

        volume = load_volume_from_dir(tomo_dir)
        if volume is None:
            print(f"[경고] {tomo_id} 볼륨 로드 실패")
            continue

        # 라벨 볼륨: volume과 동일한 shape, 기본값 0
        label = np.zeros(volume.shape, dtype=np.uint8)
        if tomo_id in motor_coords:
            for coord in motor_coords[tomo_id]:
                z, y, x = coord  # (slice, row, col)로 가정
                # 좌표가 volume 범위 내에 있는지 확인 후 1로 마킹
                if 0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]:
                    label[z, y, x] = 1
        data_list.append({"image": volume, "label": label, "tomo_id": tomo_id})
    return data_list

