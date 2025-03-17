import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from BYU_Locating_Bacterial_Flagellar_Motors_2025.Workstation.Test_Step import test_step
from BYU_Locating_Bacterial_Flagellar_Motors_2025.Workstation.Train_Step import train_step
from .TomoDataSet import TomoDataset
from ..Models.UNET3D import UNet3D
from ..Preprocess.process_data import build_data_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = r"C:\junha\Datasets\Kaggle\byu-locating-bacterial-flagellar-motors-2025"
data_list_train = build_data_list("train_labels_train.csv", base_dir+"/train")
data_list_val = build_data_list("train_labels_val.csv", base_dir+"/val")

print(f"Train 데이터: {len(data_list_train)} 개의 tomo 볼륨 로드 완료.")
print(f"Val 데이터: {len(data_list_val)} 개의 tomo 볼륨 로드 완료.")

train_dataset = TomoDataset(data_list_train, transform=None)
test_dataset   = TomoDataset(data_list_val, transform=None)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
test_dataloader   = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

model = UNet3D().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
NUM_EPOCHS = 20

best_dice = -1
for epoch in range(NUM_EPOCHS):
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

    train_loss = train_step(model, train_dataloader, optimizer, loss_fn, device)
    print(f"  Train Loss: {train_loss:.4f}")

    val_loss, val_dice = test_step(model, test_dataloader, loss_fn, device, use_softmax=False)
    print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "best_model.pth")
        print("  Best model saved.")
