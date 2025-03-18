import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Test_Step import test_step
from Train_Step import train_step
from TomoDataSet import TomoDataset
from Models.UNET3D import UNet3D
from Preprocess.process_data import build_data_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = r"/home/junha/Kaggle/BYU/Datasets/"

train_dataset = TomoDataset("/home/junha/Kaggle/BYU/Datasets/train_labels_train.csv", os.path.join(base_dir, "train"), transform=None)
test_dataset   = TomoDataset("/home/junha/Kaggle/BYU/Datasets/train_labels_val.csv", os.path.join(base_dir, "val"), transform=None)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
test_dataloader   = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
print("Dataloader load completed...")

model = UNet3D(in_channels=1, out_channels=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
NUM_EPOCHS = 20

best_dice = -1
for epoch in range(NUM_EPOCHS):
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

    train_loss = train_step(model, train_dataloader, optimizer, loss_fn, device)
    print(f"Train Loss: {train_loss:.4f}")

    val_loss, val_dice = test_step(model, test_dataloader, loss_fn, device, use_softmax=True)
    print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
    
    torch.save(model.state_dict(), f"/home/junha/Kaggle/BYU/CheckPoints/3D_UNET_{epoch}.pth")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), f"/home/junha/Kaggle/BYU/CheckPoints/best_model_{epoch}.pth")
        print("Best model saved.")
