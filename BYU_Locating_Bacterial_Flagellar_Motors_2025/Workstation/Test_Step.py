import torch
import torch.nn.functional as F


def test_step(model, dataloader, criterion, device, use_softmax=True):
    model.eval()
    running_loss = 0.0
    dice_score_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            if use_softmax:
                probs = F.softmax(outputs, dim=1)  # (B, C, D, H, W)
                preds = torch.argmax(probs, dim=1)  # (B, D, H, W)
            else:
                # 예: 이진 세분화(채널=1)라면 sigmoid + threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()  # 0 또는 1

            dice_score = dice_coefficient(preds, targets)
            dice_score_sum += dice_score * inputs.size(0)

            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    avg_dice = dice_score_sum / total_samples
    return avg_loss, avg_dice


def dice_coefficient(preds, targets, eps=1e-6):
    preds = preds.float()
    targets = targets.float()

    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.item()
