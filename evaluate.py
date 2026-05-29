import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet
from src.utils.metrics import dice_score, iou_score
from src.utils.morphology import refine_crack_mask


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CrackDataset(
    image_dir="data/DeepCrack/test_img",
    mask_dir="data/DeepCrack/test_lab",
    image_size=256,
    use_clahe=True,
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/unet_raw.pth",
        map_location=device,
    )
)

model.eval()

baseline_dices = []
baseline_ious = []

refined_dices = []
refined_ious = []

with torch.no_grad():

    for batch in loader:

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        pred_mask = preds[0][0].cpu().numpy()
        gt_mask = masks[0][0].cpu().numpy()

        refined_mask = refine_crack_mask(pred_mask)

        # baseline
        baseline_dices.append(
            dice_score(pred_mask, gt_mask)
        )

        baseline_ious.append(
            iou_score(pred_mask, gt_mask)
        )

        # refined
        refined_dices.append(
            dice_score(refined_mask, gt_mask)
        )

        refined_ious.append(
            iou_score(refined_mask, gt_mask)
        )

print("\n=== Baseline ===")
print(f"Dice: {np.mean(baseline_dices):.4f}")
print(f"IoU : {np.mean(baseline_ious):.4f}")

print("\n=== Refined ===")
print(f"Dice: {np.mean(refined_dices):.4f}")
print(f"IoU : {np.mean(refined_ious):.4f}")