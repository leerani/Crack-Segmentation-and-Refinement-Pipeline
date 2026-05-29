import os

import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CrackDataset(
    image_dir="data/DeepCrack/test_img",
    mask_dir="data/DeepCrack/test_lab",
    image_size=256,
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

os.makedirs("outputs/predictions", exist_ok=True)

with torch.no_grad():

    for idx, batch in enumerate(loader):

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        image = images[0].cpu().permute(1, 2, 0).numpy()
        gt_mask = masks[0][0].cpu().numpy()
        pred_mask = preds[0][0].cpu().numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("GT Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()

        save_path = f"outputs/predictions/pred_{idx:03d}.png"

        plt.savefig(save_path)

        plt.close()

        print(f"Saved: {save_path}")

        if idx == 9:
            break