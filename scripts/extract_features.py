import json
import os

import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet
from src.utils.features import extract_crack_features
from src.utils.morphology import refine_crack_mask


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CrackDataset(
    image_dir="data/DeepCrack/test_img",
    mask_dir="data/DeepCrack/test_lab",
    image_size=256,
    use_clahe=False,
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

os.makedirs("outputs/features", exist_ok=True)

results = []

with torch.no_grad():
    for idx, batch in enumerate(loader):
        images = batch["image"].to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        pred_mask = preds[0][0].cpu().numpy()

        refined_mask = refine_crack_mask(pred_mask)

        features = extract_crack_features(refined_mask)

        item = {
            "index": idx,
            "image_path": batch["image_path"][0],
            "features": features,
        }

        results.append(item)

        if idx < 5:
            print(item)

output_path = "outputs/features/crack_features.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved features to {output_path}")