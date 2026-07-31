import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet


SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
PATIENCE = 10
VAL_RATIO = 0.2
THRESHOLD = 0.5

TRAIN_IMAGE_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_img")
TRAIN_MASK_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_lab")
CHECKPOINT_DIR = Path("/media/rani/새 볼륨/crack-seg-project/outputs/checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "unet_raw_best.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "unet_raw_last.pth"
SPLIT_PATH = CHECKPOINT_DIR / "train_val_split.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 재현성을 우선한다. 일부 환경에서는 속도가 조금 느려질 수 있다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_image_paths(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

    if len(image_paths) < 2:
        raise ValueError("Train/Validation 분할에는 최소 2장의 이미지가 필요합니다.")

    return image_paths


def split_paths(
    image_paths: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    indices = np.arange(len(image_paths))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_count = max(1, int(round(len(image_paths) * val_ratio)))
    val_count = min(val_count, len(image_paths) - 1)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_paths = [image_paths[index] for index in train_indices]
    val_paths = [image_paths[index] for index in val_indices]

    return sorted(train_paths), sorted(val_paths)


def save_split(train_paths: list[Path], val_paths: list[Path]) -> None:
    split_data = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "train_images": [str(path) for path in train_paths],
        "val_images": [str(path) for path in val_paths],
    }

    with SPLIT_PATH.open("w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=2, ensure_ascii=False)


def calculate_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> tuple[float, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).float()

    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    smooth = 1e-6
    intersection = (predictions * targets).sum(dim=1)
    prediction_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (
        prediction_sum + target_sum + smooth
    )

    union = prediction_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)

    return dice.mean().item(), iou.mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)
            dice, iou = calculate_batch_metrics(outputs, masks, THRESHOLD)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_dice += dice * batch_size
            total_iou += iou * batch_size
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples,
    )


def main() -> None:
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_image_paths = collect_image_paths(TRAIN_IMAGE_DIR)
    train_paths, val_paths = split_paths(all_image_paths, VAL_RATIO, SEED)
    save_split(train_paths, val_paths)

    print(f"Train images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Saved split information: {SPLIT_PATH}")

    # 이번 단계에서는 Train과 Validation 모두 원본 입력을 사용한다.
    train_dataset = CrackDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        image_size=IMAGE_SIZE,
        use_clahe=False,
        image_paths=train_paths,
    )
    val_dataset = CrackDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        image_size=IMAGE_SIZE,
        use_clahe=False,
        image_paths=val_paths,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_dice, val_iou = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        torch.save(model.state_dict(), LAST_MODEL_PATH)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "threshold": THRESHOLD,
                "seed": SEED,
            }
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"  -> Best model saved: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(
                f"  -> No improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Best checkpoint: {BEST_MODEL_PATH}")
    print("Test 데이터는 아직 사용하지 않았습니다.")


if __name__ == "__main__":
    main()
