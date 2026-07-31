import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet
from src.utils.losses import BCEDiceLoss


SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
PATIENCE = 10
THRESHOLD = 0.5

TRAIN_IMAGE_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_img")
TRAIN_MASK_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_lab")

CHECKPOINT_DIR = Path("outputs/checkpoints")
SPLIT_PATH = CHECKPOINT_DIR / "train_val_split.json"

BEST_MODEL_PATH = CHECKPOINT_DIR / "unet_bce_dice_best.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "unet_bce_dice_last.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(
    split_path: Path,
) -> tuple[list[Path], list[Path]]:
    if not split_path.exists():
        raise FileNotFoundError(
            f"분할 정보 파일이 없습니다: {split_path}\n"
            "먼저 기존 train.py를 실행해 Train/Validation 분할을 생성하세요."
        )

    with split_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    train_images = split_data.get("train_images")
    val_images = split_data.get("val_images")

    if not train_images or not val_images:
        raise ValueError(
            f"{split_path}에 train_images 또는 val_images 정보가 없습니다."
        )

    train_paths = [Path(path) for path in train_images]
    val_paths = [Path(path) for path in val_images]

    missing_paths = [
        str(path)
        for path in train_paths + val_paths
        if not path.exists()
    ]

    if missing_paths:
        preview = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            "분할 파일에 기록된 이미지 일부를 찾을 수 없습니다.\n"
            f"{preview}"
        )

    return train_paths, val_paths


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
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

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
            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            outputs = model(images)
            loss = criterion(outputs, masks)

            dice, iou = calculate_batch_metrics(
                outputs,
                masks,
                THRESHOLD,
            )

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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    train_paths, val_paths = load_split(SPLIT_PATH)

    print(f"Train images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Loaded fixed split: {SPLIT_PATH}")
    print("Loss: 0.5 × BCE + 0.5 × Dice Loss")

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

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

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

        torch.save(
            model.state_dict(),
            LAST_MODEL_PATH,
        )

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
                "loss_name": "BCEDiceLoss",
                "bce_weight": 0.5,
                "dice_weight": 0.5,
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print(
                f"  -> Best model saved: "
                f"{BEST_MODEL_PATH}"
            )
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
    print("기존 BCE 모델은 덮어쓰지 않았습니다.")
    print("Test 데이터는 아직 사용하지 않았습니다.")


if __name__ == "__main__":
    main()
