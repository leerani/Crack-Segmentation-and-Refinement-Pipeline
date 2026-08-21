import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.resnet34_unet import ResNet34UNet
from src.utils.losses import BCEDiceLoss


SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 4

STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 30

STAGE1_LR = 1e-3
ENCODER_LR = 1e-5
DECODER_LR = 1e-4

THRESHOLD = 0.50

TRAIN_IMAGE_DIR = Path(
    "/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_img"
)
TRAIN_MASK_DIR = Path(
    "/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_lab"
)

SPLIT_PATH = Path(
    "outputs/checkpoints/train_val_split.json"
)

CHECKPOINT_DIR = Path(
    "outputs/checkpoints"
)

STAGE1_CHECKPOINT = (
    CHECKPOINT_DIR
    / "resnet34_unet_stage1_best.pth"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "resnet34_unet_balanced_best.pth"
)


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_paths():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}"
        )

    with SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    train_paths = [
        Path(path)
        for path in split_data["train_images"]
    ]

    val_paths = [
        Path(path)
        for path in split_data["val_images"]
    ]

    return train_paths, val_paths


def calculate_batch_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > THRESHOLD
    ).float()

    predictions = predictions.flatten(
        start_dim=1
    )
    masks = masks.flatten(
        start_dim=1
    )

    smooth = 1e-6

    intersection = (
        predictions * masks
    ).sum(dim=1)

    prediction_sum = predictions.sum(dim=1)
    mask_sum = masks.sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum
        + mask_sum
        + smooth
    )

    union = (
        prediction_sum
        + mask_sum
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return dice, iou


def run_epoch(
    model: ResNet34UNet,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    device: torch.device,
    optimizer=None,
) -> dict:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    sample_count = 0

    context = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with context:
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            if is_training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            dice, iou = calculate_batch_metrics(
                logits,
                masks,
            )

            batch_size = images.size(0)

            loss_sum += (
                loss.item() * batch_size
            )
            dice_sum += dice.sum().item()
            iou_sum += iou.sum().item()
            sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
    }


def save_checkpoint(
    path: Path,
    model: ResNet34UNet,
    optimizer,
    epoch: int,
    stage: int,
    val_metrics: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "threshold": THRESHOLD,
            "image_size": IMAGE_SIZE,
            "architecture": (
                "ResNet34UNet"
            ),
            "pretrained_encoder": True,
        },
        path,
    )


def train_stage(
    model: ResNet34UNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: BCEDiceLoss,
    optimizer,
    device: torch.device,
    epochs: int,
    stage: int,
    checkpoint_path: Path,
) -> float:
    best_val_dice = -1.0

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"[Stage {stage}] "
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Train Dice {train_metrics['dice']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Dice {val_metrics['dice']:.4f} | "
            f"Val IoU {val_metrics['iou']:.4f}"
        )

        if (
            val_metrics["dice"]
            > best_val_dice
        ):
            best_val_dice = (
                val_metrics["dice"]
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                stage,
                val_metrics,
            )

            print(
                f"  -> Saved best checkpoint: "
                f"{checkpoint_path}"
            )

    return best_val_dice


def main() -> None:
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    train_paths, val_paths = (
        load_split_paths()
    )

    train_dataset = CrackDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        image_size=IMAGE_SIZE,
        use_clahe=False,
        image_paths=train_paths,
        lighting_mode="mixed_balanced",
    )

    val_dataset = CrackDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        image_size=IMAGE_SIZE,
        use_clahe=False,
        image_paths=val_paths,
        lighting_mode="none",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        f"Train images: {len(train_dataset)}"
    )
    print(
        f"Validation images: {len(val_dataset)}"
    )

    model = ResNet34UNet(
        pretrained=True
    ).to(device)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    print(
        "\n=== Stage 1: Freeze Encoder ==="
    )

    model.freeze_encoder()

    stage1_optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: (
                parameter.requires_grad
            ),
            model.parameters(),
        ),
        lr=STAGE1_LR,
        weight_decay=1e-4,
    )

    stage1_best = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        stage1_optimizer,
        device,
        STAGE1_EPOCHS,
        stage=1,
        checkpoint_path=STAGE1_CHECKPOINT,
    )

    print(
        f"\nStage 1 best Val Dice: "
        f"{stage1_best:.4f}"
    )

    stage1_checkpoint = torch.load(
        STAGE1_CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        stage1_checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "\n=== Stage 2: Unfreeze Encoder ==="
    )

    model.unfreeze_encoder()

    stage2_optimizer = torch.optim.AdamW(
        [
            {
                "params": list(
                    model.encoder_parameters()
                ),
                "lr": ENCODER_LR,
            },
            {
                "params": list(
                    model.decoder_parameters()
                ),
                "lr": DECODER_LR,
            },
        ],
        weight_decay=1e-4,
    )

    stage2_best = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        stage2_optimizer,
        device,
        STAGE2_EPOCHS,
        stage=2,
        checkpoint_path=FINAL_CHECKPOINT,
    )

    print(
        "\n=== Training Complete ==="
    )
    print(
        f"Stage 1 best Val Dice: "
        f"{stage1_best:.4f}"
    )
    print(
        f"Stage 2 best Val Dice: "
        f"{stage2_best:.4f}"
    )
    print(
        f"Final checkpoint: "
        f"{FINAL_CHECKPOINT}"
    )
    print(
        "\nCurrent basic U-Net reference:"
    )
    print(
        "Val Dice 0.7816 / "
        "Val IoU 0.6590"
    )


if __name__ == "__main__":
    main()
