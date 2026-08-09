import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.resnet34_unet import ResNet34UNet
from src.utils.losses import BCEDiceLoss


# =========================================================
# Experiment configuration
# =========================================================

SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 4

STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 30

STAGE1_LR = 1e-3
ENCODER_LR = 1e-5
DECODER_LR = 1e-4

WEIGHT_DECAY = 1e-4

# 학습 중 비교용 threshold.
# 최종 threshold 0.55는 별도 Validation search에서 결정.
TRAIN_METRIC_THRESHOLD = 0.50

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


# =========================================================
# MLflow configuration
# =========================================================

MLFLOW_EXPERIMENT_NAME = (
    "crack-segmentation-resnet34"
)

MLFLOW_RUN_NAME = (
    "resnet34_unet_balanced_two_stage"
)


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
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
        for path in split_data[
            "train_images"
        ]
    ]

    val_paths = [
        Path(path)
        for path in split_data[
            "val_images"
        ]
    ]

    return train_paths, val_paths


def calculate_batch_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities
        > TRAIN_METRIC_THRESHOLD
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

    prediction_sum = (
        predictions.sum(dim=1)
    )

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
    is_training = (
        optimizer is not None
    )

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
            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            masks = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            if is_training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            dice, iou = (
                calculate_batch_metrics(
                    logits,
                    masks,
                )
            )

            batch_size = (
                images.size(0)
            )

            loss_sum += (
                loss.item()
                * batch_size
            )

            dice_sum += (
                dice.sum().item()
            )

            iou_sum += (
                iou.sum().item()
            )

            sample_count += (
                batch_size
            )

    return {
        "loss": (
            loss_sum
            / sample_count
        ),
        "dice": (
            dice_sum
            / sample_count
        ),
        "iou": (
            iou_sum
            / sample_count
        ),
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
            "val_dice": (
                val_metrics["dice"]
            ),
            "val_iou": (
                val_metrics["iou"]
            ),
            "val_loss": (
                val_metrics["loss"]
            ),
            "threshold": (
                TRAIN_METRIC_THRESHOLD
            ),
            "image_size": (
                IMAGE_SIZE
            ),
            "architecture": (
                "ResNet34UNet"
            ),
            "pretrained_encoder": (
                True
            ),
        },
        path,
    )


def log_epoch_to_mlflow(
    stage: int,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    global_step: int,
) -> None:
    """
    Epoch별 지표를 MLflow에 기록한다.
    """

    metrics = {
        (
            f"stage{stage}_"
            "train_loss"
        ): train_metrics["loss"],

        (
            f"stage{stage}_"
            "train_dice"
        ): train_metrics["dice"],

        (
            f"stage{stage}_"
            "train_iou"
        ): train_metrics["iou"],

        (
            f"stage{stage}_"
            "val_loss"
        ): val_metrics["loss"],

        (
            f"stage{stage}_"
            "val_dice"
        ): val_metrics["dice"],

        (
            f"stage{stage}_"
            "val_iou"
        ): val_metrics["iou"],
    }

    mlflow.log_metrics(
        metrics,
        step=global_step,
    )

    mlflow.set_tag(
        "last_stage",
        str(stage),
    )

    mlflow.set_tag(
        "last_epoch",
        str(epoch),
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
    global_step_offset: int,
) -> dict:
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = -1

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

        global_step = (
            global_step_offset
            + epoch
        )

        log_epoch_to_mlflow(
            stage=stage,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            global_step=global_step,
        )

        print(
            f"[Stage {stage}] "
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss "
            f"{train_metrics['loss']:.4f} | "
            f"Train Dice "
            f"{train_metrics['dice']:.4f} | "
            f"Val Loss "
            f"{val_metrics['loss']:.4f} | "
            f"Val Dice "
            f"{val_metrics['dice']:.4f} | "
            f"Val IoU "
            f"{val_metrics['iou']:.4f}"
        )

        if (
            val_metrics["dice"]
            > best_val_dice
        ):
            best_val_dice = (
                val_metrics["dice"]
            )

            best_val_iou = (
                val_metrics["iou"]
            )

            best_epoch = epoch

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                stage,
                val_metrics,
            )

            mlflow.log_metric(
                (
                    f"stage{stage}_"
                    "best_val_dice"
                ),
                best_val_dice,
                step=global_step,
            )

            mlflow.log_metric(
                (
                    f"stage{stage}_"
                    "best_val_iou"
                ),
                best_val_iou,
                step=global_step,
            )

            print(
                "  -> Saved best checkpoint: "
                f"{checkpoint_path}"
            )

    return {
        "best_val_dice": (
            best_val_dice
        ),
        "best_val_iou": (
            best_val_iou
        ),
        "best_epoch": (
            best_epoch
        ),
    }


def log_common_params(
    train_count: int,
    val_count: int,
    device: torch.device,
) -> None:
    mlflow.log_params(
        {
            "architecture": (
                "ResNet34UNet"
            ),
            "encoder": (
                "ResNet34"
            ),
            "encoder_pretrained": (
                True
            ),
            "training_strategy": (
                "freeze_unfreeze"
            ),
            "image_size": (
                IMAGE_SIZE
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "seed": SEED,
            "stage1_epochs": (
                STAGE1_EPOCHS
            ),
            "stage2_epochs": (
                STAGE2_EPOCHS
            ),
            "stage1_lr": (
                STAGE1_LR
            ),
            "encoder_lr": (
                ENCODER_LR
            ),
            "decoder_lr": (
                DECODER_LR
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "loss": (
                "BCE_0.5_Dice_0.5"
            ),
            "lighting_mode": (
                "mixed_balanced"
            ),
            "train_metric_threshold": (
                TRAIN_METRIC_THRESHOLD
            ),
            "train_images": (
                train_count
            ),
            "validation_images": (
                val_count
            ),
            "optimizer": (
                "AdamW"
            ),
            "device": str(
                device
            ),
        }
    )


def main() -> None:
    set_seed(SEED)
    setup_mlflow()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    train_paths, val_paths = (
        load_split_paths()
    )

    train_dataset = CrackDataset(
        image_dir=str(
            TRAIN_IMAGE_DIR
        ),
        mask_dir=str(
            TRAIN_MASK_DIR
        ),
        image_size=IMAGE_SIZE,
        use_clahe=False,
        image_paths=train_paths,
        lighting_mode=(
            "mixed_balanced"
        ),
    )

    val_dataset = CrackDataset(
        image_dir=str(
            TRAIN_IMAGE_DIR
        ),
        mask_dir=str(
            TRAIN_MASK_DIR
        ),
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
        f"Train images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    with mlflow.start_run(
        run_name=MLFLOW_RUN_NAME
    ) as run:
        print(
            "\nMLflow run ID: "
            f"{run.info.run_id}"
        )

        log_common_params(
            train_count=(
                len(train_dataset)
            ),
            val_count=(
                len(val_dataset)
            ),
            device=device,
        )

        mlflow.set_tags(
            {
                "project": (
                    "crack-segmentation"
                ),
                "dataset": (
                    "DeepCrack"
                ),
                "task": (
                    "semantic-segmentation"
                ),
                "framework": (
                    "PyTorch"
                ),
            }
        )

        # 데이터 분할 정보도 artifact로 기록
        if SPLIT_PATH.exists():
            mlflow.log_artifact(
                str(SPLIT_PATH),
                artifact_path="data_split",
            )

        model = ResNet34UNet(
            pretrained=True
        ).to(device)

        criterion = BCEDiceLoss(
            bce_weight=0.5,
            dice_weight=0.5,
        )

        print(
            "\n=== Stage 1: "
            "Freeze Encoder ==="
        )

        model.freeze_encoder()

        stage1_optimizer = (
            torch.optim.AdamW(
                filter(
                    lambda parameter: (
                        parameter.requires_grad
                    ),
                    model.parameters(),
                ),
                lr=STAGE1_LR,
                weight_decay=(
                    WEIGHT_DECAY
                ),
            )
        )

        stage1_result = train_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            stage1_optimizer,
            device,
            STAGE1_EPOCHS,
            stage=1,
            checkpoint_path=(
                STAGE1_CHECKPOINT
            ),
            global_step_offset=0,
        )

        print(
            "\nStage 1 best Val Dice: "
            f"{stage1_result['best_val_dice']:.4f}"
        )

        stage1_checkpoint = (
            torch.load(
                STAGE1_CHECKPOINT,
                map_location=device,
            )
        )

        model.load_state_dict(
            stage1_checkpoint[
                "model_state_dict"
            ]
        )

        # Stage 1 최고 checkpoint artifact
        mlflow.log_artifact(
            str(STAGE1_CHECKPOINT),
            artifact_path="checkpoints",
        )

        print(
            "\n=== Stage 2: "
            "Unfreeze Encoder ==="
        )

        model.unfreeze_encoder()

        stage2_optimizer = (
            torch.optim.AdamW(
                [
                    {
                        "params": list(
                            model.encoder_parameters()
                        ),
                        "lr": (
                            ENCODER_LR
                        ),
                    },
                    {
                        "params": list(
                            model.decoder_parameters()
                        ),
                        "lr": (
                            DECODER_LR
                        ),
                    },
                ],
                weight_decay=(
                    WEIGHT_DECAY
                ),
            )
        )

        stage2_result = train_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            stage2_optimizer,
            device,
            STAGE2_EPOCHS,
            stage=2,
            checkpoint_path=(
                FINAL_CHECKPOINT
            ),
            global_step_offset=(
                STAGE1_EPOCHS
            ),
        )

        # 최종 모델 checkpoint artifact
        mlflow.log_artifact(
            str(FINAL_CHECKPOINT),
            artifact_path="checkpoints",
        )

        mlflow.log_metrics(
            {
                "final_best_val_dice": (
                    stage2_result[
                        "best_val_dice"
                    ]
                ),
                "final_best_val_iou": (
                    stage2_result[
                        "best_val_iou"
                    ]
                ),
                "baseline_unet_val_dice": (
                    0.7816
                ),
                "baseline_unet_val_iou": (
                    0.6590
                ),
                "val_dice_improvement": (
                    stage2_result[
                        "best_val_dice"
                    ]
                    - 0.7816
                ),
                "val_iou_improvement": (
                    stage2_result[
                        "best_val_iou"
                    ]
                    - 0.6590
                ),
            }
        )

        mlflow.set_tag(
            "best_stage2_epoch",
            str(
                stage2_result[
                    "best_epoch"
                ]
            ),
        )

        print(
            "\n=== Training Complete ==="
        )

        print(
            "Stage 1 best Val Dice: "
            f"{stage1_result['best_val_dice']:.4f}"
        )

        print(
            "Stage 2 best Val Dice: "
            f"{stage2_result['best_val_dice']:.4f}"
        )

        print(
            "Stage 2 best Val IoU : "
            f"{stage2_result['best_val_iou']:.4f}"
        )

        print(
            "Final checkpoint: "
            f"{FINAL_CHECKPOINT}"
        )

        print(
            "\nMLflow experiment: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

        print(
            "MLflow tracking DB: mlflow.db"
        )


if __name__ == "__main__":
    main()
