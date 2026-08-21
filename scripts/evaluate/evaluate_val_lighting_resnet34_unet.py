import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.crack_dataset import CrackDataset
from src.models.resnet34_unet import ResNet34UNet


IMAGE_SIZE = 256
BATCH_SIZE = 4
THRESHOLD = 0.55

TRAIN_IMAGE_DIR = Path(
    "/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_img"
)
TRAIN_MASK_DIR = Path(
    "/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_lab"
)

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/"
    "resnet34_unet_balanced_best.pth"
)

SPLIT_PATH = Path(
    "outputs/checkpoints/"
    "train_val_split.json"
)

RESULT_DIR = Path(
    "outputs/evaluation"
)

CSV_PATH = (
    RESULT_DIR
    / "lighting_robustness_resnet34_unet.csv"
)

JSON_PATH = (
    RESULT_DIR
    / "lighting_robustness_resnet34_unet.json"
)


# 기존 기본 U-Net의 Validation 조명 평가 결과
# 이전 실험 결과와 새 모델을 바로 비교하기 위한 참고값
BASELINE_RESULTS = {
    "original": {
        "dice": 0.7804,
        "iou": 0.6577,
    },
    "low_light_50": {
        "dice": 0.7487,
        "iou": None,
    },
    "low_light_35": {
        "dice": 0.7193,
        "iou": None,
    },
    "low_light_25": {
        "dice": 0.6571,
        "iou": None,
    },
    "severe_overexposure": {
        "dice": 0.6985,
        "iou": None,
    },
}


class LightingConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        condition: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.condition = condition

        allowed = {
            "original",
            "low_light_50",
            "low_light_35",
            "low_light_25",
            "severe_overexposure",
        }

        if condition not in allowed:
            raise ValueError(
                f"Unsupported condition: {condition}"
            )

    def __len__(
        self,
    ) -> int:
        return len(self.base_dataset)

    def apply_condition(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if self.condition == "original":
            return image

        if self.condition == "low_light_50":
            return torch.clamp(
                image * 0.50,
                0.0,
                1.0,
            )

        if self.condition == "low_light_35":
            return torch.clamp(
                image * 0.35,
                0.0,
                1.0,
            )

        if self.condition == "low_light_25":
            return torch.clamp(
                image * 0.25,
                0.0,
                1.0,
            )

        if self.condition == "severe_overexposure":
            return torch.clamp(
                image * 1.35 + (90.0 / 255.0),
                0.0,
                1.0,
            )

        raise RuntimeError(
            f"Condition not handled: {self.condition}"
        )

    def __getitem__(
        self,
        index: int,
    ):
        sample = self.base_dataset[index]

        output = dict(sample)
        output["image"] = self.apply_condition(
            sample["image"]
        )

        return output


def load_validation_paths() -> list[Path]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}"
        )

    with SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    val_images = split_data.get(
        "val_images"
    )

    if not val_images:
        raise ValueError(
            "val_images not found in split file."
        )

    return [
        Path(path)
        for path in val_images
    ]


def load_model(
    device: torch.device,
) -> tuple[ResNet34UNet, dict]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = ResNet34UNet(
        pretrained=False
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def calculate_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = (
        probabilities > THRESHOLD
    ).float()

    predictions = predictions.flatten(
        start_dim=1
    )
    targets = targets.flatten(
        start_dim=1
    )

    smooth = 1e-6

    intersection = (
        predictions * targets
    ).sum(dim=1)

    prediction_sum = predictions.sum(
        dim=1
    )
    target_sum = targets.sum(
        dim=1
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum
        + target_sum
        + smooth
    )

    union = (
        prediction_sum
        + target_sum
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return dice, iou


def evaluate_condition(
    model: ResNet34UNet,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    dice_sum = 0.0
    iou_sum = 0.0
    sample_count = 0

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

            logits = model(images)
            probabilities = torch.sigmoid(
                logits
            )

            dice, iou = calculate_metrics(
                probabilities,
                masks,
            )

            dice_sum += dice.sum().item()
            iou_sum += iou.sum().item()
            sample_count += images.size(0)

    return {
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
        "sample_count": sample_count,
    }


def save_results(
    results: list[dict],
    checkpoint: dict,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "condition",
                "resnet34_dice",
                "resnet34_iou",
                "baseline_dice",
                "dice_improvement",
                "sample_count",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "condition": result[
                        "condition"
                    ],
                    "resnet34_dice": (
                        f"{result['dice']:.6f}"
                    ),
                    "resnet34_iou": (
                        f"{result['iou']:.6f}"
                    ),
                    "baseline_dice": (
                        ""
                        if result[
                            "baseline_dice"
                        ] is None
                        else (
                            f"{result['baseline_dice']:.6f}"
                        )
                    ),
                    "dice_improvement": (
                        ""
                        if result[
                            "dice_improvement"
                        ] is None
                        else (
                            f"{result['dice_improvement']:.6f}"
                        )
                    ),
                    "sample_count": result[
                        "sample_count"
                    ],
                }
            )

    output = {
        "model": "ResNet34UNet",
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "checkpoint_epoch": (
            checkpoint.get("epoch")
        ),
        "threshold": THRESHOLD,
        "validation_split": str(
            SPLIT_PATH
        ),
        "lighting_definition": {
            "original": "image",
            "low_light_50": "image * 0.50",
            "low_light_35": "image * 0.35",
            "low_light_25": "image * 0.25",
            "severe_overexposure": (
                "clamp(image * 1.35 + 90/255)"
            ),
        },
        "results": results,
    }

    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    val_paths = load_validation_paths()

    base_dataset = CrackDataset(
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

    model, checkpoint = load_model(
        device
    )

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(
        f"Threshold: {THRESHOLD:.2f}"
    )
    print(
        f"Validation images: "
        f"{len(base_dataset)}"
    )
    print(
        "Test 데이터는 사용하지 않습니다.\n"
    )

    conditions = [
        "original",
        "low_light_50",
        "low_light_35",
        "low_light_25",
        "severe_overexposure",
    ]

    results = []

    for condition in conditions:
        condition_dataset = (
            LightingConditionDataset(
                base_dataset,
                condition,
            )
        )

        loader = DataLoader(
            condition_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                device.type == "cuda"
            ),
        )

        metrics = evaluate_condition(
            model,
            loader,
            device,
        )

        baseline = BASELINE_RESULTS.get(
            condition,
            {},
        ).get("dice")

        improvement = (
            None
            if baseline is None
            else metrics["dice"] - baseline
        )

        results.append(
            {
                "condition": condition,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "baseline_dice": baseline,
                "dice_improvement": (
                    improvement
                ),
                "sample_count": (
                    metrics["sample_count"]
                ),
            }
        )

    print(
        "=== Lighting Robustness: "
        "ResNet-34 U-Net ==="
    )

    print(
        f"{'Condition':>23} | "
        f"{'Dice':>8} | "
        f"{'IoU':>8} | "
        f"{'Old Dice':>8} | "
        f"{'Change':>8}"
    )

    print("-" * 72)

    for result in results:
        old_text = (
            "-"
            if result["baseline_dice"] is None
            else (
                f"{result['baseline_dice']:.4f}"
            )
        )

        change_text = (
            "-"
            if result["dice_improvement"] is None
            else (
                f"{result['dice_improvement']:+.4f}"
            )
        )

        print(
            f"{result['condition']:>23} | "
            f"{result['dice']:>8.4f} | "
            f"{result['iou']:>8.4f} | "
            f"{old_text:>8} | "
            f"{change_text:>8}"
        )

    save_results(
        results,
        checkpoint,
    )

    print(
        f"\nSaved CSV : {CSV_PATH}"
    )
    print(
        f"Saved JSON: {JSON_PATH}"
    )


if __name__ == "__main__":
    main()
