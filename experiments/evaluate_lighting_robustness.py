import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.unet import UNet


IMAGE_SIZE = 256
BATCH_SIZE = 4
THRESHOLD = 0.5

TRAIN_MASK_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_lab")
SPLIT_PATH = Path("outputs/checkpoints/train_val_split.json")

MODEL_CONFIGS = {
    "bce_dice": {
        "checkpoint": Path(
            "outputs/checkpoints/unet_bce_dice_best.pth"
        ),
        "label": "BCE + Dice",
    },
    "bce_dice_lighting": {
        "checkpoint": Path(
            "outputs/checkpoints/unet_bce_dice_lighting_best.pth"
        ),
        "label": "BCE + Dice + Lighting Augmentation",
    },
}

RESULT_DIR = Path("outputs/evaluation")
CSV_PATH = RESULT_DIR / "lighting_robustness_comparison.csv"
JSON_PATH = RESULT_DIR / "lighting_robustness_comparison.json"

CONDITIONS = {
    "original": "Original",
    "dark": "Severe low-light",
    "bright": "Severe overexposure",
}


class LightingRobustnessDataset(Dataset):
    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        image_paths: list[Path],
        mask_dir: Path,
        condition: str,
        image_size: int = 256,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(
                f"Unsupported condition: {condition}"
            )

        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.condition = condition
        self.image_size = image_size

        self.samples = []

        for image_path in image_paths:
            mask_path = self._find_mask_path(image_path)
            self.samples.append((image_path, mask_path))

    def _find_mask_path(
        self,
        image_path: Path,
    ) -> Path:
        for extension in self.SUPPORTED_EXTENSIONS:
            candidate = (
                self.mask_dir
                / f"{image_path.stem}{extension}"
            )
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Mask not found for image: {image_path.name}"
        )

    @staticmethod
    def apply_condition(
        image: np.ndarray,
        condition: str,
    ) -> np.ndarray:
        image_float = image.astype(np.float32)

        if condition == "original":
            transformed = image_float

        elif condition == "dark":
            # 심한 저조도: 밝기를 25% 수준으로 낮춘다.
            # 구조가 완전히 사라지는 순수 검정 이미지는 사용하지 않는다.
            transformed = image_float * 0.25

        elif condition == "bright":
            # 심한 과노출: 대비가 줄고 밝은 영역이 포화되도록 만든다.
            transformed = image_float * 1.35 + 90.0

        else:
            raise ValueError(
                f"Unsupported condition: {condition}"
            )

        return np.clip(
            transformed,
            0,
            255,
        ).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise ValueError(
                f"Failed to read image: {image_path}"
            )

        if mask is None:
            raise ValueError(
                f"Failed to read mask: {mask_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = self.apply_condition(
            image,
            self.condition,
        )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = (
            image.astype(np.float32) / 255.0
        )
        mask = (
            mask > 127
        ).astype(np.float32)

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "image_path": str(image_path),
            "condition": self.condition,
        }


def load_validation_paths(
    split_path: Path,
) -> list[Path]:
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    with split_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    val_images = split_data.get("val_images")

    if not val_images:
        raise ValueError(
            f"{split_path} does not contain val_images."
        )

    paths = [Path(path) for path in val_images]

    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            "Some validation images were not found:\n"
            f"{preview}"
        )

    return paths


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[UNet, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Invalid checkpoint format: {checkpoint_path}"
        )

    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        raise KeyError(
            f"model_state_dict not found in {checkpoint_path}"
        )

    model = UNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def calculate_sample_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = (
        probabilities > threshold
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

    prediction_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

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


def evaluate_model(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    dice_sum = 0.0
    iou_sum = 0.0
    sample_count = 0

    per_image_results = []

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
            probabilities = torch.sigmoid(logits)

            dice, iou = calculate_sample_metrics(
                probabilities,
                masks,
                THRESHOLD,
            )

            batch_size = images.size(0)

            dice_sum += dice.sum().item()
            iou_sum += iou.sum().item()
            sample_count += batch_size

            for index in range(batch_size):
                per_image_results.append(
                    {
                        "image_path": batch["image_path"][index],
                        "dice": float(dice[index].item()),
                        "iou": float(iou[index].item()),
                    }
                )

    if sample_count == 0:
        raise RuntimeError(
            "No validation images were evaluated."
        )

    return {
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
        "sample_count": sample_count,
        "per_image_results": per_image_results,
    }


def calculate_drop(
    original_value: float,
    changed_value: float,
) -> float:
    return original_value - changed_value


def save_results(
    summary_rows: list[dict],
    detailed_output: dict,
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
                "model",
                "condition",
                "threshold",
                "dice",
                "iou",
                "dice_drop_from_original",
                "iou_drop_from_original",
                "sample_count",
            ],
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "condition": row["condition"],
                    "threshold": f"{row['threshold']:.2f}",
                    "dice": f"{row['dice']:.6f}",
                    "iou": f"{row['iou']:.6f}",
                    "dice_drop_from_original": (
                        f"{row['dice_drop_from_original']:.6f}"
                    ),
                    "iou_drop_from_original": (
                        f"{row['iou_drop_from_original']:.6f}"
                    ),
                    "sample_count": row["sample_count"],
                }
            )

    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            detailed_output,
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
    print(f"Threshold: {THRESHOLD:.2f}")

    val_paths = load_validation_paths(
        SPLIT_PATH
    )

    print(f"Validation images: {len(val_paths)}")
    print("Test 데이터는 사용하지 않습니다.\n")

    all_results = {}
    summary_rows = []

    for model_key, config in MODEL_CONFIGS.items():
        model, checkpoint = load_model(
            config["checkpoint"],
            device,
        )

        print(
            f"=== Model: {config['label']} ==="
        )
        print(
            f"Checkpoint epoch: "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        model_results = {}

        for condition_key, condition_label in CONDITIONS.items():
            dataset = LightingRobustnessDataset(
                image_paths=val_paths,
                mask_dir=TRAIN_MASK_DIR,
                condition=condition_key,
                image_size=IMAGE_SIZE,
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=(
                    device.type == "cuda"
                ),
            )

            result = evaluate_model(
                model,
                loader,
                device,
            )

            model_results[condition_key] = result

            print(
                f"{condition_label:>22} | "
                f"Dice: {result['dice']:.4f} | "
                f"IoU: {result['iou']:.4f}"
            )

        original_dice = model_results["original"]["dice"]
        original_iou = model_results["original"]["iou"]

        for condition_key, result in model_results.items():
            summary_rows.append(
                {
                    "model": model_key,
                    "condition": condition_key,
                    "threshold": THRESHOLD,
                    "dice": result["dice"],
                    "iou": result["iou"],
                    "dice_drop_from_original": calculate_drop(
                        original_dice,
                        result["dice"],
                    ),
                    "iou_drop_from_original": calculate_drop(
                        original_iou,
                        result["iou"],
                    ),
                    "sample_count": result["sample_count"],
                }
            )

        all_results[model_key] = {
            "label": config["label"],
            "checkpoint": str(config["checkpoint"]),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "results": model_results,
        }

        print()

    print("=== Robustness Comparison ===")

    for condition_key, condition_label in CONDITIONS.items():
        base = all_results["bce_dice"]["results"][condition_key]
        augmented = all_results["bce_dice_lighting"]["results"][condition_key]

        dice_gain = augmented["dice"] - base["dice"]
        iou_gain = augmented["iou"] - base["iou"]

        print(
            f"{condition_label:>22} | "
            f"Dice gain: {dice_gain:+.4f} | "
            f"IoU gain: {iou_gain:+.4f}"
        )

    detailed_output = {
        "threshold": THRESHOLD,
        "validation_split": str(SPLIT_PATH),
        "condition_definition": {
            "original": "원본 이미지",
            "dark": "RGB 값을 25%로 감소시킨 심한 저조도 조건",
            "bright": (
                "RGB에 1.35배와 +90 밝기를 적용한 "
                "심한 과노출 조건"
            ),
        },
        "models": all_results,
    }

    save_results(
        summary_rows,
        detailed_output,
    )

    print(f"\nSaved CSV : {CSV_PATH}")
    print(f"Saved JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
