import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet


IMAGE_SIZE = 256
BATCH_SIZE = 4

TRAIN_IMAGE_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_img")
TRAIN_MASK_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_lab")

CHECKPOINT_PATH = Path("/media/rani/새 볼륨/crack-seg-project/outputs/checkpoints/unet_raw_best.pth")
SPLIT_PATH = Path("/media/rani/새 볼륨/crack-seg-project/outputs/checkpoints/train_val_split.json")

RESULT_DIR = Path("outputs/evaluation")
CSV_PATH = RESULT_DIR / "threshold_search.csv"
JSON_PATH = RESULT_DIR / "threshold_search.json"

THRESHOLDS = [
    round(value / 100, 2)
    for value in range(30, 71, 5)
]


def load_validation_paths(split_path: Path) -> list[Path]:
    if not split_path.exists():
        raise FileNotFoundError(
            f"분할 정보 파일이 없습니다: {split_path}\n"
            "먼저 train.py를 실행해 Train/Validation 분할을 생성하세요."
        )

    with split_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    val_images = split_data.get("val_images")
    if not val_images:
        raise ValueError(
            f"{split_path}에 val_images 정보가 없습니다."
        )

    paths = [Path(path) for path in val_images]

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            "Validation 이미지 경로 일부를 찾을 수 없습니다.\n"
            f"{preview}"
        )

    return paths


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[UNet, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"최적 모델 파일이 없습니다: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "현재 checkpoint 형식이 올바르지 않습니다."
        )

    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise KeyError(
            "checkpoint에 model_state_dict가 없습니다."
        )

    model = UNet().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model, checkpoint


def calculate_sample_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    return dice, iou


def evaluate_thresholds(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
) -> list[dict]:
    totals = {
        threshold: {
            "dice_sum": 0.0,
            "iou_sum": 0.0,
            "sample_count": 0,
        }
        for threshold in THRESHOLDS
    }

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

            for threshold in THRESHOLDS:
                dice, iou = calculate_sample_metrics(
                    probabilities,
                    masks,
                    threshold,
                )

                batch_size = images.size(0)
                totals[threshold]["dice_sum"] += dice.sum().item()
                totals[threshold]["iou_sum"] += iou.sum().item()
                totals[threshold]["sample_count"] += batch_size

    results = []

    for threshold in THRESHOLDS:
        values = totals[threshold]
        count = values["sample_count"]

        if count == 0:
            raise RuntimeError(
                "평가할 Validation 이미지가 없습니다."
            )

        results.append(
            {
                "threshold": threshold,
                "dice": values["dice_sum"] / count,
                "iou": values["iou_sum"] / count,
                "sample_count": count,
            }
        )

    return results


def save_results(
    results: list[dict],
    checkpoint: dict,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    best_result = max(
        results,
        key=lambda item: item["dice"],
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "dice",
                "iou",
                "sample_count",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "threshold": f"{result['threshold']:.2f}",
                    "dice": f"{result['dice']:.6f}",
                    "iou": f"{result['iou']:.6f}",
                    "sample_count": result["sample_count"],
                }
            )

    json_output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_dice_at_0_5": checkpoint.get("val_dice"),
        "validation_split": str(SPLIT_PATH),
        "best_threshold": best_result["threshold"],
        "best_dice": best_result["dice"],
        "best_iou": best_result["iou"],
        "results": results,
    }

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            json_output,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    val_paths = load_validation_paths(SPLIT_PATH)

    val_dataset = CrackDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        image_size=IMAGE_SIZE,
        use_clahe=False,
        image_paths=val_paths,
    )

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model, checkpoint = load_model(
        CHECKPOINT_PATH,
        device,
    )

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(f"Validation images: {len(val_dataset)}")
    print("Test 데이터는 사용하지 않습니다.\n")

    results = evaluate_thresholds(
        model,
        loader,
        device,
    )

    print("=== Threshold Search ===")
    print(f"{'Threshold':>10} | {'Dice':>8} | {'IoU':>8}")
    print("-" * 34)

    for result in results:
        print(
            f"{result['threshold']:>10.2f} | "
            f"{result['dice']:>8.4f} | "
            f"{result['iou']:>8.4f}"
        )

    best_result = max(
        results,
        key=lambda item: item["dice"],
    )

    print("\n=== Best Result ===")
    print(
        f"Best threshold: {best_result['threshold']:.2f}"
    )
    print(f"Validation Dice: {best_result['dice']:.4f}")
    print(f"Validation IoU : {best_result['iou']:.4f}")

    save_results(results, checkpoint)

    print(f"\nSaved CSV : {CSV_PATH}")
    print(f"Saved JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
