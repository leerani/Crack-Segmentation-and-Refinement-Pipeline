import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet


IMAGE_SIZE = 256
BATCH_SIZE = 4
THRESHOLD = 0.65

TRAIN_IMAGE_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_img")
TRAIN_MASK_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/train_lab")

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/"
    "unet_bce_dice_balanced_lighting_best.pth"
)
SPLIT_PATH = Path(
    "outputs/checkpoints/train_val_split.json"
)

RESULT_DIR = Path("outputs/evaluation")
CSV_PATH = RESULT_DIR / "morphology_validation_comparison.csv"
JSON_PATH = RESULT_DIR / "morphology_validation_comparison.json"

MIN_AREAS = [3, 5, 10, 20]


def load_validation_paths(
    split_path: Path,
) -> list[Path]:
    if not split_path.exists():
        raise FileNotFoundError(
            f"분할 정보 파일이 없습니다: {split_path}"
        )

    with split_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    val_images = split_data.get("val_images")

    if not val_images:
        raise ValueError(
            f"{split_path}에 val_images 정보가 없습니다."
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
            "Validation 이미지 일부를 찾을 수 없습니다.\n"
            f"{preview}"
        )

    return paths


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[UNet, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "체크포인트 형식이 올바르지 않습니다."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:
        raise KeyError(
            "checkpoint에 model_state_dict가 없습니다."
        )

    model = UNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    mask_uint8 = (
        mask > 0
    ).astype(np.uint8)

    (
        num_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    cleaned = np.zeros_like(
        mask_uint8
    )

    for label in range(
        1,
        num_labels,
    ):
        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area:
            cleaned[
                labels == label
            ] = 1

    return cleaned.astype(np.float32)


def closing_then_remove(
    mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    mask_uint8 = (
        mask > 0
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    closed = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    return remove_small_components(
        (closed > 0).astype(np.float32),
        min_area=min_area,
    )


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)

    intersection = np.sum(
        prediction * target
    )

    return float(
        (
            2.0 * intersection
            + smooth
        )
        / (
            np.sum(prediction)
            + np.sum(target)
            + smooth
        )
    )


def iou_score(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)

    intersection = np.sum(
        prediction * target
    )

    union = (
        np.sum(prediction)
        + np.sum(target)
        - intersection
    )

    return float(
        (
            intersection + smooth
        )
        / (
            union + smooth
        )
    )


def build_methods():
    methods = [
        {
            "name": "none",
            "label": "No post-processing",
            "type": "none",
            "min_area": None,
        }
    ]

    for min_area in MIN_AREAS:
        methods.append(
            {
                "name": f"remove_{min_area}",
                "label": (
                    f"Remove small components "
                    f"(min_area={min_area})"
                ),
                "type": "remove",
                "min_area": min_area,
            }
        )

    for min_area in MIN_AREAS:
        methods.append(
            {
                "name": (
                    f"closing_remove_{min_area}"
                ),
                "label": (
                    f"3x3 closing + remove "
                    f"(min_area={min_area})"
                ),
                "type": "closing_remove",
                "min_area": min_area,
            }
        )

    return methods


def apply_method(
    mask: np.ndarray,
    method: dict,
) -> np.ndarray:
    if method["type"] == "none":
        return mask.astype(np.float32)

    if method["type"] == "remove":
        return remove_small_components(
            mask,
            min_area=method["min_area"],
        )

    if method["type"] == "closing_remove":
        return closing_then_remove(
            mask,
            min_area=method["min_area"],
        )

    raise ValueError(
        f"Unknown method type: {method['type']}"
    )


def evaluate_methods(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    methods: list[dict],
) -> list[dict]:
    totals = {
        method["name"]: {
            "dice_sum": 0.0,
            "iou_sum": 0.0,
            "sample_count": 0,
        }
        for method in methods
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            masks = batch["mask"].cpu().numpy()

            logits = model(images)
            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities > THRESHOLD
            ).float().cpu().numpy()

            batch_size = predictions.shape[0]

            for index in range(batch_size):
                pred_mask = predictions[
                    index,
                    0,
                ]

                gt_mask = masks[
                    index,
                    0,
                ]

                for method in methods:
                    processed = apply_method(
                        pred_mask,
                        method,
                    )

                    totals[
                        method["name"]
                    ]["dice_sum"] += dice_score(
                        processed,
                        gt_mask,
                    )

                    totals[
                        method["name"]
                    ]["iou_sum"] += iou_score(
                        processed,
                        gt_mask,
                    )

                    totals[
                        method["name"]
                    ]["sample_count"] += 1

    results = []

    for method in methods:
        values = totals[
            method["name"]
        ]
        count = values[
            "sample_count"
        ]

        results.append(
            {
                "name": method["name"],
                "label": method["label"],
                "type": method["type"],
                "min_area": method["min_area"],
                "dice": (
                    values["dice_sum"] / count
                ),
                "iou": (
                    values["iou_sum"] / count
                ),
                "sample_count": count,
            }
        )

    return results


def save_results(
    results: list[dict],
    checkpoint: dict,
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = next(
        result
        for result in results
        if result["name"] == "none"
    )

    best_dice = max(
        results,
        key=lambda item: item["dice"],
    )

    best_iou = max(
        results,
        key=lambda item: item["iou"],
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "label",
                "type",
                "min_area",
                "dice",
                "iou",
                "dice_change_vs_none",
                "iou_change_vs_none",
                "sample_count",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "name": result["name"],
                    "label": result["label"],
                    "type": result["type"],
                    "min_area": result["min_area"],
                    "dice": f"{result['dice']:.6f}",
                    "iou": f"{result['iou']:.6f}",
                    "dice_change_vs_none": (
                        f"{result['dice'] - baseline['dice']:+.6f}"
                    ),
                    "iou_change_vs_none": (
                        f"{result['iou'] - baseline['iou']:+.6f}"
                    ),
                    "sample_count": result["sample_count"],
                }
            )

    output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": THRESHOLD,
        "validation_split": str(SPLIT_PATH),
        "note": (
            "Validation-only morphology comparison. "
            "Official Test result remains unchanged."
        ),
        "baseline": baseline,
        "best_dice_result": best_dice,
        "best_iou_result": best_iou,
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
    print(f"Threshold: {THRESHOLD:.2f}")

    val_paths = load_validation_paths(
        SPLIT_PATH
    )

    dataset = CrackDataset(
        image_dir=str(TRAIN_IMAGE_DIR),
        mask_dir=str(TRAIN_MASK_DIR),
        image_size=IMAGE_SIZE,
        use_clahe=False,
        image_paths=val_paths,
        lighting_mode="none",
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

    model, checkpoint = load_model(
        CHECKPOINT_PATH,
        device,
    )

    methods = build_methods()

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(
        f"Validation images: "
        f"{len(dataset)}"
    )
    print(
        "Test 데이터는 사용하지 않습니다."
    )
    print(
        "공식 Test 결과는 변경하지 않습니다.\n"
    )

    results = evaluate_methods(
        model,
        loader,
        device,
        methods,
    )

    baseline = next(
        result
        for result in results
        if result["name"] == "none"
    )

    print("=== Morphology Validation Comparison ===")
    print(
        f"{'Method':<46} | "
        f"{'Dice':>8} | "
        f"{'IoU':>8} | "
        f"{'ΔDice':>8} | "
        f"{'ΔIoU':>8}"
    )
    print("-" * 90)

    for result in results:
        print(
            f"{result['label']:<46} | "
            f"{result['dice']:>8.4f} | "
            f"{result['iou']:>8.4f} | "
            f"{result['dice'] - baseline['dice']:>+8.4f} | "
            f"{result['iou'] - baseline['iou']:>+8.4f}"
        )

    best_dice = max(
        results,
        key=lambda item: item["dice"],
    )

    best_iou = max(
        results,
        key=lambda item: item["iou"],
    )

    print("\n=== Best Dice Result ===")
    print(best_dice["label"])
    print(f"Dice: {best_dice['dice']:.4f}")
    print(f"IoU : {best_dice['iou']:.4f}")

    print("\n=== Best IoU Result ===")
    print(best_iou["label"])
    print(f"Dice: {best_iou['dice']:.4f}")
    print(f"IoU : {best_iou['iou']:.4f}")

    save_results(
        results,
        checkpoint,
    )

    print(f"\nSaved CSV : {CSV_PATH}")
    print(f"Saved JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
