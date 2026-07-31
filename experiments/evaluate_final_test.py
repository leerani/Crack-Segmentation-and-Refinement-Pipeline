import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.unet import UNet


IMAGE_SIZE = 256
BATCH_SIZE = 4
THRESHOLD = 0.65

TEST_IMAGE_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/test_img")
TEST_MASK_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/test_lab")

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/"
    "unet_bce_dice_balanced_lighting_best.pth"
)

RESULT_DIR = Path("outputs/final_test")
CSV_PATH = RESULT_DIR / "final_test_metrics.csv"
JSON_PATH = RESULT_DIR / "final_test_metrics.json"
VIS_DIR = RESULT_DIR / "visualizations"

CONDITIONS = {
    "original": {
        "label": "Original",
        "type": "original",
        "value": 1.0,
        "official": True,
    },
    "dark_50": {
        "label": "Low-light 50%",
        "type": "dark",
        "value": 0.50,
        "official": False,
    },
    "dark_35": {
        "label": "Low-light 35%",
        "type": "dark",
        "value": 0.35,
        "official": False,
    },
    "dark_25": {
        "label": "Low-light 25%",
        "type": "dark",
        "value": 0.25,
        "official": False,
    },
    "bright": {
        "label": "Severe overexposure",
        "type": "bright",
        "value": None,
        "official": False,
    },
}


class FinalTestDataset(Dataset):
    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        condition_key: str,
        image_size: int = 256,
    ) -> None:
        if condition_key not in CONDITIONS:
            raise ValueError(
                f"Unsupported condition: {condition_key}"
            )

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.condition_key = condition_key
        self.condition = CONDITIONS[condition_key]
        self.image_size = image_size

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        if not self.image_paths:
            raise ValueError(
                f"No test images found in {self.image_dir}"
            )

        self.samples = []

        for image_path in self.image_paths:
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
        condition: dict,
    ) -> np.ndarray:
        image_float = image.astype(np.float32)

        condition_type = condition["type"]

        if condition_type == "original":
            transformed = image_float

        elif condition_type == "dark":
            transformed = (
                image_float * float(condition["value"])
            )

        elif condition_type == "bright":
            transformed = (
                image_float * 1.35 + 90.0
            )

        else:
            raise ValueError(
                f"Unsupported condition type: {condition_type}"
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

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image_bgr is None:
            raise ValueError(
                f"Failed to read image: {image_path}"
            )

        if mask is None:
            raise ValueError(
                f"Failed to read mask: {mask_path}"
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        transformed_rgb = self.apply_condition(
            image_rgb,
            self.condition,
        )

        transformed_rgb = cv2.resize(
            transformed_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image_tensor = (
            transformed_rgb.astype(np.float32) / 255.0
        )
        image_tensor = np.transpose(
            image_tensor,
            (2, 0, 1),
        )

        mask_tensor = (
            mask > 127
        ).astype(np.float32)

        return {
            "image": torch.from_numpy(
                image_tensor
            ).float(),
            "mask": torch.from_numpy(
                mask_tensor
            ).float().unsqueeze(0),
            "image_path": str(image_path),
        }


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
            "Checkpoint format is invalid."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:
        raise KeyError(
            "model_state_dict not found in checkpoint."
        )

    model = UNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def calculate_sample_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = (
        probabilities > THRESHOLD
    ).float()

    predictions_flat = predictions.flatten(
        start_dim=1
    )
    targets_flat = targets.flatten(
        start_dim=1
    )

    smooth = 1e-6

    intersection = (
        predictions_flat * targets_flat
    ).sum(dim=1)

    prediction_sum = predictions_flat.sum(dim=1)
    target_sum = targets_flat.sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum + target_sum + smooth
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
    model: UNet,
    dataset: FinalTestDataset,
    device: torch.device,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

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
            )

            predictions = (
                probabilities > THRESHOLD
            ).float()

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

    return {
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
        "sample_count": sample_count,
        "per_image_results": per_image_results,
    }


def save_visualization(
    model: UNet,
    image_path: Path,
    mask_path: Path,
    save_path: Path,
    device: torch.device,
) -> None:
    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    resized_image = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    resized_mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    tensor = (
        resized_image.astype(np.float32) / 255.0
    )
    tensor = np.transpose(
        tensor,
        (2, 0, 1),
    )

    tensor = (
        torch.from_numpy(tensor)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor)
        probability = torch.sigmoid(logits)
        prediction = (
            probability > THRESHOLD
        ).float()

    pred_mask = prediction[0, 0].cpu().numpy()
    gt_mask = (
        resized_mask > 127
    ).astype(np.float32)

    overlay = resized_image.copy()
    overlay[pred_mask > 0] = (
        0.6 * overlay[pred_mask > 0]
        + 0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(resized_image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title("Prediction Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def save_outputs(
    condition_results: dict,
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
                "official_result",
                "threshold",
                "dice",
                "iou",
                "sample_count",
            ],
        )

        writer.writeheader()

        for condition_key, result in condition_results.items():
            writer.writerow(
                {
                    "condition": condition_key,
                    "official_result": (
                        CONDITIONS[condition_key]["official"]
                    ),
                    "threshold": f"{THRESHOLD:.2f}",
                    "dice": f"{result['dice']:.6f}",
                    "iou": f"{result['iou']:.6f}",
                    "sample_count": result["sample_count"],
                }
            )

    output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": THRESHOLD,
        "official_test_result": {
            "condition": "original",
            "dice": condition_results["original"]["dice"],
            "iou": condition_results["original"]["iou"],
            "sample_count": (
                condition_results["original"]["sample_count"]
            ),
        },
        "robustness_reference_results": {
            key: value
            for key, value in condition_results.items()
            if key != "original"
        },
        "condition_definition": {
            "original": "원본 Test 이미지",
            "dark_50": "RGB 값을 50%로 감소",
            "dark_35": "RGB 값을 35%로 감소",
            "dark_25": "RGB 값을 25%로 감소",
            "bright": (
                "RGB에 1.35배와 +90 밝기를 적용한 "
                "심한 과노출 조건"
            ),
        },
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
    print(f"Final threshold: {THRESHOLD:.2f}")
    print(
        "이 실행 결과를 확인한 뒤 "
        "threshold와 모델 설정을 변경하지 않습니다.\n"
    )

    model, checkpoint = load_model(
        CHECKPOINT_PATH,
        device,
    )

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    condition_results = {}

    for condition_key, condition_config in CONDITIONS.items():
        dataset = FinalTestDataset(
            image_dir=TEST_IMAGE_DIR,
            mask_dir=TEST_MASK_DIR,
            condition_key=condition_key,
            image_size=IMAGE_SIZE,
        )

        result = evaluate_condition(
            model,
            dataset,
            device,
        )

        condition_results[condition_key] = result

        result_type = (
            "OFFICIAL"
            if condition_config["official"]
            else "REFERENCE"
        )

        print(
            f"[{result_type}] "
            f"{condition_config['label']:>22} | "
            f"Dice: {result['dice']:.4f} | "
            f"IoU: {result['iou']:.4f} | "
            f"Images: {result['sample_count']}"
        )

    save_outputs(
        condition_results,
        checkpoint,
    )

    original_results = condition_results["original"][
        "per_image_results"
    ]

    ranked = sorted(
        original_results,
        key=lambda item: item["dice"],
    )

    selected = {
        "worst": ranked[0],
        "median": ranked[len(ranked) // 2],
        "best": ranked[-1],
    }

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for label, item in selected.items():
        image_path = Path(item["image_path"])

        mask_path = None
        for extension in FinalTestDataset.SUPPORTED_EXTENSIONS:
            candidate = (
                TEST_MASK_DIR
                / f"{image_path.stem}{extension}"
            )
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(
                f"Mask not found for {image_path.name}"
            )

        save_path = (
            VIS_DIR
            / f"{label}_dice_{item['dice']:.4f}.png"
        )

        save_visualization(
            model,
            image_path,
            mask_path,
            save_path,
            device,
        )

    print("\n=== Official Final Test Result ===")
    print(
        f"Dice: "
        f"{condition_results['original']['dice']:.4f}"
    )
    print(
        f"IoU : "
        f"{condition_results['original']['iou']:.4f}"
    )
    print(
        f"Images: "
        f"{condition_results['original']['sample_count']}"
    )

    print(f"\nSaved CSV : {CSV_PATH}")
    print(f"Saved JSON: {JSON_PATH}")
    print(f"Saved visualizations: {VIS_DIR}")


if __name__ == "__main__":
    main()
