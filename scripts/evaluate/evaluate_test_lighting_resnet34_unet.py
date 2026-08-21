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

TEST_IMAGE_DIR = Path(
    "/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/test_img"
)
TEST_MASK_DIR = Path(
    "/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/test_lab"
)

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/"
    "resnet34_unet_balanced_best.pth"
)

RESULT_DIR = Path(
    "outputs/final_test_resnet34"
)

CSV_PATH = (
    RESULT_DIR
    / "resnet34_unet_test_lighting_robustness.csv"
)

JSON_PATH = (
    RESULT_DIR
    / "resnet34_unet_test_lighting_robustness.json"
)


CONDITIONS = [
    "original",
    "low_light_50",
    "low_light_35",
    "low_light_25",
    "severe_overexposure",
]


class LightingConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        condition: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.condition = condition

        if condition not in CONDITIONS:
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
                image * 1.35
                + (90.0 / 255.0),
                0.0,
                1.0,
            )

        raise RuntimeError(
            f"Condition not handled: "
            f"{self.condition}"
        )

    def __getitem__(
        self,
        index: int,
    ):
        sample = self.base_dataset[index]

        output = dict(sample)

        output["image"] = (
            self.apply_condition(
                sample["image"]
            )
        )

        return output


def find_test_images() -> list[Path]:
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found: "
            f"{TEST_IMAGE_DIR}"
        )

    paths = sorted(
        path
        for path in TEST_IMAGE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in extensions
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"No test images found in "
            f"{TEST_IMAGE_DIR}"
        )

    return paths


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

    if sample_count == 0:
        raise RuntimeError(
            "No test samples were evaluated."
        )

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

    original_result = next(
        item
        for item in results
        if item["condition"] == "original"
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
                "dice",
                "iou",
                "dice_drop_from_original",
                "iou_drop_from_original",
                "sample_count",
                "official",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "condition": result[
                        "condition"
                    ],
                    "dice": (
                        f"{result['dice']:.6f}"
                    ),
                    "iou": (
                        f"{result['iou']:.6f}"
                    ),
                    "dice_drop_from_original": (
                        f"{result['dice_drop']:.6f}"
                    ),
                    "iou_drop_from_original": (
                        f"{result['iou_drop']:.6f}"
                    ),
                    "sample_count": result[
                        "sample_count"
                    ],
                    "official": (
                        result["condition"]
                        == "original"
                    ),
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
        "checkpoint_stage": (
            checkpoint.get("stage")
        ),
        "threshold": THRESHOLD,
        "image_size": IMAGE_SIZE,
        "official_test_result": {
            "condition": "original",
            "dice": original_result[
                "dice"
            ],
            "iou": original_result[
                "iou"
            ],
            "test_images": original_result[
                "sample_count"
            ],
        },
        "lighting_definition": {
            "original": "image",
            "low_light_50": (
                "clamp(image * 0.50)"
            ),
            "low_light_35": (
                "clamp(image * 0.35)"
            ),
            "low_light_25": (
                "clamp(image * 0.25)"
            ),
            "severe_overexposure": (
                "clamp("
                "image * 1.35 "
                "+ 90/255"
                ")"
            ),
        },
        "note": (
            "Only the original condition is the "
            "official Test result. Lighting-modified "
            "conditions are reference robustness "
            "experiments."
        ),
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

    test_paths = find_test_images()

    base_dataset = CrackDataset(
        image_dir=str(
            TEST_IMAGE_DIR
        ),
        mask_dir=str(
            TEST_MASK_DIR
        ),
        image_size=IMAGE_SIZE,
        use_clahe=False,
        image_paths=test_paths,
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
        f"Loaded checkpoint stage: "
        f"{checkpoint.get('stage', 'unknown')}"
    )

    print(
        f"Threshold fixed at: "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Test images: "
        f"{len(base_dataset)}\n"
    )

    raw_results = []

    for condition in CONDITIONS:
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

        raw_results.append(
            {
                "condition": condition,
                **metrics,
            }
        )

    original = next(
        item
        for item in raw_results
        if item["condition"] == "original"
    )

    results = []

    for result in raw_results:
        results.append(
            {
                **result,
                "dice_drop": (
                    original["dice"]
                    - result["dice"]
                ),
                "iou_drop": (
                    original["iou"]
                    - result["iou"]
                ),
            }
        )

    print(
        "=== Test Lighting Robustness: "
        "ResNet-34 U-Net ==="
    )

    print(
        f"{'Condition':>23} | "
        f"{'Dice':>8} | "
        f"{'IoU':>8} | "
        f"{'Dice Drop':>10}"
    )

    print("-" * 62)

    for result in results:
        print(
            f"{result['condition']:>23} | "
            f"{result['dice']:>8.4f} | "
            f"{result['iou']:>8.4f} | "
            f"{result['dice_drop']:>10.4f}"
        )

    print(
        "\n공식 Test 성능은 original 조건만 사용합니다."
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
