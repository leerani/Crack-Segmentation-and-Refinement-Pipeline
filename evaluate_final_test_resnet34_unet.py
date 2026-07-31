import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    / "resnet34_unet_test_per_image.csv"
)

JSON_PATH = (
    RESULT_DIR
    / "resnet34_unet_test_summary.json"
)


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

    image_paths = sorted(
        path
        for path in TEST_IMAGE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in extensions
        )
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in "
            f"{TEST_IMAGE_DIR}"
        )

    return image_paths


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

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "model_state_dict not found "
            "in checkpoint."
        )

    model = ResNet34UNet(
        pretrained=False
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def calculate_sample_metrics(
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


def evaluate(
    model: ResNet34UNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict], dict]:
    per_image_results = []

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

            dice, iou = (
                calculate_sample_metrics(
                    probabilities,
                    masks,
                )
            )

            batch_size = images.size(0)

            dice_sum += dice.sum().item()
            iou_sum += iou.sum().item()
            sample_count += batch_size

            names = batch.get("name")

            for index in range(batch_size):
                if names is None:
                    image_name = (
                        f"sample_{sample_count - batch_size + index:04d}"
                    )
                else:
                    image_name = str(
                        names[index]
                    )

                per_image_results.append(
                    {
                        "image": image_name,
                        "dice": float(
                            dice[index].item()
                        ),
                        "iou": float(
                            iou[index].item()
                        ),
                    }
                )

    if sample_count == 0:
        raise RuntimeError(
            "No test samples were evaluated."
        )

    summary = {
        "test_images": sample_count,
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
    }

    return per_image_results, summary


def save_results(
    per_image_results: list[dict],
    summary: dict,
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
                "image",
                "dice",
                "iou",
            ],
        )

        writer.writeheader()

        for result in per_image_results:
            writer.writerow(
                {
                    "image": result["image"],
                    "dice": (
                        f"{result['dice']:.6f}"
                    ),
                    "iou": (
                        f"{result['iou']:.6f}"
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
        "image_size": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "test_image_directory": str(
            TEST_IMAGE_DIR
        ),
        "test_mask_directory": str(
            TEST_MASK_DIR
        ),
        "test_images": summary[
            "test_images"
        ],
        "dice": summary["dice"],
        "iou": summary["iou"],
        "note": (
            "Threshold 0.55 was selected "
            "using the validation split "
            "before this test evaluation."
        ),
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

    test_dataset = CrackDataset(
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
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
        f"{len(test_dataset)}"
    )
    print(
        "원본 Test 이미지만 평가합니다.\n"
    )

    per_image_results, summary = evaluate(
        model,
        test_loader,
        device,
    )

    print(
        "=== Final Test Result: "
        "ResNet-34 U-Net ==="
    )
    print(
        f"Test images: "
        f"{summary['test_images']}"
    )
    print(
        f"Dice: "
        f"{summary['dice']:.4f}"
    )
    print(
        f"IoU : "
        f"{summary['iou']:.4f}"
    )

    save_results(
        per_image_results,
        summary,
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
