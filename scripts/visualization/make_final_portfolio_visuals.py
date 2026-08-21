import csv
import json
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34UNet


IMAGE_SIZE = 256

BASELINE_THRESHOLD = 0.65
FINAL_THRESHOLD = 0.55

# 너무 작은 균열 마스크가 개선 1위로 선택되는 것을 방지
MIN_GROUND_TRUTH_PIXELS = 150

VAL_SPLIT_PATH = Path(
    "outputs/checkpoints/train_val_split.json"
)

MASK_DIR = Path(
    "data/DeepCrack/train_lab"
)

BASELINE_CHECKPOINT = Path(
    "outputs/checkpoints/"
    "unet_bce_dice_balanced_lighting_best.pth"
)

FINAL_CHECKPOINT = Path(
    "outputs/checkpoints/"
    "resnet34_unet_balanced_best.pth"
)

OUTPUT_DIR = Path(
    "outputs/portfolio_visuals_final"
)

RANKING_CSV_PATH = (
    OUTPUT_DIR
    / "model_improvement_ranking.csv"
)

# 조명/대표 사례는 기존 선택 유지
SELECTED_STEMS = {
    "lighting": "11113",
    "best": "11190-3",
    "typical": "IMG33-2",
}


def load_basic_unet(
    checkpoint_path: Path,
    device: torch.device,
) -> UNet:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = UNet().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_resnet34_unet(
    checkpoint_path: Path,
    device: torch.device,
) -> ResNet34UNet:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = ResNet34UNet(
        pretrained=False
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_validation_paths() -> list[Path]:
    with VAL_SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    return [
        Path(path)
        for path in split_data["val_images"]
    ]


def find_selected_image(
    val_paths: list[Path],
    stem: str,
) -> Path:
    for path in val_paths:
        if path.stem == stem:
            return path

    raise FileNotFoundError(
        f"Validation image not found: {stem}"
    )


def find_mask_path(
    image_path: Path,
) -> Path:
    for extension in (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    ):
        candidate = (
            MASK_DIR
            / f"{image_path.stem}{extension}"
        )

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Mask not found: {image_path.name}"
    )


def read_image_and_mask(
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    mask = cv2.imread(
        str(find_mask_path(image_path)),
        cv2.IMREAD_GRAYSCALE,
    )

    if image_bgr is None:
        raise ValueError(
            f"Failed to read image: {image_path}"
        )

    if mask is None:
        raise ValueError(
            f"Failed to read mask: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image_rgb = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    mask = (
        mask > 127
    ).astype(np.float32)

    return image_rgb, mask


def image_to_tensor(
    image_rgb: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    array = (
        image_rgb.astype(np.float32)
        / 255.0
    )

    array = np.transpose(
        array,
        (2, 0, 1),
    )

    return (
        torch.from_numpy(array)
        .float()
        .unsqueeze(0)
        .to(device)
    )


def predict(
    model: nn.Module,
    image_rgb: np.ndarray,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    tensor = image_to_tensor(
        image_rgb,
        device,
    )

    with torch.no_grad():
        logits = model(tensor)

        probabilities = torch.sigmoid(
            logits
        )

        prediction = (
            probabilities > threshold
        ).float()

    return (
        prediction[0, 0]
        .cpu()
        .numpy()
    )


def calculate_dice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    prediction = (
        prediction > 0.5
    ).astype(np.float32)

    ground_truth = (
        ground_truth > 0.5
    ).astype(np.float32)

    intersection = float(
        (prediction * ground_truth).sum()
    )

    denominator = float(
        prediction.sum()
        + ground_truth.sum()
    )

    return (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )


def find_strong_improvement_sample(
    val_paths: list[Path],
    baseline_model: UNet,
    final_model: ResNet34UNet,
    device: torch.device,
) -> tuple[Path, list[dict]]:
    ranking = []

    for index, image_path in enumerate(
        val_paths,
        start=1,
    ):
        image_rgb, ground_truth = (
            read_image_and_mask(
                image_path
            )
        )

        gt_pixels = int(
            ground_truth.sum()
        )

        baseline_prediction = predict(
            baseline_model,
            image_rgb,
            BASELINE_THRESHOLD,
            device,
        )

        final_prediction = predict(
            final_model,
            image_rgb,
            FINAL_THRESHOLD,
            device,
        )

        baseline_dice = calculate_dice(
            baseline_prediction,
            ground_truth,
        )

        final_dice = calculate_dice(
            final_prediction,
            ground_truth,
        )

        improvement = (
            final_dice
            - baseline_dice
        )

        eligible = (
            gt_pixels
            >= MIN_GROUND_TRUTH_PIXELS
            and final_dice > baseline_dice
        )

        ranking.append(
            {
                "image_path": str(
                    image_path
                ),
                "stem": image_path.stem,
                "ground_truth_pixels": (
                    gt_pixels
                ),
                "baseline_dice": (
                    baseline_dice
                ),
                "final_dice": (
                    final_dice
                ),
                "improvement": (
                    improvement
                ),
                "eligible": eligible,
            }
        )

        print(
            f"[{index:02d}/{len(val_paths)}] "
            f"{image_path.stem} | "
            f"Basic {baseline_dice:.4f} | "
            f"ResNet34 {final_dice:.4f} | "
            f"Change {improvement:+.4f}"
        )

    ranking.sort(
        key=lambda item: (
            item["eligible"],
            item["improvement"],
            item["final_dice"],
        ),
        reverse=True,
    )

    eligible_items = [
        item
        for item in ranking
        if item["eligible"]
    ]

    if not eligible_items:
        raise RuntimeError(
            "No eligible improvement sample found."
        )

    selected = eligible_items[0]

    return (
        Path(selected["image_path"]),
        ranking,
    )


def save_ranking_csv(
    ranking: list[dict],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RANKING_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "stem",
                "ground_truth_pixels",
                "baseline_dice",
                "final_dice",
                "improvement",
                "eligible",
                "image_path",
            ],
        )

        writer.writeheader()

        for rank, item in enumerate(
            ranking,
            start=1,
        ):
            writer.writerow(
                {
                    "rank": rank,
                    "stem": item["stem"],
                    "ground_truth_pixels": (
                        item[
                            "ground_truth_pixels"
                        ]
                    ),
                    "baseline_dice": (
                        f"{item['baseline_dice']:.6f}"
                    ),
                    "final_dice": (
                        f"{item['final_dice']:.6f}"
                    ),
                    "improvement": (
                        f"{item['improvement']:.6f}"
                    ),
                    "eligible": (
                        item["eligible"]
                    ),
                    "image_path": (
                        item["image_path"]
                    ),
                }
            )


def apply_low_light(
    image_rgb: np.ndarray,
    ratio: float,
) -> np.ndarray:
    transformed = (
        image_rgb.astype(np.float32)
        * ratio
    )

    return np.clip(
        transformed,
        0,
        255,
    ).astype(np.uint8)


def apply_overexposure(
    image_rgb: np.ndarray,
) -> np.ndarray:
    transformed = (
        image_rgb.astype(np.float32)
        * 1.35
        + 90.0
    )

    return np.clip(
        transformed,
        0,
        255,
    ).astype(np.uint8)


def style_axes(
    axes: Union[
        np.ndarray,
        list,
    ],
) -> None:
    for axis in np.array(
        axes
    ).reshape(-1):
        axis.axis("off")


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    figure.tight_layout(
        pad=0.8,
        w_pad=0.8,
        h_pad=1.0,
    )

    figure.savefig(
        OUTPUT_DIR / filename,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


def save_improvement_visual(
    image_path: Path,
    baseline_model: UNet,
    final_model: ResNet34UNet,
    device: torch.device,
) -> None:
    image_rgb, ground_truth = (
        read_image_and_mask(
            image_path
        )
    )

    baseline_prediction = predict(
        baseline_model,
        image_rgb,
        BASELINE_THRESHOLD,
        device,
    )

    final_prediction = predict(
        final_model,
        image_rgb,
        FINAL_THRESHOLD,
        device,
    )

    baseline_dice = calculate_dice(
        baseline_prediction,
        ground_truth,
    )

    final_dice = calculate_dice(
        final_prediction,
        ground_truth,
    )

    improvement = (
        final_dice
        - baseline_dice
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    panels = [
        (
            image_rgb,
            "Input Image",
            None,
        ),
        (
            ground_truth,
            "Ground Truth",
            "gray",
        ),
        (
            baseline_prediction,
            (
                "Basic U-Net\n"
                f"Dice {baseline_dice:.3f}"
            ),
            "gray",
        ),
        (
            final_prediction,
            (
                "ResNet-34 U-Net\n"
                f"Dice {final_dice:.3f} "
                f"(+{improvement:.3f})"
            ),
            "gray",
        ),
    ]

    for axis, (
        image,
        title,
        cmap,
    ) in zip(
        axes,
        panels,
    ):
        axis.imshow(
            image,
            cmap=cmap,
            vmin=(
                0
                if cmap
                else None
            ),
            vmax=(
                1
                if cmap
                else None
            ),
        )

        axis.set_title(
            title,
            fontsize=14,
        )

    style_axes(axes)

    save_figure(
        figure,
        "model_improvement_comparison.png",
    )


def save_lighting_visual(
    image_path: Path,
    final_model: ResNet34UNet,
    device: torch.device,
) -> None:
    image_rgb, ground_truth = (
        read_image_and_mask(
            image_path
        )
    )

    conditions = [
        (
            "Original",
            image_rgb,
        ),
        (
            "Low-light 50%",
            apply_low_light(
                image_rgb,
                0.50,
            ),
        ),
        (
            "Low-light 25%",
            apply_low_light(
                image_rgb,
                0.25,
            ),
        ),
        (
            "Severe Overexposure",
            apply_overexposure(
                image_rgb
            ),
        ),
    ]

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    for column, (
        condition_title,
        condition_image,
    ) in enumerate(conditions):
        prediction = predict(
            final_model,
            condition_image,
            FINAL_THRESHOLD,
            device,
        )

        dice = calculate_dice(
            prediction,
            ground_truth,
        )

        axes[0, column].imshow(
            condition_image
        )

        axes[0, column].set_title(
            condition_title,
            fontsize=14,
        )

        axes[1, column].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[1, column].set_title(
            f"Prediction · Dice {dice:.3f}",
            fontsize=13,
        )

    style_axes(axes)

    save_figure(
        figure,
        "lighting_robustness.png",
    )


def save_case_visual(
    image_path: Path,
    final_model: ResNet34UNet,
    device: torch.device,
    filename: str,
) -> None:
    image_rgb, ground_truth = (
        read_image_and_mask(
            image_path
        )
    )

    final_prediction = predict(
        final_model,
        image_rgb,
        FINAL_THRESHOLD,
        device,
    )

    dice = calculate_dice(
        final_prediction,
        ground_truth,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    panels = [
        (
            image_rgb,
            "Input Image",
            None,
        ),
        (
            ground_truth,
            "Ground Truth",
            "gray",
        ),
        (
            final_prediction,
            (
                "ResNet-34 U-Net Prediction\n"
                f"Dice {dice:.3f}"
            ),
            "gray",
        ),
    ]

    for axis, (
        image,
        title,
        cmap,
    ) in zip(
        axes,
        panels,
    ):
        axis.imshow(
            image,
            cmap=cmap,
            vmin=(
                0
                if cmap
                else None
            ),
            vmax=(
                1
                if cmap
                else None
            ),
        )

        axis.set_title(
            title,
            fontsize=14,
        )

    style_axes(axes)

    save_figure(
        figure,
        filename,
    )


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_paths = (
        load_validation_paths()
    )

    baseline_model = load_basic_unet(
        BASELINE_CHECKPOINT,
        device,
    )

    final_model = load_resnet34_unet(
        FINAL_CHECKPOINT,
        device,
    )

    print(
        "\nSearching all Validation images "
        "for a strong improvement sample...\n"
    )

    improvement_path, ranking = (
        find_strong_improvement_sample(
            val_paths,
            baseline_model,
            final_model,
            device,
        )
    )

    save_ranking_csv(
        ranking
    )

    selected = ranking[0]

    print(
        "\nSelected improvement sample:"
    )
    print(
        f"Stem: {selected['stem']}"
    )
    print(
        f"Basic U-Net Dice: "
        f"{selected['baseline_dice']:.4f}"
    )
    print(
        f"ResNet-34 U-Net Dice: "
        f"{selected['final_dice']:.4f}"
    )
    print(
        f"Improvement: "
        f"{selected['improvement']:+.4f}"
    )

    selected_paths = {
        key: find_selected_image(
            val_paths,
            stem,
        )
        for key, stem
        in SELECTED_STEMS.items()
    }

    save_improvement_visual(
        improvement_path,
        baseline_model,
        final_model,
        device,
    )

    save_lighting_visual(
        selected_paths["lighting"],
        final_model,
        device,
    )

    save_case_visual(
        selected_paths["best"],
        final_model,
        device,
        filename="best_case.png",
    )

    save_case_visual(
        selected_paths["typical"],
        final_model,
        device,
        filename="typical_case.png",
    )

    print(
        "\nUpdated visuals saved:"
    )

    for filename in (
        "model_improvement_comparison.png",
        "lighting_robustness.png",
        "best_case.png",
        "typical_case.png",
        "model_improvement_ranking.csv",
    ):
        print(
            OUTPUT_DIR / filename
        )


if __name__ == "__main__":
    main()
