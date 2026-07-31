import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.resnet34_unet import ResNet34UNet


IMAGE_SIZE = 256
BATCH_SIZE = 1
THRESHOLD = 0.55
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

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

OUTPUT_DIR = Path(
    "outputs/onnx_resnet34"
)

ONNX_PATH = (
    OUTPUT_DIR
    / "resnet34_unet_256.onnx"
)

RESULT_PATH = (
    OUTPUT_DIR
    / "resnet34_unet_onnx_benchmark.json"
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


def export_onnx(
    model: ResNet34UNet,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_cpu = model.to("cpu")
    model_cpu.eval()

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=torch.float32,
    )

    torch.onnx.export(
        model_cpu,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {
                0: "batch_size",
            },
            "logits": {
                0: "batch_size",
            },
        },
    )

    print(f"Exported ONNX: {ONNX_PATH}")


def calculate_metrics_from_probabilities(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = (
        probabilities > THRESHOLD
    ).astype(np.float32)

    predictions = predictions.reshape(
        predictions.shape[0],
        -1,
    )

    targets = targets.reshape(
        targets.shape[0],
        -1,
    )

    smooth = 1e-6

    intersection = (
        predictions * targets
    ).sum(axis=1)

    prediction_sum = predictions.sum(
        axis=1
    )

    target_sum = targets.sum(
        axis=1
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


def evaluate_pytorch(
    model: ResNet34UNet,
    loader: DataLoader,
) -> dict:
    model = model.to("cpu")
    model.eval()

    dice_sum = 0.0
    iou_sum = 0.0
    sample_count = 0

    logits_list = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].cpu()
            masks = batch["mask"].cpu()

            logits = model(images)
            probabilities = torch.sigmoid(
                logits
            )

            dice, iou = (
                calculate_metrics_from_probabilities(
                    probabilities.numpy(),
                    masks.numpy(),
                )
            )

            dice_sum += float(
                dice.sum()
            )
            iou_sum += float(
                iou.sum()
            )
            sample_count += images.size(0)

            logits_list.append(
                logits.numpy()
            )

    return {
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
        "sample_count": sample_count,
        "logits": logits_list,
    }


def evaluate_onnx(
    session: ort.InferenceSession,
    loader: DataLoader,
) -> dict:
    input_name = (
        session.get_inputs()[0].name
    )

    dice_sum = 0.0
    iou_sum = 0.0
    sample_count = 0

    logits_list = []

    for batch in loader:
        images = (
            batch["image"]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        masks = (
            batch["mask"]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        logits = session.run(
            None,
            {
                input_name: images,
            },
        )[0]

        probabilities = 1.0 / (
            1.0 + np.exp(-logits)
        )

        dice, iou = (
            calculate_metrics_from_probabilities(
                probabilities,
                masks,
            )
        )

        dice_sum += float(
            dice.sum()
        )
        iou_sum += float(
            iou.sum()
        )
        sample_count += images.shape[0]

        logits_list.append(
            logits
        )

    return {
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
        "sample_count": sample_count,
        "logits": logits_list,
    }


def compare_outputs(
    pytorch_logits: list[np.ndarray],
    onnx_logits: list[np.ndarray],
) -> dict:
    if len(pytorch_logits) != len(
        onnx_logits
    ):
        raise ValueError(
            "PyTorch and ONNX output counts differ."
        )

    max_logit_difference = 0.0
    max_probability_difference = 0.0

    for torch_output, onnx_output in zip(
        pytorch_logits,
        onnx_logits,
    ):
        logit_difference = np.max(
            np.abs(
                torch_output
                - onnx_output
            )
        )

        torch_probability = 1.0 / (
            1.0 + np.exp(
                -torch_output
            )
        )

        onnx_probability = 1.0 / (
            1.0 + np.exp(
                -onnx_output
            )
        )

        probability_difference = np.max(
            np.abs(
                torch_probability
                - onnx_probability
            )
        )

        max_logit_difference = max(
            max_logit_difference,
            float(logit_difference),
        )

        max_probability_difference = max(
            max_probability_difference,
            float(
                probability_difference
            ),
        )

    return {
        "max_logit_difference": (
            max_logit_difference
        ),
        "max_probability_difference": (
            max_probability_difference
        ),
    }


def benchmark_pytorch(
    model: ResNet34UNet,
    sample: torch.Tensor,
) -> dict:
    model = model.to("cpu")
    model.eval()

    sample = sample.cpu()

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(sample)

        start = time.perf_counter()

        for _ in range(
            BENCHMARK_RUNS
        ):
            _ = model(sample)

        elapsed = (
            time.perf_counter()
            - start
        )

    average_latency_ms = (
        elapsed
        / BENCHMARK_RUNS
        * 1000.0
    )

    return {
        "average_latency_ms": (
            average_latency_ms
        ),
        "fps": (
            1000.0
            / average_latency_ms
        ),
    }


def benchmark_onnx(
    session: ort.InferenceSession,
    sample: torch.Tensor,
) -> dict:
    input_name = (
        session.get_inputs()[0].name
    )

    sample_numpy = (
        sample.cpu()
        .numpy()
        .astype(np.float32)
    )

    for _ in range(WARMUP_RUNS):
        _ = session.run(
            None,
            {
                input_name: sample_numpy,
            },
        )

    start = time.perf_counter()

    for _ in range(
        BENCHMARK_RUNS
    ):
        _ = session.run(
            None,
            {
                input_name: sample_numpy,
            },
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    average_latency_ms = (
        elapsed
        / BENCHMARK_RUNS
        * 1000.0
    )

    return {
        "average_latency_ms": (
            average_latency_ms
        ),
        "fps": (
            1000.0
            / average_latency_ms
        ),
    }


def main() -> None:
    print(
        "ONNX Runtime providers:"
    )
    print(
        ort.get_available_providers()
    )

    test_paths = find_test_images()

    dataset = CrackDataset(
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

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model, checkpoint = load_model(
        torch.device("cpu")
    )

    export_onnx(model)

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(
        f"Threshold: {THRESHOLD:.2f}"
    )
    print(
        f"Test images: {len(dataset)}\n"
    )

    pytorch_result = evaluate_pytorch(
        model,
        loader,
    )

    onnx_result = evaluate_onnx(
        session,
        loader,
    )

    output_comparison = compare_outputs(
        pytorch_result["logits"],
        onnx_result["logits"],
    )

    sample = dataset[0]["image"].unsqueeze(
        0
    )

    pytorch_benchmark = benchmark_pytorch(
        model,
        sample,
    )

    onnx_benchmark = benchmark_onnx(
        session,
        sample,
    )

    speedup = (
        pytorch_benchmark[
            "average_latency_ms"
        ]
        / onnx_benchmark[
            "average_latency_ms"
        ]
    )

    result = {
        "model": "ResNet34UNet",
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "checkpoint_epoch": (
            checkpoint.get("epoch")
        ),
        "onnx_path": str(
            ONNX_PATH
        ),
        "opset_version": 17,
        "input_size": [
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
        "threshold": THRESHOLD,
        "test_images": len(dataset),
        "pytorch_cpu": {
            "dice": (
                pytorch_result["dice"]
            ),
            "iou": (
                pytorch_result["iou"]
            ),
            **pytorch_benchmark,
        },
        "onnx_runtime_cpu": {
            "dice": (
                onnx_result["dice"]
            ),
            "iou": (
                onnx_result["iou"]
            ),
            **onnx_benchmark,
        },
        "comparison": {
            "dice_difference": abs(
                pytorch_result["dice"]
                - onnx_result["dice"]
            ),
            "iou_difference": abs(
                pytorch_result["iou"]
                - onnx_result["iou"]
            ),
            **output_comparison,
            "speedup": speedup,
        },
        "benchmark": {
            "batch_size": BATCH_SIZE,
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": BENCHMARK_RUNS,
            "provider": (
                "CPUExecutionProvider"
            ),
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "=== Accuracy Comparison ==="
    )
    print(
        f"PyTorch Dice: "
        f"{pytorch_result['dice']:.4f}"
    )
    print(
        f"PyTorch IoU : "
        f"{pytorch_result['iou']:.4f}"
    )
    print(
        f"ONNX Dice   : "
        f"{onnx_result['dice']:.4f}"
    )
    print(
        f"ONNX IoU    : "
        f"{onnx_result['iou']:.4f}"
    )
    print(
        f"Dice diff   : "
        f"{result['comparison']['dice_difference']:.8f}"
    )
    print(
        f"IoU diff    : "
        f"{result['comparison']['iou_difference']:.8f}"
    )
    print(
        f"Max prob diff: "
        f"{result['comparison']['max_probability_difference']:.8f}"
    )

    print(
        "\n=== CPU Benchmark ==="
    )
    print(
        f"PyTorch: "
        f"{pytorch_benchmark['average_latency_ms']:.2f} ms "
        f"/ {pytorch_benchmark['fps']:.2f} FPS"
    )
    print(
        f"ONNX Runtime: "
        f"{onnx_benchmark['average_latency_ms']:.2f} ms "
        f"/ {onnx_benchmark['fps']:.2f} FPS"
    )
    print(
        f"Speedup: {speedup:.2f}x"
    )

    print(
        f"\nSaved result: "
        f"{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
