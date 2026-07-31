import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.unet import UNet


IMAGE_SIZE = 256
BATCH_SIZE = 1
THRESHOLD = 0.65
WARMUP_RUNS = 20
TIMED_RUNS = 100

TEST_IMAGE_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/test_img")
TEST_MASK_DIR = Path("/media/rani/새 볼륨/crack-seg-project/data/DeepCrack/test_lab")

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/"
    "unet_bce_dice_balanced_lighting_best.pth"
)

OUTPUT_DIR = Path("outputs/onnx")
ONNX_PATH = OUTPUT_DIR / "unet_balanced_final.onnx"
CSV_PATH = OUTPUT_DIR / "onnx_evaluation.csv"
JSON_PATH = OUTPUT_DIR / "onnx_evaluation.json"


class OriginalTestDataset(Dataset):
    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int = 256,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        if not self.image_paths:
            raise ValueError(
                f"No images found in {image_dir}"
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
            f"Mask not found for {image_path.name}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]

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

        image_rgb = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = (
            image_rgb.astype(np.float32) / 255.0
        )
        image = np.transpose(
            image,
            (2, 0, 1),
        )

        mask = (
            mask > 127
        ).astype(np.float32)

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "image_path": str(image_path),
        }


def load_pytorch_model() -> tuple[UNet, dict]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        raise KeyError(
            "model_state_dict not found in checkpoint."
        )

    model = UNet()
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def export_to_onnx(
    model: UNet,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def check_onnx_model() -> None:
    try:
        import onnx
    except ImportError as error:
        raise ImportError(
            "onnx가 설치되어 있지 않습니다. "
            "pip install onnx onnxruntime 명령으로 설치하세요."
        ) from error

    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)


def create_onnx_session():
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError(
            "onnxruntime이 설치되어 있지 않습니다. "
            "pip install onnxruntime 명령으로 설치하세요."
        ) from error

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    return session


def sigmoid_numpy(
    logits: np.ndarray,
) -> np.ndarray:
    return 1.0 / (
        1.0 + np.exp(-logits)
    )


def calculate_metrics(
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

    prediction_sum = predictions.sum(axis=1)
    target_sum = targets.sum(axis=1)

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


def evaluate_pytorch(
    model: UNet,
    loader: DataLoader,
) -> dict:
    dice_sum = 0.0
    iou_sum = 0.0
    sample_count = 0
    probabilities_by_image = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            masks = batch["mask"].numpy()

            logits = model(images)
            probabilities = torch.sigmoid(
                logits
            ).numpy()

            dice, iou = calculate_metrics(
                probabilities,
                masks,
            )

            dice_sum += float(dice.sum())
            iou_sum += float(iou.sum())
            sample_count += images.shape[0]

            for index, image_path in enumerate(
                batch["image_path"]
            ):
                probabilities_by_image[image_path] = (
                    probabilities[index]
                )

    return {
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
        "sample_count": sample_count,
        "probabilities_by_image": probabilities_by_image,
    }


def evaluate_onnx(
    session,
    loader: DataLoader,
    pytorch_probabilities: dict,
) -> dict:
    input_name = session.get_inputs()[0].name

    dice_sum = 0.0
    iou_sum = 0.0
    sample_count = 0

    max_abs_difference = 0.0
    mean_abs_difference_sum = 0.0

    for batch in loader:
        images = batch["image"].numpy()
        masks = batch["mask"].numpy()

        logits = session.run(
            None,
            {input_name: images},
        )[0]

        probabilities = sigmoid_numpy(logits)

        dice, iou = calculate_metrics(
            probabilities,
            masks,
        )

        dice_sum += float(dice.sum())
        iou_sum += float(iou.sum())
        sample_count += images.shape[0]

        for index, image_path in enumerate(
            batch["image_path"]
        ):
            pytorch_probability = (
                pytorch_probabilities[image_path]
            )

            difference = np.abs(
                probabilities[index]
                - pytorch_probability
            )

            max_abs_difference = max(
                max_abs_difference,
                float(difference.max()),
            )

            mean_abs_difference_sum += float(
                difference.mean()
            )

    return {
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
        "sample_count": sample_count,
        "max_abs_probability_difference": (
            max_abs_difference
        ),
        "mean_abs_probability_difference": (
            mean_abs_difference_sum / sample_count
        ),
    }


def benchmark_pytorch(
    model: UNet,
    sample: torch.Tensor,
) -> dict:
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(sample)

        times = []

        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            _ = model(sample)
            end = time.perf_counter()

            times.append(
                (end - start) * 1000.0
            )

    return summarize_latency(times)


def benchmark_onnx(
    session,
    sample: np.ndarray,
) -> dict:
    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        _ = session.run(
            None,
            {input_name: sample},
        )

    times = []

    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        _ = session.run(
            None,
            {input_name: sample},
        )
        end = time.perf_counter()

        times.append(
            (end - start) * 1000.0
        )

    return summarize_latency(times)


def summarize_latency(
    times: list[float],
) -> dict:
    times_array = np.array(
        times,
        dtype=np.float64,
    )

    mean_ms = float(times_array.mean())

    return {
        "mean_ms": mean_ms,
        "median_ms": float(
            np.median(times_array)
        ),
        "p95_ms": float(
            np.percentile(times_array, 95)
        ),
        "fps": (
            1000.0 / mean_ms
            if mean_ms > 0
            else 0.0
        ),
        "runs": len(times),
    }


def save_results(
    checkpoint: dict,
    pytorch_result: dict,
    onnx_result: dict,
    pytorch_latency: dict,
    onnx_latency: dict,
    providers: list[str],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        {
            "runtime": "PyTorch CPU",
            "dice": pytorch_result["dice"],
            "iou": pytorch_result["iou"],
            "mean_latency_ms": pytorch_latency["mean_ms"],
            "median_latency_ms": pytorch_latency["median_ms"],
            "p95_latency_ms": pytorch_latency["p95_ms"],
            "fps": pytorch_latency["fps"],
        },
        {
            "runtime": "ONNX Runtime CPU",
            "dice": onnx_result["dice"],
            "iou": onnx_result["iou"],
            "mean_latency_ms": onnx_latency["mean_ms"],
            "median_latency_ms": onnx_latency["median_ms"],
            "p95_latency_ms": onnx_latency["p95_ms"],
            "fps": onnx_latency["fps"],
        },
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "runtime": row["runtime"],
                    "dice": f"{row['dice']:.6f}",
                    "iou": f"{row['iou']:.6f}",
                    "mean_latency_ms": (
                        f"{row['mean_latency_ms']:.4f}"
                    ),
                    "median_latency_ms": (
                        f"{row['median_latency_ms']:.4f}"
                    ),
                    "p95_latency_ms": (
                        f"{row['p95_latency_ms']:.4f}"
                    ),
                    "fps": f"{row['fps']:.4f}",
                }
            )

    output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "onnx_path": str(ONNX_PATH),
        "onnx_opset": 17,
        "threshold": THRESHOLD,
        "image_size": IMAGE_SIZE,
        "test_images": pytorch_result["sample_count"],
        "providers": providers,
        "accuracy": {
            "pytorch_cpu": {
                "dice": pytorch_result["dice"],
                "iou": pytorch_result["iou"],
            },
            "onnx_runtime_cpu": {
                "dice": onnx_result["dice"],
                "iou": onnx_result["iou"],
            },
            "difference": {
                "dice": (
                    onnx_result["dice"]
                    - pytorch_result["dice"]
                ),
                "iou": (
                    onnx_result["iou"]
                    - pytorch_result["iou"]
                ),
                "max_abs_probability_difference": (
                    onnx_result[
                        "max_abs_probability_difference"
                    ]
                ),
                "mean_abs_probability_difference": (
                    onnx_result[
                        "mean_abs_probability_difference"
                    ]
                ),
            },
        },
        "latency_batch_1_cpu": {
            "pytorch": pytorch_latency,
            "onnx_runtime": onnx_latency,
            "speedup": (
                pytorch_latency["mean_ms"]
                / onnx_latency["mean_ms"]
                if onnx_latency["mean_ms"] > 0
                else None
            ),
        },
        "note": (
            "ONNX 변환 후 성능 보존과 CPU 추론 속도를 "
            "확인한 배포 형식 평가입니다. "
            "모델과 threshold는 변경하지 않았습니다."
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
    print("=== Final Model ONNX Evaluation ===")
    print(f"Threshold: {THRESHOLD:.2f}")
    print("Runtime comparison: CPU, batch size 1")
    print(
        "모델과 threshold는 변경하지 않습니다.\n"
    )

    model, checkpoint = load_pytorch_model()

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    export_to_onnx(model)
    check_onnx_model()

    print(f"ONNX export completed: {ONNX_PATH}")
    print("ONNX model validation passed.")

    session = create_onnx_session()

    dataset = OriginalTestDataset(
        TEST_IMAGE_DIR,
        TEST_MASK_DIR,
        IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Test images: {len(dataset)}\n")

    pytorch_result = evaluate_pytorch(
        model,
        loader,
    )

    onnx_result = evaluate_onnx(
        session,
        loader,
        pytorch_result["probabilities_by_image"],
    )

    first_sample = dataset[0]["image"].unsqueeze(0)

    pytorch_latency = benchmark_pytorch(
        model,
        first_sample,
    )

    onnx_latency = benchmark_onnx(
        session,
        first_sample.numpy(),
    )

    print("=== Accuracy Preservation ===")
    print(
        f"PyTorch CPU     | "
        f"Dice: {pytorch_result['dice']:.4f} | "
        f"IoU: {pytorch_result['iou']:.4f}"
    )
    print(
        f"ONNX Runtime CPU| "
        f"Dice: {onnx_result['dice']:.4f} | "
        f"IoU: {onnx_result['iou']:.4f}"
    )
    print(
        f"Dice difference | "
        f"{onnx_result['dice'] - pytorch_result['dice']:+.6f}"
    )
    print(
        f"IoU difference  | "
        f"{onnx_result['iou'] - pytorch_result['iou']:+.6f}"
    )
    print(
        f"Max probability difference  | "
        f"{onnx_result['max_abs_probability_difference']:.8f}"
    )
    print(
        f"Mean probability difference | "
        f"{onnx_result['mean_abs_probability_difference']:.8f}"
    )

    print("\n=== CPU Latency: Batch 1 ===")
    print(
        f"PyTorch CPU      | "
        f"{pytorch_latency['mean_ms']:.2f} ms | "
        f"{pytorch_latency['fps']:.2f} FPS"
    )
    print(
        f"ONNX Runtime CPU | "
        f"{onnx_latency['mean_ms']:.2f} ms | "
        f"{onnx_latency['fps']:.2f} FPS"
    )

    speedup = (
        pytorch_latency["mean_ms"]
        / onnx_latency["mean_ms"]
    )

    print(
        f"ONNX speedup     | "
        f"{speedup:.2f}x"
    )

    save_results(
        checkpoint,
        pytorch_result,
        onnx_result,
        pytorch_latency,
        onnx_latency,
        session.get_providers(),
    )

    print(f"\nSaved ONNX: {ONNX_PATH}")
    print(f"Saved CSV : {CSV_PATH}")
    print(f"Saved JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
