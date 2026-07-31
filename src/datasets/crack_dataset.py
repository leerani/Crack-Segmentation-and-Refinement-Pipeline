from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int = 256,
        use_clahe: bool = False,
        image_paths: Optional[Iterable[str | Path]] = None,
        lighting_mode: str = "none",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.lighting_mode = lighting_mode

        valid_modes = {
            "none",
            "mixed_balanced",
        }
        if lighting_mode not in valid_modes:
            raise ValueError(
                f"lighting_mode must be one of {valid_modes}, "
                f"got {lighting_mode}"
            )

        if image_paths is None:
            self.image_paths = sorted(
                path
                for path in self.image_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            )
        else:
            self.image_paths = sorted(
                Path(path) for path in image_paths
            )

        if not self.image_paths:
            raise ValueError(
                f"No images found in {self.image_dir}"
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
    def _make_low_light(
        image: np.ndarray,
    ) -> np.ndarray:
        """명확한 저조도 샘플을 만든다."""
        image_float = image.astype(np.float32) / 255.0

        brightness_scale = np.random.uniform(
            0.25,
            0.65,
        )
        gamma = np.random.uniform(
            1.3,
            2.2,
        )

        darkened = np.power(
            image_float,
            gamma,
        )
        darkened = darkened * brightness_scale

        noise_std = np.random.uniform(
            0.0,
            0.025,
        )
        noise = np.random.normal(
            0.0,
            noise_std,
            darkened.shape,
        ).astype(np.float32)

        darkened = darkened + noise

        return np.clip(
            darkened * 255.0,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _make_overexposed(
        image: np.ndarray,
    ) -> np.ndarray:
        """명확한 과노출 샘플을 만든다."""
        image_float = image.astype(np.float32)

        contrast_scale = np.random.uniform(
            1.1,
            1.5,
        )
        brightness_add = np.random.uniform(
            30.0,
            90.0,
        )

        overexposed = (
            image_float * contrast_scale
            + brightness_add
        )

        return np.clip(
            overexposed,
            0,
            255,
        ).astype(np.uint8)

    def _apply_balanced_lighting(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """원본 40%, 저조도 30%, 과노출 30%로 분리한다."""
        probability = np.random.rand()

        if probability < 0.40:
            return image, "original"

        if probability < 0.70:
            return self._make_low_light(image), "low_light"

        return self._make_overexposed(image), "overexposed"

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

        if self.use_clahe:
            lab = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2LAB,
            )
            lightness, channel_a, channel_b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            )
            lightness = clahe.apply(lightness)

            lab = cv2.merge(
                (
                    lightness,
                    channel_a,
                    channel_b,
                )
            )
            image = cv2.cvtColor(
                lab,
                cv2.COLOR_LAB2RGB,
            )

        applied_lighting = "none"

        if self.lighting_mode == "mixed_balanced":
            image, applied_lighting = (
                self._apply_balanced_lighting(image)
            )

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "lighting_type": applied_lighting,
        }
