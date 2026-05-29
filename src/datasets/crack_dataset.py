from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256, use_clahe=False,):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.use_clahe = use_clahe

        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpeg"))
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        stem = image_path.stem # 확장자 뺀 파일 이름

        mask_candidates = [
            self.mask_dir / f"{stem}.png",
            self.mask_dir / f"{stem}.jpg",
            self.mask_dir / f"{stem}.jpeg",
        ]

        mask_path = None
        for candidate in mask_candidates:
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {image_path.name}")

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_clahe:

            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            )

            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))

            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0

        # mask를 0 또는 1로 변환
        mask = (mask > 127).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }