import cv2
import numpy as np


def remove_small_components(mask, min_area=30):
    mask_uint8 = (mask * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask_uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 255

    return (cleaned > 0).astype(np.float32)


def refine_crack_mask(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 끊긴 crack 연결
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 작은 점 노이즈 제거
    cleaned = remove_small_components((closed > 0).astype(np.float32), min_area=5)

    return cleaned