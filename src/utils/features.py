import cv2
import numpy as np


def get_area_ratio(mask):
    total_pixels = mask.shape[0] * mask.shape[1]
    crack_pixels = np.sum(mask > 0)

    return crack_pixels / total_pixels


def get_component_count(mask, min_area=5):
    mask_uint8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    count = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            count += 1

    return count


def get_total_crack_length(mask):

    return float(np.sum(mask > 0))


def extract_crack_features(mask):
    area_ratio = get_area_ratio(mask)
    component_count = get_component_count(mask, min_area=5)
    total_length_px = get_total_crack_length(mask)

    return {
        "crack_area_ratio": round(float(area_ratio), 6),
        "component_count": int(component_count),
        "total_crack_length_px_proxy": round(float(total_length_px), 2),
    }
