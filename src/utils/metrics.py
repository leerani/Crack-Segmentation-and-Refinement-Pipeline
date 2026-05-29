import numpy as np


def dice_score(pred, target, smooth=1e-6):

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    intersection = np.sum(pred * target)

    return (2.0 * intersection + smooth) / (
        np.sum(pred) + np.sum(target) + smooth
    )


def iou_score(pred, target, smooth=1e-6):

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    intersection = np.sum(pred * target)

    union = np.sum(pred) + np.sum(target) - intersection

    return (intersection + smooth) / (union + smooth)