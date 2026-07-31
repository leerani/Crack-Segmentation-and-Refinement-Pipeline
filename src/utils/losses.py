import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Sigmoid 확률과 정답 마스크의 겹침을 기준으로 계산하는 Dice Loss."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE와 Dice Loss를 가중합한 손실함수."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")

        weight_sum = bce_weight + dice_weight
        if weight_sum <= 0:
            raise ValueError("At least one loss weight must be positive.")

        self.bce_weight = bce_weight / weight_sum
        self.dice_weight = dice_weight / weight_sum

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )
