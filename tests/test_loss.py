import torch

from src.utils.losses import BCEDiceLoss


def test_bce_dice_loss_returns_finite_value():
    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    logits = torch.randn(
        2,
        1,
        64,
        64,
    )

    targets = torch.randint(
        low=0,
        high=2,
        size=(
            2,
            1,
            64,
            64,
        ),
    ).float()

    loss = criterion(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(
        loss
    ).item()


def test_bce_dice_loss_backward():
    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    logits = torch.randn(
        2,
        1,
        32,
        32,
        requires_grad=True,
    )

    targets = torch.randint(
        low=0,
        high=2,
        size=(
            2,
            1,
            32,
            32,
        ),
    ).float()

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()
