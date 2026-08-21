import torch

from src.models.resnet34_unet import ResNet34UNet


def test_resnet34_unet_output_shape():
    model = ResNet34UNet(
        pretrained=False
    )

    model.eval()

    x = torch.randn(
        1,
        3,
        64,
        64,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        1,
        1,
        64,
        64,
    )


def test_resnet34_unet_output_is_finite():
    model = ResNet34UNet(
        pretrained=False
    )

    model.eval()

    x = torch.randn(
        1,
        3,
        64,
        64,
    )

    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(
        output
    ).all()
