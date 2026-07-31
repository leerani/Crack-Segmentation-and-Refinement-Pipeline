import torch
import torch.nn as nn
from torchvision.models import (
    ResNet34_Weights,
    resnet34,
)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if (
            x.shape[-2:] != skip.shape[-2:]
        ):
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.conv(x)


class ResNet34UNet(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = (
            ResNet34_Weights.DEFAULT
            if pretrained
            else None
        )

        encoder = resnet34(
            weights=weights
        )

        self.input_block = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )

        self.maxpool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.center = ConvBlock(
            512,
            512,
        )

        self.decoder4 = DecoderBlock(
            512,
            256,
            256,
        )
        self.decoder3 = DecoderBlock(
            256,
            128,
            128,
        )
        self.decoder2 = DecoderBlock(
            128,
            64,
            64,
        )
        self.decoder1 = DecoderBlock(
            64,
            64,
            64,
        )

        self.final_up = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.final_conv = nn.Sequential(
            ConvBlock(
                32,
                32,
            ),
            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        skip0 = self.input_block(x)

        x = self.maxpool(skip0)
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        x = self.encoder4(skip3)

        x = self.center(x)

        x = self.decoder4(
            x,
            skip3,
        )
        x = self.decoder3(
            x,
            skip2,
        )
        x = self.decoder2(
            x,
            skip1,
        )
        x = self.decoder1(
            x,
            skip0,
        )

        x = self.final_up(x)
        x = self.final_conv(x)

        return x

    def freeze_encoder(
        self,
    ) -> None:
        encoder_modules = [
            self.input_block,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
        ]

        for module in encoder_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_encoder(
        self,
    ) -> None:
        encoder_modules = [
            self.input_block,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
        ]

        for module in encoder_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def encoder_parameters(
        self,
    ):
        modules = [
            self.input_block,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
        ]

        for module in modules:
            yield from module.parameters()

    def decoder_parameters(
        self,
    ):
        modules = [
            self.center,
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.final_up,
            self.final_conv,
        ]

        for module in modules:
            yield from module.parameters()
