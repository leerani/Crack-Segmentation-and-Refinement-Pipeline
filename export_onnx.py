import torch

from src.models.unet import UNet


device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/unet_raw.pth",
        map_location=device,
    )
)

model.eval()

dummy_input = torch.randn(1, 3, 256, 256).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "outputs/checkpoints/unet_raw.onnx",

    export_params=True,

    opset_version=11,

    do_constant_folding=True,

    input_names=["input"],
    output_names=["output"],

    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)

print("Exported ONNX model")