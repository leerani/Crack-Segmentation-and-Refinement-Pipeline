import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CrackDataset(
    image_dir="data/DeepCrack/train_img",
    mask_dir="data/DeepCrack/train_lab",
    image_size=256,
    use_clahe=False,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for batch in loader:

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "outputs/checkpoints/unet_raw.pth")
    print("Saved model to outputs/checkpoints/unet_raw.pth")