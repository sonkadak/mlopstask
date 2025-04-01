import os
import argparse
import logging
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help="Enable debug logging")
args = parser.parse_args()

# -------------------------
# Logging configuration
# -------------------------
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log_level)

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -------------------------
# Dataset
# -------------------------
train_dataset = torchvision.datasets.MNIST(
    root='/app/data',
    train=True,
    download=True,
    transform=transform
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# -------------------------
# Environment variables
# -------------------------
_cont_name = os.getenv('CONT_NAME', 'train/resnet18-mnist')
_cont_tag = os.getenv('CONT_TAG', 'latest')
logging.info(f"CONT_NAME: {_cont_name}, CONT_TAG: {_cont_tag}")

# -------------------------
# Model setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

model = torchvision.models.resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# Training
# -------------------------
num_epochs = 1
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if args.debug and i % 100 == 0:
            logging.debug(f"Epoch [{epoch+1}], Step [{i}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# -------------------------
# Save model
# -------------------------
_model = f"/app/data/model.onnx"
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(model, dummy_input, _model, input_names=["input"], output_names=["output"], opset_version=11)
logging.info(f"Model training and saving completed.: {_model}")

