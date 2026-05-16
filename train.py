import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
import os

# Config
data_dir = "flowers"
batch_size = 32
num_epochs = 5
num_classes = 5
model_save_path = "flower_model.pth"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
dataset = datasets.ImageFolder(r"C:\project\flower recognition project\flowers", transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Class names
class_names = dataset.classes
torch.save(class_names, "class_names.pth")

# Model
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
model.train()
print("Starting training...")
for epoch in range(num_epochs):
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.4f}")
print("Training complete!")

# Save model
torch.save(model.state_dict(), model_save_path)
print("✅ Model saved to", model_save_path)
print("✅ Training script completed successfully.")
