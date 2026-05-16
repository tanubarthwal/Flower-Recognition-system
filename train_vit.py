import torch
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

# -------------------- Config --------------------
DATA_DIR = "./flowers"
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Dataset --------------------
dataset = datasets.ImageFolder(DATA_DIR)

# 🔥 IMPORTANT FIX: custom collate to keep images as list
def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class_names = dataset.classes
num_classes = len(class_names)

# Save class names
torch.save(class_names, "class_names.pth")

# -------------------- Image Processor --------------------
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# -------------------- Model --------------------
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

model.to(device)

# -------------------- Loss & Optimizer --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# -------------------- Training --------------------
print("🚀 Training Started...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in loader:

        # 🔥 Convert images properly
        inputs = processor(images=images, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# -------------------- Save Model --------------------
torch.save(model.state_dict(), "flower_vit.pth")

print("\n✅ Training Completed!")
print("📁 Model saved as flower_vit.pth")