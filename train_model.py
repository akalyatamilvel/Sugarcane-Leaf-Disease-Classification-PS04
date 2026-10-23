import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path(r"E:\Sugarcane_PS04\data")
OUT = Path(r"E:\Sugarcane_PS04\outputs")
OUT.mkdir(exist_ok=True)

class_folders = {
    "BrownRust": BASE / "Diseases" / "BrownRust",
    "Mawa": BASE / "Diseases" / "Mawa",
    "Mites": BASE / "Diseases" / "Mites",
    "RedSpot": BASE / "Diseases" / "RedSpot",
    "YellowLeaf": BASE / "Diseases" / "YellowLeaf",
    "Dried Leaves": BASE / "Dried Leaves",
    "Healthy Leaves": BASE / "Healthy Leaves",
}

class_names = list(class_folders.keys())
class_to_idx = {name: i for i, name in enumerate(class_names)}

paths, labels = [], []
for name, folder in class_folders.items():
    for p in folder.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(p))
            labels.append(class_to_idx[name])

train_p, temp_p, train_y, temp_y = train_test_split(
    paths, labels, test_size=0.30, random_state=SEED, stratify=labels
)
val_p, test_p, val_y, test_y = train_test_split(
    temp_p, temp_y, test_size=0.50, random_state=SEED, stratify=temp_y
)

train_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        return self.transform(image), self.labels[i]

train_loader = DataLoader(
    LeafDataset(train_p, train_y, train_tf),
    batch_size=16, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    LeafDataset(val_p, val_y, eval_tf),
    batch_size=32, shuffle=False, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Use pretrained weights; first download requires internet access.
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

for p in model.features.parameters():
    p.requires_grad = False

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, len(class_names)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(), lr=0.001
)

history = {"train_loss": [], "val_loss": [],
           "train_acc": [], "val_acc": []}

def evaluate(loader):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss_sum += criterion(outputs, y).item() * len(y)
            correct += (outputs.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total

epochs = 3  # Quick initial run; increase later if time permits.

for epoch in range(epochs):
    model.train()
    loss_sum = correct = total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * len(y)
        correct += (outputs.argmax(1) == y).sum().item()
        total += len(y)

    train_loss = loss_sum / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.4f}"
    )

torch.save({
    "model_state": model.state_dict(),
    "class_names": class_names,
    "history": history,
    "image_size": 160,
    "seed": SEED
}, OUT / "mobilenetv2_checkpoint.pth")

print("Saved checkpoint to", OUT / "mobilenetv2_checkpoint.pth")