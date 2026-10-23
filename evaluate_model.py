import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)

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

eval_tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class LeafDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        return eval_tf(image), self.labels[i]

test_loader = DataLoader(
    LeafDataset(test_p, test_y), batch_size=32,
    shuffle=False, num_workers=0
)

device = torch.device("cpu")
checkpoint = torch.load(
    OUT / "mobilenetv2_checkpoint.pth",
    map_location=device, weights_only=False
)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, len(class_names)
)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

y_true, y_pred, misclassified = [], [], []

with torch.no_grad():
    offset = 0
    for x, y in test_loader:
        outputs = model(x)
        preds = outputs.argmax(1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

        for j in range(len(y)):
            if preds[j].item() != y[j].item():
                misclassified.append(
                    (test_p[offset + j], y[j].item(), preds[j].item())
                )
        offset += len(y)

print("Test accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Macro F1:", round(f1_score(
    y_true, y_pred, average="macro", zero_division=0
), 4))
print("\nClassification report:\n")
print(classification_report(
    y_true, y_pred, labels=range(len(class_names)),
    target_names=class_names, zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Sugarcane Leaf Classification — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=160)
plt.show()

# Plot up to four real misclassified test images
if misclassified:
    examples = misclassified[:4]
    fig, axes = plt.subplots(1, len(examples), figsize=(14, 4))
    if len(examples) == 1:
        axes = [axes]

    for ax, (path, true_id, pred_id) in zip(axes, examples):
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(
            f"True: {class_names[true_id]}\n"
            f"Predicted: {class_names[pred_id]}"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "misclassified_examples.png", dpi=160)
    plt.show()
else:
    print("No misclassified examples in this test run.")