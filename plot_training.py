import torch
import matplotlib.pyplot as plt
from pathlib import Path

out = Path(r"E:\Sugarcane_PS04\outputs")
checkpoint = torch.load(
    out / "mobilenetv2_checkpoint.pth",
    map_location="cpu",
    weights_only=False
)
history = checkpoint["history"]
epochs = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(epochs, history["train_acc"], marker="o", label="Training")
axes[0].plot(epochs, history["val_acc"], marker="o", label="Validation")
axes[0].set_title("Training and Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history["train_loss"], marker="o", label="Training")
axes[1].plot(epochs, history["val_loss"], marker="o", label="Validation")
axes[1].set_title("Training and Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out / "training_curves.png", dpi=160)
plt.show()