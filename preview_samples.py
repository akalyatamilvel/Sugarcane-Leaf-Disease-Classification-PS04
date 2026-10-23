import os
import random
import matplotlib.pyplot as plt
from PIL import Image

base = r"E:\Sugarcane_PS04\data"
random.seed(42)

classes = [
    ("BrownRust", os.path.join(base, "Diseases", "BrownRust")),
    ("Mawa", os.path.join(base, "Diseases", "Mawa")),
    ("Mites", os.path.join(base, "Diseases", "Mites")),
    ("RedSpot", os.path.join(base, "Diseases", "RedSpot")),
    ("YellowLeaf", os.path.join(base, "Diseases", "YellowLeaf")),
    ("Dried Leaves", os.path.join(base, "Dried Leaves")),
    ("Healthy Leaves", os.path.join(base, "Healthy Leaves")),
]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i, (name, folder) in enumerate(classes):
    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_path = os.path.join(folder, random.choice(images))

    with Image.open(image_path) as img:
        axes[i].imshow(img.convert("RGB"))

    axes[i].set_title(name)
    axes[i].axis("off")

axes[7].axis("off")
plt.suptitle("Sugarcane Leaf Dataset — Sample Images")
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/sample_images.png", dpi=150, bbox_inches="tight")
plt.show()