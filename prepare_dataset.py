from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

BASE = Path(r"E:\Sugarcane_PS04\data")
SEED = 42

class_folders = {
    "BrownRust": BASE / "Diseases" / "BrownRust",
    "Mawa": BASE / "Diseases" / "Mawa",
    "Mites": BASE / "Diseases" / "Mites",
    "RedSpot": BASE / "Diseases" / "RedSpot",
    "YellowLeaf": BASE / "Diseases" / "YellowLeaf",
    "Dried Leaves": BASE / "Dried Leaves",
    "Healthy Leaves": BASE / "Healthy Leaves",
}

paths, labels = [], []

for label, folder in class_folders.items():
    for path in folder.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(path))
            labels.append(label)

train_paths, temp_paths, train_y, temp_y = train_test_split(
    paths, labels, test_size=0.30, random_state=SEED, stratify=labels
)

val_paths, test_paths, val_y, test_y = train_test_split(
    temp_paths, temp_y, test_size=0.50,
    random_state=SEED, stratify=temp_y
)

for name, y in [
    ("Train", train_y),
    ("Validation", val_y),
    ("Test", test_y),
]:
    print(f"\n{name}: {len(y)} images")
    print(Counter(y))

print("\nTotal:", len(paths))