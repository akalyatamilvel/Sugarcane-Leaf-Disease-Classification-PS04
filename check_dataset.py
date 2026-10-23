import os

base = r"E:\Sugarcane_PS04\data"

for root, dirs, files in os.walk(base):
    images = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if images:
        print(f"{root}: {len(images)} images")