from pathlib import Path
import shutil
import random

random.seed(42)

source_dir = Path("data/dataset-resized")
target_dir = Path("data/split")

for class_dir in source_dir.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*"))
    random.shuffle(images)

    train_size = int(0.7 * len(images))
    val_size = int(0.15 * len(images))

    train = images[:train_size]
    val = images[train_size:train_size + val_size]
    test = images[train_size + val_size:]

    for split_name, split_images in {
        "train": train,
        "val": val,
        "test": test
    }.items():

        split_folder = target_dir / split_name / class_dir.name
        split_folder.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(img, split_folder / img.name)

print("Dataset split completed successfully!")