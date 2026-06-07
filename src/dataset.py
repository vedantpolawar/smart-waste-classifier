from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_dataloaders():

    train_path = BASE_DIR / "data" / "split" / "train"
    val_path = BASE_DIR / "data" / "split" / "val"
    test_path = BASE_DIR / "data" / "split" / "test"

    train_dataset = datasets.ImageFolder(
        train_path,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_path,
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        test_path,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes