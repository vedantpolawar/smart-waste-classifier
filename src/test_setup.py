from dataset import get_dataloaders

train_loader, val_loader, test_loader, classes = get_dataloaders()

print("Classes:", classes)
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))