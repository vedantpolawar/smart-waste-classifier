import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader, classes = get_dataloaders()

model = get_model()
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(
    all_labels,
    all_preds,
    target_names=classes
))

print(confusion_matrix(
    all_labels,
    all_preds
))