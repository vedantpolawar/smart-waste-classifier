from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms

from src.model import get_model

app = FastAPI()

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

device = torch.device("cpu")

model = get_model()
model.load_state_dict(
    torch.load("models/best_model.pth", map_location=device)
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)

        pred = torch.argmax(probs)

    return {
        "class": classes[pred.item()],
        "confidence": round(
            probs[0][pred].item(),
            4
        )
    }