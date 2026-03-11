from torchvision import transforms
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(model, image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        pred = outputs.argmax(1).item()

    return pred