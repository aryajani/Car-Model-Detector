from __future__ import annotations
import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda")

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    ckpt = torch.load('./efficientnet_checkpoint.pth', map_location=device)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
    dataset = datasets.ImageFolder(root="../../../../VMMRdb_make_model/VMMRdb_make_model", transform=transform)
    print("Number of classes:", len(dataset.classes))
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1085)

    model_state_dict = ckpt['model_state_dict']
    model.load_state_dict(model_state_dict)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model.to(device)
    model.eval()

    total = 0
    correct = 0
    print("eval start")
    with torch.no_grad():
        for _, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    print(correct/total)

if __name__ == '__main__':
    main()