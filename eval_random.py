from __future__ import annotations
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from torchvision import datasets, transforms
import torchvision.models as models

from sklearn.metrics import f1_score


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda")

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    ckpt = torch.load('./random_enet.pth', map_location=device)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
    dataset = datasets.ImageFolder(root="../../../../VMMRdb_make_model/VMMRdb_make_model", transform=transform)
    num_classes = len(dataset.classes)
    print("Number of classes:", len(dataset.classes))
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    print('Fixed the last layer.')


    model.load_state_dict(ckpt['model_state_dict'])
    print('model loaded successfully.')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print('loader loaded successfully')

    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    print("eval start")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, dim=1)
    accuracies = predictions.eq(labels)
    accuracy = accuracies.float().mean().item()
    print(f"Total Accuracy: {accuracy * 100:.2f}%")
    print(f'{f1_score(labels, predictions, average='macro')}')

if __name__ == '__main__':
    main()