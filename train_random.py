from __future__ import annotations
import numpy as np
import random

import torch
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
    print(device)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
    dataset = datasets.ImageFolder(root="../../../../VMMRdb_make_model/VMMRdb_make_model", transform=transform)
    num_classes = len(dataset.classes)
    print("Number of classes:", num_classes)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    print('Fixed the last layer.')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print('loader loaded successfully')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    epoch_loss = []
    epoch_acc = []
    epoch_val_loss = []
    epoch_val_acc = []

    print('train start')
    for epoch in range(15):
        model.train()
        running_loss, running_corrects, train_sample = 0.0, 0, 0
        for train_step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            train_sample += labels.size(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss.append(running_loss / (train_step + 1))
        epoch_acc.append(running_corrects.double() / train_sample)

        model.eval()
        val_loss, val_corrects, val_sample = 0.0, 0, 0
        with torch.no_grad():
            for val_step, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_sample += labels.size(0)

        epoch_val_loss.append(val_loss / (val_step + 1))
        epoch_val_acc.append(val_corrects.double() / val_sample)
        print(f"Epoch {epoch+1}")

    torch.save({
        'epoch': 15,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'random_enet.pth')

    with open(f'random_train_output.txt', 'w', newline='', encoding='utf-8-sig') as f:
        for i in range(len(epoch_acc)):
            f.write(f'{epoch_acc[i]}, {epoch_loss[i]}, {epoch_val_acc[i]}, {epoch_val_loss[i]}\n')

if __name__ == '__main__':
    main()