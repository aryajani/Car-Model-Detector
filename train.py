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

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
    dataset = datasets.ImageFolder(root="../../../../VMMRdb_make_model/VMMRdb_make_model", transform=transform)
    print("Number of classes:", len(dataset.classes))
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(dataset.classes))
    print('Data loaded successfully. Fixing the last layer...')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print('loader loaded successfully')

    # AMP 스케일러 설정
    scaler = torch.amp.GradScaler('cuda')

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # MAX_STEPS = 1562

    print('train start')
    for epoch in range(15):
        running_loss = 0.0
        step = 0
        for step, (images, labels) in enumerate(train_loader):
            # if step >= MAX_STEPS:
            #     break

            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Mixed Precision 적용
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

        print(f"Epoch {epoch+1}, Loss: {running_loss / step}")

    torch.save({
        'epoch': 5,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'efficientnet_checkpoint.pth')

if __name__ == '__main__':
    main()