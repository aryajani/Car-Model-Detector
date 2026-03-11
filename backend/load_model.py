import torch
from torchvision import models, datasets
import torch.nn as nn

NUM_CLASSES = 1085

def load_model():

    model = models.efficientnet_b0()

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load("efficientnet_checkpoint.pth", map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    dataset = datasets.ImageFolder("../VMMRdb_make_model")
    classes = dataset.classes

    return model, classes