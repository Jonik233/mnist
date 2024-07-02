import torch
from torch import nn
from model import Net


def load_model() -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device=device)
    model.load_state_dict(torch.load("model.pth"))
    return model


def predict(model:nn.Module, sample:torch.tensor) -> int:
    model.eval()
    outputs = model(sample)
    label = torch.argmax(outputs, dim=1).item()
    return label