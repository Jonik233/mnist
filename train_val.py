import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def train_model(model:nn.Module, loss_fn:nn.Module, optimizer:optim.Optimizer, loader:DataLoader, device:str) -> None:
    model.train()
    num_batches = len(loader)
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 200 == 0:
            loss = loss.item()
            print(f"Loss: {loss:.4f} [{i+1}/{num_batches}]")


def validate_model(model:nn.Module, loss_fn:nn.Module, loader:DataLoader, device:str) -> None:
    model.eval()
    correct = 0
    losses = []
    size = len(loader) * loader.batch_size

    for inputs, labels in loader:
        inputs = inputs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        loss = loss_fn(outputs, labels).item()

        losses.append(loss)
        correct += (preds == labels).sum().item()
    
    accuracy = (correct / size) * 100
    avg_loss = sum(losses) / len(losses)
    print(f"Val Error: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")