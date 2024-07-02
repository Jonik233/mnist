import torch
import pandas as pd
from torch import nn
from model import Net
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, sampler
from train_val import train_model, validate_model

df = pd.read_csv("data/train.csv")

labels = torch.tensor(df.iloc[:, 0].values).reshape(-1, 1)
pixels = torch.tensor(df.iloc[:, 1:].values).reshape(-1, 28, 28)

total_num = labels.shape[0]
train_num = int(0.8 * total_num)

epochs = 10
batch_size = 32
learning_rate = 1e-3

random_seed = 42
torch.manual_seed(random_seed)
generator = torch.Generator().manual_seed(random_seed)

train_sampler = sampler.SubsetRandomSampler(range(train_num), generator=generator)
val_sampler = sampler.SubsetRandomSampler(range(train_num, total_num), generator=generator)

dataset = TensorDataset(pixels, labels)
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True)
loaders = {"train":train_loader, "val":val_loader}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Net().to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

#Training loop
for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}")
    print("-"*40)
    train_model(model, loss_fn, optimizer, train_loader, device)
    validate_model(model, loss_fn, val_loader, device)


#Saving model's weights
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")