import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, sampler

df = pd.read_csv("data/train.csv")

labels = torch.tensor(df.iloc[:, 0].values)
pixels = torch.tensor(df.iloc[:, 1:].values)

total_num = labels.shape[0]
train_num = int(0.8 * total_num)

#Setting up generator
random_seed = 42
generator = torch.Generator().manual_seed(random_seed)

#Samplers for data-loaders
train_sampler = sampler.SubsetRandomSampler(range(train_num), generator=generator)
val_sampler = sampler.SubsetRandomSampler(range(train_num, total_num), generator=generator)

#Setting up data-loaders
dataset = TensorDataset(pixels, labels)
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler, drop_last=True)
loaders = {"train":train_loader, "val":val_loader}