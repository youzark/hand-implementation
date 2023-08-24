#!/usr/bin/env python3
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from model import vannillaVariationalAutoEncoder
import matplotlib.pyplot as plt


#Configuration 
DEVICE = torch.device("cuda")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20

NUM_EPOCHS = 30
BATCH_SIZE = 256 
LR_RATE = 3e-4

dataset = datasets.MNIST(root = "dataset/", train = True, transform=transforms.ToTensor(), download=True)

test_dataset  = datasets.MNIST(root="dataset", transform=transforms.ToTensor(), train=False, download=True)

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8,shuffle=False)

model = vannillaVariationalAutoEncoder(input_dim= INPUT_DIM,
                                       h_dim= H_DIM,
                                       z_dim= Z_DIM).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.MSELoss(reduction="mean")

# Training 
def train():
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, (x, _) in loop:
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            img, mu, sigma = model(x)

            recLoss = loss_fn(img, x)
            KLLoss = - 0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            loss = recLoss + KLLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
    torch.save(model,"checkpoint.pt")

def show_image(x, idx):
    x = x.view(8, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())

model = torch.load("./checkpoint.pt")
model.to(DEVICE)
with torch.no_grad():
    for batch_id, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(8, INPUT_DIM)
        x = x.to(DEVICE)
        x_hat, _, _ = model(x)
        break
    show_image(x, idx = 0)
    show_image(x_hat, idx = 0)






