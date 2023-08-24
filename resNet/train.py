#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.functional as F

import torchvision
from torchvision import transforms

from model import ResNet

# Data loader
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding= 4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

trainloader = DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

testloader = DataLoader(test, batch_size=128,shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#  Model
ResNet50Config = [
    {"in_channels":64,"out_channels":256,"repetition":3,"stride":1},
    {"in_channels":256,"out_channels":512,"repetition":4,"stride":2},
    {"in_channels":512,"out_channels":1024,"repetition":6,"stride":2},
    {"in_channels":1024,"out_channels":2048,"repetition":3,"stride":2},
]
resnet = ResNet(ResNet50Config,len(classes)).to("cuda")

# Train Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

EPOCHS=200
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0
    for index, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
    
        logits = resnet(inputs)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if index%100 == 0 and index > 0:
            print(f'Loss [{epoch+1}, {index}](epoch, minibatch): ', running_loss / 100)
            running_loss = 0.0

    avg_loss = sum(losses)/len(losses)
    scheduler.step(avg_loss)
