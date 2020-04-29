"""
@file   ChessNet.py
@author Taylor Nelms
The code that actually has the neural network implemented
"""
import logging
import enum
import re
import torch
import numpy as np


class ChessNet(torch.nn.Module):
    def __init__(self):
        self.layers = torch.nn.ModuleList()

        #input dim: 7x8x8
        self.layers.append(torch.nn.Conv2d(7, 21, 3))
        #21x6x6
        self.layers.append(torch.nn.BatchNorm2d(21))
        self.layers.append(torch.nn.PReLU())
        self.layers.append(torch.nn.Conv2d(21, 64, 5))
        self.layers.append(torch.nn.BatchNorm2d(64))
        #64x2x2
        self.layers.append(torch.nn.PReLU())
        self.layers.append(torch.nn.Flatten())
        #256
        self.layers.append(torch.nn.Linear(256, 192))
        self.layers.append(torch.nn.PReLU())
        self.layers.append(torch.nn.Linear(192,132))
        self.layers.append(torch.nn.sigmoid())

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x
        

def trainModel(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in num_epochs:
        model.model()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            







