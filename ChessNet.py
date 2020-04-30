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

import pdb

class ChessNet(torch.nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.layers = torch.nn.ModuleList([])

        #input dim: 7x8x8
        self.layers.append(torch.nn.Conv2d(7, 32, 1))
        #21x6x6
        self.layers.append(torch.nn.BatchNorm2d(32))
        self.layers.append(torch.nn.PReLU())
        self.layers.append(torch.nn.Conv2d(32, 64, 7))
        self.layers.append(torch.nn.BatchNorm2d(64))
        #64x2x2
        self.layers.append(torch.nn.PReLU())
        self.layers.append(torch.nn.Flatten())
        #256
        self.layers.append(torch.nn.Linear(256, 192))
        self.layers.append(torch.nn.BatchNorm1d(192))
        self.layers.append(torch.nn.PReLU())
        self.layers.append(torch.nn.Linear(192,132))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x
        

def trainModel(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            if (batch_idx % 10 == 0):
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    return model 


def testModel(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.float())
            test_loss += torch.nn.functional.binary_cross_entropy(output, target.float(), reduction='sum').item()  # sum up batch loss
            outputS = output[:, :66]
            outputE = output[:, 66:]
            targetS = target[:, :66]
            targetE = target[:, 66:]
            predS = outputS.argmax(dim=1, keepdim = True)
            predE = outputE.argmax(dim=1, keepdim = True)
            correctS = predS.eq(targetS.argmax(dim=1).view_as(predS)).sum().item()
            correctE = predE.eq(targetE.argmax(dim=1).view_as(predS)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: ({},{})/{} ({:.0f}%/{:.0f}%)\n'.format(
        test_loss, correctS, correctE, len(test_loader.dataset),
        100. * correctS / len(test_loader.dataset),
        100. * correctE / len(test_loader.dataset)))


    return test_loss







