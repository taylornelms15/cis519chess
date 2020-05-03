"""
@file   ChessNet.py
@author Taylor Nelms
The code that actually has the neural network implemented

Note: chunks of code blatantly stolen from the following article:
https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
(guide to implementing residual neural network in pytorch)
"""
import logging
import enum
import re
import torch
import torch.nn as nn
import numpy as np
import functools

import pdb

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = functools.partial(Conv2dAuto, kernel_size=3, bias=False)
conv = conv3x3(in_channels=32, out_channels=64)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.activate = nn.ReLU()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    """
    Makes a convolutional layer with a batchnorm at the end of it
    """
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=conv3x3, bias=False, stride=self.downsampling),
            nn.ReLU(),
            conv_bn(self.out_channels, self.expanded_channels, conv=conv3x3, bias=False),
        )
    
class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ChessNet(torch.nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        #input dim: 7x8x8

        #Attempting to use two networks, effectively; one for "piece to move", one for "space to move to"
        #They get combined into one tensor in the output, but the loss should go backwards in a sensible fashion
        self.sNet = torch.nn.Sequential(\
                        nn.Conv2d(7, 32, 5, padding=2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        ResNetBasicBlock(32, 32),
                        nn.PReLU(),
                        nn.Dropout2d(p=0.25),
                        ResNetBasicBlock(32, 48),
                        nn.ReLU(),
                        ResNetBasicBlock(48, 48),
                        nn.ReLU(),
                        nn.Dropout2d(),
                        ResNetBasicBlock(48, 64),
                        nn.ReLU(),
                        nn.Conv2d(64, 96, 5, padding=1),
                        nn.MaxPool2d(2, 1),
                        nn.BatchNorm2d(96),
                        nn.Conv2d(96, 384, 5),
                        nn.BatchNorm2d(384),
                        nn.Dropout(),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(384, 192),
                        nn.PReLU(),
                        nn.Linear(192, 66),
                        nn.Sigmoid(),
                        )
        self.eNet = torch.nn.Sequential(\
                        nn.Conv2d(7, 32, 5, padding=2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        ResNetBasicBlock(32, 32),
                        nn.PReLU(),
                        nn.Dropout2d(p=0.25),
                        ResNetBasicBlock(32, 48),
                        nn.ReLU(),
                        ResNetBasicBlock(48, 48),
                        nn.ReLU(),
                        nn.Dropout2d(),
                        ResNetBasicBlock(48, 64),
                        nn.ReLU(),
                        nn.Conv2d(64, 96, 5, padding=1),
                        nn.MaxPool2d(2, 1),
                        nn.BatchNorm2d(96),
                        nn.Conv2d(96, 384, 5),
                        nn.BatchNorm2d(384),
                        nn.Dropout(),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(384, 192),
                        nn.PReLU(),
                        nn.Linear(192, 66),
                        nn.Sigmoid(),
                        )

    def forward(self, x):
        starts  = self.sNet(x)
        ends    = self.eNet(x)
        return torch.cat((starts, ends), dim=1)
        

def trainModel(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            if (batch_idx % 20 == 0):
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    return model 


def testModel(model, test_loader):
    model.eval()
    test_loss = 0
    correctS = 0
    correctE = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.float())
            test_loss += torch.nn.functional.binary_cross_entropy(output, target.float(), reduction='sum').item()/len(data)  # sum up batch loss
            outputS = output[:, :66]
            outputE = output[:, 66:]
            targetS = target[:, :66]
            targetE = target[:, 66:]
            predS = outputS.argmax(dim=1, keepdim = True)
            predE = outputE.argmax(dim=1, keepdim = True)
            correctS += predS.eq(targetS.argmax(dim=1).view_as(predS)).sum().item()
            correctE += predE.eq(targetE.argmax(dim=1).view_as(predS)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: ({},{})/{} ({:.1f}%/{:.1f}%)\n'.format(
        test_loss, correctS, correctE, len(test_loader.dataset),
        100. * correctS / len(test_loader.dataset),
        100. * correctE / len(test_loader.dataset)))


    return test_loss







