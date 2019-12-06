# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:03:53 2019

@author: vishal
"""

###AUTOGRAD   AUTOMATIC DIFFERENTIATION

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='D:/datasets/data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)