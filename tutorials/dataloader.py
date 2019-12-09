# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:46:10 2019

@author: vishal
"""

import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def load_split_train_test(datadir, valid_size = .2):
    data_transforms = transforms.Compose([transforms.Resize(224),  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_data = datasets.ImageFolder(datadir, transform= data_transforms)
    test_data = datasets.ImageFolder(datadir, transform= data_transforms)
    
    num_train = len(train_data)
    indices  = list (range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=4)
    testloader = torch.utils.data.DataLoader(test_data,sampler=test_sampler, batch_size=4)

    return trainloader, testloader

datadir = "D:/datasets/anomaly/data/"
valid_size = .2

trainloader, testloader = load_split_train_test(datadir, valid_size)



import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5  ## unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.xlabel('Radius')
    plt.show()

def imshow_label(img, label):
    img = img / 2 + 0.5  ## unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.xlabel(label)
    plt.show()
    
# get some random training images
dataiter= iter(trainloader)
imgs, labels = dataiter.next()
imshow(torchvision.utils.make_grid(imgs))
for i in range(len(imgs)):
    imshow_label(imgs[i], str(labels[0].item()))