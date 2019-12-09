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
    data_transforms = transforms.Compose([transforms.Resize(64),  transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
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

###  Img show 

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
    
    
    #####

### Net    
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr= 0.01, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    
    for i , data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        
        ## forward + backward + opimize
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        ## print(statics)
        if i%100 ==99:
            print('[%d, %5d] loss: %3f' % (epoch +1, i+1, running_loss/ 100 ))
            running_loss = 0.0
print('Finished Traning')

    
    
    
    