#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:42:51 2019

@author: vishal
"""

## load and normalized data


import torch
import torchvision
import torchvision.transforms as transforms

##The output of torchvision datasets are PILImage images of range [0, 1]. 
##We transform them to Tensors of normalized range [-1, 1]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root = "/home/vishal/datasets/CIFAR10",
                                        train=True, download=True,  transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root = "/home/vishal/datasets/CIFAR10",
                                        train=False, download=True,  transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False,
                                          num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



## Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  ## unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
# get some random training images
dataiter= iter(trainloader)
imgs, labels = dataiter.next()
## show img
imshow(torchvision.utils.make_grid(imgs))
##print labels

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


### Define a Convolutional Neural Network

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
## Define a Loss function and optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

### Train the network

for epoch in range(2):   ###   loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize   #### Mast hai
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        
        if i% 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch +1, i+1, running_loss / 100 ))
            running_loss = 0.0

print('Finished Training')


PATH = '/home/vishal/pyresearch/tutorials/cifar_net.pth'
torch.save(net.state_dict(), PATH)
        
        
        
        
    

        

