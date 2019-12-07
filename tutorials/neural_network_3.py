#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:31:24 2019

@author: vishal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output chennels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        
        ## an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(16 * 6 * 6, 120)   # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_fearure  = 1
        
        for s in size:
            num_fearure *=s
            
        return num_fearure
        
net = Net()
print(net)

## The learnable parameters of a model are returned by net.parameters()

params = list(net.parameters())
print(len(params))
print(params[0].size())


## Let’s try a random 32x32 input.
x = torch.randn(1, 1, 32, 32)
out = net(x)
print(out)

##Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))   


### Loss fuction
#A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target.
#A simple loss is: nn.MSELoss which computes the mean-squared error between the input and the target.

output = net(x)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)   ###make it the same shape as output

criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  ### MSE Loss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) ## Relu 


### Backprop

### To backpropagate the error all we have to do is to loss.backward(). You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.


                ### IMp ###
            ## Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.
        
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)



##Update the weights
#The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):

#weight = weight - learning_rate * gradient


learning_rate = 0.01

for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    
    
#However, as you use neural networks, you want to use various different update rules
# such as SGD, Nesterov-SGD, Adam, RMSProp, etc. To enable this, we built a small
# package: torch.optim that implements all these methods
    
    
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01)

## in ur traning loop

optimizer.zero_grad()  ## zero the gradient buffer
output = net(x)
loss = criterion(output, target)
loss.backward()
optimizer.step()