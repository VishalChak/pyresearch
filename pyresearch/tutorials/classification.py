# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:12:54 2019

@author: vishal
"""

import torch
from torchvision import datasets , transforms, models

def load_split_train_test(data, valid_size=.2):
    train_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    
    train_data = datasets.ImageFolder(data, transforms = train_transform)
    test_data = datasets.ImageFolder(data, transforms = test_transform)