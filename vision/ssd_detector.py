#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:57:14 2019

@author: vishal
"""

import torch 
precision = 'fp32'
#ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision, map_location=torch.device('cpu'))
for model in torch.hub.list('NVIDIA/DeepLearningExamples'):
    print(model)


