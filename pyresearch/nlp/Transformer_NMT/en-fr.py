#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:33:45 2019

@author: vishal
"""

import torch

sent = "I love Dolly!!"

## List of model avilable on hub
for model in torch.hub.list('pytorch/fairseq'):
    print(model)    
    
    
# Load an En-De Transformer model trained on WMT'19 data: German
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
## Access the underlying TransformerModel
#assert isinstance(en2de.models[0], torch.nn.Module)

de = en2de.translate(sent)
print("German: ", de)
