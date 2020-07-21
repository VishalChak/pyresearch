import torch
tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')


import numpy as np
from scipy.io.wavfile import write

tacotron2 = tacotron2.to('cuda')
tacotron2.eval()