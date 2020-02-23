import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms



data_root = "D:/datasets/bkp_old/datasets/CNN/data_cnn_2_1/train/"

train_transforms = transforms.Compose([transforms.CenterCrop(10),
                                    transforms.ToTensor])
dataloader = datasets.ImageFolder(root = data_root, transform= train_transforms)