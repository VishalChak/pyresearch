import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1,6,3)




if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle= True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)

    #### train###
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    print(len(train_dataset))
    print(images.shape)
    print(labels.shape)

    ### Test ###
    dataiter  = iter(test_dataloader)
    images , labels = dataiter.next()
    print(len(test_dataset))
    print(images.shape)
    print(labels.shape)

