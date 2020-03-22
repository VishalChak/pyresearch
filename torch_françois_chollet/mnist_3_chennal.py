import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms



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
    
    net = Net()
    print(net)

