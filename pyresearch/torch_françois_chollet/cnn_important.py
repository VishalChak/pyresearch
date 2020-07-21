import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5, ) , (.5, ))])

trainset = torchvision.datasets.MNIST(root='./data',train=True, download=True,  transform=transform)
testset = torchvision.datasets.MNIST(root='./data',train=False, download=True,  transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 7 * 7, 10) 

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2 
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out


if __name__ == "__main__":
    print('#### Train data ####')
    print(len(trainset))
    data_iterator = iter(trainloader)
    images, labels = next(data_iterator)
    print(images.shape)
    print(labels.shape)


    print('#### Test data ####')
    print(len(testset))
    data_iterator = iter(testloader)
    images, labels = next(data_iterator)
    print(images.shape)
    print(labels.shape)

    model = CNNModel()
    print(model)

    creterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(10):
        running_loss = 0.0
        for i ,data in enumerate(trainloader):
            inputs, labels = data
            out = model(inputs)

            optimizer.zero_grad()
            loss = creterion(out, labels)
            loss.backward()
            optimizer.step()

##            print statistic
            running_loss += loss.item()

            if i%100 == 99:
                correct = 0
                total = 0

                for images, labels in testloader:
                    out = model(images)
                    _, pred = torch.max(out.data, 1)

                    total += labels.size(0)
                    correct +=(pred == labels).sum()
                acc = 100.00 * correct /total

                print( 'Epoch: {}, loss: {:.4f}, Acc: {:.2f}%'.format(epoch, running_loss/100, acc))
                running_loss == 0.0






