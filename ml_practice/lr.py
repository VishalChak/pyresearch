import torch
from torch.autograd import Variable
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.liner = nn.Linear(3,2)
        self.liner2 = nn.Linear(2,1)


    def forward(self, x):
        y_pred = self.liner2(self.liner(x))
        # or
        #x = self.liner(x)
        #x = self.liner2(x)
        #print(x)
        return y_pred

if __name__ == "__main__":

    # X = Variable(torch.Tensor([[2.0, 3.0, 23.0], [6.0, 12.0, 4.0], [4.0,26.0, 33.0],  [4.0,56.0,2.0] ]))
    # y = Variable(torch.Tensor([[28.0], [22.0] , [63.0], [62.0] ]))
    
    #X = Variable(torch.Tensor([  [2.0,4.0,1.0], [2.0,7.0,5.0],  [9.0,3.0,6.0],  [23.0,32.0,15.0],  [12.0,17.0,50.0]  ]))
    #y = Variable(torch.Tensor([  [5.0],          [4.0],           [6.0],        [40.0],            [21.0]]))

    X = Variable(torch.Tensor([[1.0, 1.0, 4.0], [2.0,2.0, 5.0], [9.0,3.0, 3.0]])) 
    y = Variable(torch.Tensor([[2.0], [4.0], [-6.0]]))

    model = Net()

    # criterion = nn.MSELoss(size_average = False)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 00.1)


    criterion = nn.MSELoss(size_average = False)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    for epoch in range(10000):
        y_pred = model(X)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch %100 == 99:
            print('epoch {}, loss {}'.format(epoch, loss.data.item())) 
    new_var = Variable(torch.Tensor([[10,45, 4]]))
    print(model(new_var))