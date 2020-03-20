import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.liner = nn.Linear(2,1)
    
    def forward(self, x):
        pred_y = self.liner(x)
        return pred_y

if __name__ == "__main__":
    net = Net()
    print(net, net.state_dict())

    X = Variable(torch.Tensor([[1.0,1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]))
    y = Variable(torch.Tensor([[2.0], [4.0], [6.0], [8.0]]))
    print(X)


    critirion = nn.MSELoss(size_average = False)
    optimizer = optim.SGD(net.parameters(), lr = 0.01)
    for epoch in range(5000):
        y_pred = net(X)

        loss = critirion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch %100 ==99:
            print('epoch:{} , loss:{}'.format(epoch, loss.data.item()))
    
    new_var = Variable(torch.Tensor([[100,12]]))
    print(net(new_var))