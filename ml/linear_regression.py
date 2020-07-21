import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class LinearRegrassion(nn.Module):
    def __init__(self):
        super(LinearRegrassion, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred



if __name__ == "__main__":
    X = Variable(torch.Tensor([[1.00], [2.00], [3.00]]))
    y = Variable(torch.Tensor([[2.00], [4.00], [6.00]]))

    model = LinearRegrassion()
    critirion = nn.MSELoss(size_average = False)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    for epoch in range(100000):
        y_pred = model(X)
        
        loss = critirion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch %100 ==99:
            print('epoch : {}, Loss: {}'.format(epoch, loss.data.item()))

    new_var = Variable(torch.Tensor([[10]]))
    print(model(new_var).item())
        
