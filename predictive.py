


import torch
import torch.nn as nn
from torch.autograd import Variable

X = Variable(torch.Tensor([[1.0], [2.0], [3.0]])) 
y = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.liner = nn.Linear(1,1)
    def forward(self, x):
        y_pred = self.liner(x)
        return y_pred


if __name__ == "__main__":
    model = LinearRegressionModel()
    criterion = nn.MSELoss(size_average = False)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    for epoch in range(10000):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 ==99:
            print('epoch {}, loss {}'.format(epoch, loss.data.item())) 
    
    new_var = Variable(torch.Tensor([[4.0]]))
    print(model(new_var))
