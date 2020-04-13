import numpy as np
import pandas as pd
import torch



wine_path = "/home/vishal/datasets/winequality-white.csv"
dataframe = pd.read_csv(wine_path , sep = ';')

print("Pandas ----------")
print(dataframe.head())
print(dataframe.shape)
print(dataframe.columns)
print(dataframe.dtypes)

wine_torch = torch.from_numpy(dataframe.to_numpy())

print("Torch-----------")
print(wine_torch.shape)
print(wine_torch.type())


X = wine_torch[:, :-1]
print(X)
print(X.shape)

y = wine_torch[:, -1]
print(y)
print(y.shape)

y = wine_torch[:, -1].long()
print(y)
print(y.shape)

print("If target are purely qualitative, such as color, one-hot encoding is a much better fit")

y_one_hot = torch.zeros(y.shape[0], 10)
y_one_hot = y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
print(y_one_hot)
print(y_one_hot.shape)

print(y.unsqueeze(1))


print("data manupulation----")
X_mean = torch.mean(X, dim=0)
print("data_mean" ,  X_mean, X_mean.shape)
X_variance = torch.var(X, dim = 0)

X_normalized = (X - X_mean) / torch.sqrt(X_variance)
print(X_normalized)

print(y)
