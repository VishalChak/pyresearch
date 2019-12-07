#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:54:47 2019

@author: vishal
"""

import torch
print(torch.__version__)

x = torch.ones(2,2, requires_grad = True)
y = x+2

print(y)

print(y.grad_fn)

z = y*y*3

out = z.mean()

print(z, out)

a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)

print(a.requires_grad)

b = (a*a).sum()
print(b.grad_fn)

## Gradient
out.backward()


## print d(out)/dx
print(x.grad)



##Generally speaking, torch.autograd is an engine for computing vector-Jacobian product.
## That is, given any vector v=(v1v2⋯vm)T, compute the product vT⋅J.
## If v happens to be the gradient of a scalar function l=g(y⃗ ),
## that is, v=(∂l∂y1⋯∂l∂ym)T, then by the chain rule, the vector-Jacobian product
## would be the gradient of l with respect to x⃗ :


##This characteristic of vector-Jacobian product makes it very convenient to feed external gradients into a model that has non-scalar output.


x = torch.randn(3, requires_grad = True)
y = x*2


while y.data.norm() < 1000:
    y = y*2
print(y)


v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)

y.backward(v)
print(x.grad)


print(x.requires_grad)

print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)


print(x.requires_grad)
y = x.detach() 
print(y.requires_grad)   
print(x.eq(y).all()) 