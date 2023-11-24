# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:54:39 2023

@author: Flint
"""
import numpy as np
X = np.linspace(-1,1,1000)

def h_D(x):
    x1 = x[np.random.randint(len(x))]
    x2 = x[np.random.randint(len(x))]
    
    return((x1**2+x2**2)/2)

def get_h_avg(x,p):
    h = 0
    for i in range(p):
        h += h_D(x)
    h /= p
    return(h)

h_avg = get_h_avg(X, 100000)
print("h_avg:",h_avg)

def get_bias(x,p,h):
    bias =0
    for i in range(p):
        x_i = x[np.random.randint(len(x))]
        bias += (h-x_i**2)**2
    bias /= p
    return(bias)

bias = get_bias(X, 100000, h_avg)
print("bias:",bias)

def get_var(x,p,h):
    var = 0
    for i in range(p):
        var += (h_D(x)-h)**2
    var /= p
    return(var)

var = get_var(X, 100000,h_avg)
print("varience:",var)