import numpy as np


def auto_gradient(fun, x, *params):
    h = 0.001
    dim = len(x)
    x0 = x.copy()
    grad = np.zeros(dim)
    for i in range(dim):
        x1 = x0.copy()
        x1[i] += h
        x2 = x0.copy()
        x2[i] -= h
        grad[i] = (fun(x1, *params) - fun(x2, *params)) / (h * 2.0)
    return grad
