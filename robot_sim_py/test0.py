import trimesh
import numpy as np


def add(a, b):
    return a + b


def fun(c, *param):
    c = add(*param)
    return c


c = 0
a = 1
b = 2
cc = fun(c, a, b)

print(cc)
