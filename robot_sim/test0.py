import trimesh
import numpy as np

a = np.zeros(7)
b = np.random.rand(5)

c = a[:, np.newaxis] + b[np.newaxis, :]

print(c.shape)
print(c)
print(np.min(c))
