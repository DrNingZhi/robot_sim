import trimesh
import numpy as np
import pyvista as pv


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = np.vstack((a, b))
print(c)

print(np.min(c))
