import trimesh
import numpy as np

sphere = trimesh.primitives.Sphere(1.0)
print(type(sphere) == trimesh.primitives.Trimesh)

a = np.array([0.0, 1.0])
print(type(a) == np.ndarray)
