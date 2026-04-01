import numpy as np
import time
import trimesh
from scipy.spatial.transform import Rotation
import pyvista as pv
from robot_sim.collision import Collision, CollisionDetectionMethod

model_file = "model/panda/meshes/link2.obj"
mesh = trimesh.load(model_file)

method = CollisionDetectionMethod.BoundingSphere
# method = CollisionDetectionMethod.BoundingBox
# method = CollisionDetectionMethod.BoundingCylinder
# method = CollisionDetectionMethod.OrientedBoundingBox
# method = CollisionDetectionMethod.SphereByBoundingCylinder
# method = CollisionDetectionMethod.SphereByBoundingBox
# method = CollisionDetectionMethod.SphereByConvexHull
# method = CollisionDetectionMethod.SphereByOriginalMesh
# method = CollisionDetectionMethod.ConvexHull
# method = CollisionDetectionMethod.OriginalMesh

collision = Collision(0, mesh, method)


T1 = np.eye(4)
T1[:3, :3] = Rotation.from_rotvec([1.0, 0.0, 0.0]).as_matrix()
T1[:3, 3] = np.array([0.0, 0.0, 0.04])
collision.apply_transform(T1)

plotter = pv.Plotter()
R0 = np.eye(3)
p0 = np.zeros(3)
plotter.add_arrows(p0, R0[:, 0] * 0.1, color="red")
plotter.add_arrows(p0, R0[:, 1] * 0.1, color="green")
plotter.add_arrows(p0, R0[:, 2] * 0.1, color="blue")
collision.show(plotter)
plotter.show()
