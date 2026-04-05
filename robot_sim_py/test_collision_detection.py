import numpy as np
from robot_sim.robot_model import RobotModel
import time
from scipy.spatial.transform import Rotation
import trimesh
import pyvista as pv
from robot_sim.collision import Collision, CollisionDetectionMethod

# method = CollisionDetectionMethod.BoundingSphere
# method = CollisionDetectionMethod.BoundingBox
# method = CollisionDetectionMethod.BoundingCylinder
# method = CollisionDetectionMethod.OrientedBoundingBox
# method = CollisionDetectionMethod.SphereByBoundingCylinder
# method = CollisionDetectionMethod.SphereByBoundingBox
method = CollisionDetectionMethod.SphereByConvexHull
# method = CollisionDetectionMethod.SphereByOriginalMesh
# method = CollisionDetectionMethod.ConvexHull
# method = CollisionDetectionMethod.OriginalMesh

model_file = "model/panda/meshes/link2.obj"
tri_mesh = trimesh.load(model_file)
mesh1 = Collision(0, tri_mesh, method)

mesh2 = Collision(1, trimesh.primitives.Sphere(0.05), method)

T1 = np.eye(4)
# T1[:3, :3] = Rotation.from_rotvec([1.0, 0.0, 0.0]).as_matrix()
T1[:3, 3] = np.array([0.0, 0.0, 0.0])

T2 = np.eye(4)
# T2[:3, :3] = Rotation.from_rotvec([0.0, 1.0, 0.0]).as_matrix()
T2[:3, 3] = np.array([0.0, 0.0, -0.06])

mesh1.apply_transform(T1)
mesh2.apply_transform(T2)
t0 = time.time()
dis = mesh1.collision_detection(mesh2)
print("Min distance: ", dis)
use_time = (time.time() - t0) * 1000
print("计算耗时：", use_time)

plotter = pv.Plotter()
R0 = np.eye(3)
p0 = np.zeros(3)
plotter.add_arrows(p0, R0[:, 0] * 0.1, color="red")
plotter.add_arrows(p0, R0[:, 1] * 0.1, color="green")
plotter.add_arrows(p0, R0[:, 2] * 0.1, color="blue")
mesh1.show(plotter)
mesh2.show(plotter)
plotter.show()
