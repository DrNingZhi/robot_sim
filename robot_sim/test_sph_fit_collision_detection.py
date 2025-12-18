import numpy as np
from robot_sim.robot_model import RobotModel
import time
import trimesh
import pyvista as pv
from robot_sim.collision import SphereFittingCollision

model_file = "model/panda/meshes/link2.obj"
tri_mesh = trimesh.load(model_file)
sph_fit_1 = SphereFittingCollision(0, tri_mesh)
sph_fit_1.cluster(10)

cylinder = trimesh.primitives.Cylinder(radius=0.05, height=0.1)
sph_fit_2 = SphereFittingCollision(1, cylinder)
sph_fit_2.cluster(10)

T1 = np.eye(4)
T1[:3, 3] = np.array([0.0, 0.0, 0.06])
sph_fit_1.apply_transform(T1)

T2 = np.eye(4)
T2[:3, 3] = np.array([0.0, 0.0, 0.16])
sph_fit_2.apply_transform(T2)

t0 = time.time()
dis = sph_fit_1.collision_detection(sph_fit_2)
print("Min distance: ", dis)
print("计算耗时：", time.time() - t0)

plotter = pv.Plotter()
sph_fit_1.show_spheres(plotter)
sph_fit_2.show_spheres(plotter)
plotter.show()
