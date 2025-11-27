import numpy as np
from robot_sim.robot_model import RobotModel
import time
import trimesh
import pyvista as pv
from robot_sim.collision import Collision

model_file = "model/panda/meshes/link2.obj"
tri_mesh = trimesh.load(model_file)
vertices = tri_mesh.vertices
faces = tri_mesh.faces


# mesh = Collision(vertices, faces)
mesh1 = Collision(0, tri_mesh)
mesh2 = Collision(1, trimesh.primitives.Sphere(0.05))

T1 = np.eye(4)
T2 = np.eye(4)
T2[:3, 3] = np.array([0.0, 0.0, -0.06])

t0 = time.time()
level = 3
mesh1.apply_transform(T1, level=level)
mesh2.apply_transform(T2, level=level)
dis = mesh1.collision_detection(mesh2, level=level)
print("Min distance: ", dis)
print("计算耗时：", time.time() - t0)

plotter = pv.Plotter()
mesh1.show(
    plotter,
    transform=T1,
    show_ori_mesh=True,
    show_convex_hull=True,
    show_bounding_box=False,
    show_bounding_box_oriented=False,
    show_bounding_sphere=False,
)
mesh2.show(
    plotter,
    transform=T2,
    show_ori_mesh=True,
    show_convex_hull=True,
    show_bounding_box=False,
    show_bounding_box_oriented=False,
    show_bounding_sphere=False,
)
plotter.show()
