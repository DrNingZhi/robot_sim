import numpy as np
from robot_sim.robot_model import RobotModel
import time
import trimesh
import pyvista as pv
from robot_sim.collision import Collision

model_file = "model/panda/meshes/link2.obj"
mesh = trimesh.load(model_file)
collision = Collision(0, mesh)

level = 3

T1 = np.eye(4)
collision.apply_transform(T1, level=level)

plotter = pv.Plotter()
collision.show(
    plotter,
    transform=T1,
    show_ori_mesh=True,
    show_convex_hull=True,
    show_bounding_box=False,
    show_bounding_box_oriented=False,
    show_bounding_sphere=False,
)
plotter.show()
