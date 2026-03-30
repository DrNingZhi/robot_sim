import numpy as np
from robot_sim.robot_model import RobotModel
import time
from robot_sim.collision import Collision
import trimesh

mjcf_file = "model/ur5/ur5.xml"
robot_model = RobotModel(mjcf_file)


# q = np.zeros(robot_model.dof)
# q = np.random.rand(robot_model.dof)
q = np.array([[0.6900721, 0.95876795, 0.06526024, 0.2282442, 0.66531103, 0.38351114]])

level = 1

mesh1 = Collision(10, trimesh.primitives.Sphere(0.05))

t0 = time.time()

Ts = np.eye(4)
Ts[:3, 3] = np.array([0.3, 0.3, 0.5])
mesh1.apply_transform(Ts, level=level)

dis = robot_model.collision_detection(q, [mesh1], level)
t1 = time.time()

print("碰撞检测耗时：", t1 - t0)
print("最小距离：", dis)

robot_model.show_collision(q, level, [mesh1], [Ts])
