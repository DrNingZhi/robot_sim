import numpy as np
from robot_sim.robot_model import RobotModel
import time
import trimesh

mjcf_file = "model/ur5/ur5.xml"
# mjcf_file = "model/panda/panda.xml"
robot_model = RobotModel(mjcf_file)


# q = np.zeros(robot_model.dof)
# q = np.random.rand(robot_model.dof)
q = np.array([[0.6900721, 0.95876795, 0.06526024, 0.2282442, 0.66531103, 0.38351114]])

t0 = time.time()
min_dis, cd_res = robot_model.self_sphere_fitting_collision_detection(q)
t1 = time.time()
print("自碰撞耗时：", t1 - t0)
print("最小距离：", min_dis)
print(cd_res)


robot_model.show_sphere_fitting_collision(q)
