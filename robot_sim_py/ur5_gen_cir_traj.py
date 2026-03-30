from robot_sim.motion_planner_ur5 import plan_ee_circle, save_trajectory_data
from robot_sim.robot_model import RobotModel
import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt

mjcf_file = "model/ur5/ur5.xml"
robot_model = RobotModel(mjcf_file)

t, q, qd, qdd, pos, vel, acc = plan_ee_circle(robot_model, 1.0, 5.0, 1.0)

# 测试生成轨迹：使用mujoco inverse的方式（理想跟踪）
model = mujoco.MjModel.from_xml_path(mjcf_file)
data = mujoco.MjData(model)
dt = model.opt.timestep
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and step < len(t):
        step_start = time.time()

        data.qpos = q[step].copy()
        data.qvel = qd[step].copy()
        data.qacc = qdd[step].copy()
        mujoco.mj_inverse(model, data)
        viewer.sync()
        step = step + 1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# 画出空间轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", label="plan")
plt.axis("equal")

# 画出关节规划信息
plt.figure()
plt.subplot(131)
plt.plot(t, q)
plt.xlabel("t")
plt.ylabel("q")
plt.subplot(132)
plt.plot(t, qd)
plt.xlabel("t")
plt.ylabel("qd")
plt.subplot(133)
plt.plot(t, qdd)
plt.xlabel("t")
plt.ylabel("qdd")
plt.show()

# 保存为文件便于后续使用
for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
    data = plan_ee_circle(robot_model, 1.0, i, 1.0)
    data_file = "data/traj_1_" + str(int(i)) + "_1.txt"
    save_trajectory_data(data, data_file)
    print("轨迹规划完成，轨迹数据存储在" + data_file)
