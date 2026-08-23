import mujoco
import mujoco.viewer
import numpy as np
import time
import os

from robot_sim.mink_model_g1 import MinkModelG1
from robot_sim.marker_controller import MarkerController


current_dir = os.path.dirname(os.path.abspath(__file__))
model_file = "model/unitree_g1/g1_23dof.xml"

model = mujoco.MjModel.from_xml_path(model_file)
data = mujoco.MjData(model)
dt = model.opt.timestep
dof = model.nv
q0 = data.qpos.copy()

mink_model = MinkModelG1(model_file)
marker_controller = MarkerController("left_marker", "right_marker")

q_cmd = q0.copy()

current_step = 0
fps = 25
view_skip = int(1.0 / dt / fps)
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 150.0
    viewer.cam.elevation = -30.0
    viewer.cam.lookat = [0.2, 0.0, 0.8]
    while viewer.is_running():
        step_start = time.time()

        marker_controller.step(model)
        marker_targets = marker_controller.get_transforms(model)

        # 初始参考位置
        # wrist_l_marker: [0.23577428 0.14865452 0.8882308]
        # wrist_r_marker: [0.23577428 -0.14865452 0.8882308]
        # foot_l_marker: [0.0 0.1185 0.0]
        # foot_r_marker: [0.0 -0.1185 0.0]
        # com: [0.015 0.0 0.717]

        # test
        # targets1 = np.eye(4)
        # targets1[:3, 3] = np.array([0.23577428, 0.14865452, 0.8882308])
        # targets2 = np.eye(4)
        # targets2[:3, 3] = np.array([0.23577428, -0.14865452, 0.8882308])
        targets3 = np.eye(4)
        targets3[:3, 3] = np.array([0.0, 0.1185, 0.0])
        targets4 = np.eye(4)
        targets4[:3, 3] = np.array([0.0, -0.1185, 0.0])

        com_target = np.array([0.015, 0.0, 0.717])
        targets = [marker_targets[0], marker_targets[1], targets3, targets4]

        q_cmd = mink_model.solve(q_cmd, targets, com_target)
        data.qpos = q_cmd.copy()
        data.qvel = np.zeros(dof)
        data.qacc = np.zeros(dof)

        # data.qpos[7:] = np.zeros(dof-6)
        # data.qvel[6:] = np.zeros(dof-6)
        # data.qacc[6:] = np.zeros(dof-6)
        mujoco.mj_inverse(model, data)

        if current_step % view_skip == 0:
            viewer.sync()

        current_step += 1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)



# 足底目标位置[0.0, +-0.1185, 0.0]