import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from robot_sim.robot_model import RobotModel
from robot_sim.motion_planner import load_trajectory_data
from robot_sim.controller import RobotForceController
from robot_sim.contact_force import get_contact_force, show_contact_force

mjcf_file = "model/ur5/ur5_block.xml"

# Load model and initialize data
model = mujoco.MjModel.from_xml_path(mjcf_file)
data = mujoco.MjData(model)

dt = model.opt.timestep
dof = model.nv

# t, q, qd, qdd, pos, vel, acc = plan(1.0, 5.0, 1.0)
robot_model = RobotModel(mjcf_file)
t, q, qd, qdd, pos, vel, acc = load_trajectory_data("data/traj_1_5_1.txt", dof)

# Initialize joint states
data.qpos = q[0]
data.qvel = qd[0]

# Initialize controller
kp = 0.05
ki = 0.05
robot_controller = RobotForceController(
    kp, ki, robot_model, "wrist_3_link", np.array([0, 0, 0.05])
)

# record actual data
F_dir = 20.0 + 10.0 * np.sin(t)
F_act = F_dir.copy()

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and step < len(t):
        step_start = time.time()

        f, p = get_contact_force(model, data, "end_effector")
        show_contact_force(viewer, f, p)

        F = np.zeros(6)
        if not len(f) == 0:
            F = np.array([f[0][0], 0.0, 0.0, 0.0, 0.0, 0.0])
        F_act[step] = F[0]
        F_d = np.array([F_dir[step], 0.0, 0.0, 0.0, 0.0, 0.0])
        data.ctrl = robot_controller.update(data.qpos, data.qvel, F_d, F)
        mujoco.mj_step(model, data)

        viewer.sync()
        step = step + 1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

plt.figure()
plt.plot(t, F_dir, "b", label="plan")
plt.plot(t, F_act, "r--", label="actual")
plt.xlabel("t")
plt.ylabel("contact force (N)")
plt.legend()

plt.show()
