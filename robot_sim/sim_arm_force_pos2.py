import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from robot_sim.robot_model import RobotModel
from scipy.spatial.transform import Rotation
from robot_sim.motion_planner import load_trajectory_data
from robot_sim.controller import RobotForcePositionController
from robot_sim.contact_force import get_contact_force, show_contact_force

mjcf_file = "model/ur5/ur5_move_block.xml"

# Load model and initialize data
model = mujoco.MjModel.from_xml_path(mjcf_file)
data = mujoco.MjData(model)

dt = model.opt.timestep
dof = model.nv

# t, q, qd, qdd, pos, vel, acc = plan(1.0, 5.0, 1.0)
robot_model = RobotModel("model/ur5/ur5.xml")
t, q, qd, qdd, pos, vel, acc = load_trajectory_data("data/traj_1_5_1.txt", 6)
orientation = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()

# Initialize joint states
data.qpos = np.hstack((q[0], 0.0))
data.qvel = np.hstack((qd[0], 0.0))

# Initialize controller
kp_pos = np.array([2000.0, 2000.0, 2000.0, 100.0, 100.0, 100.0])
kd_pos = np.array([20.0, 20.0, 20.0, 1.0, 1.0, 1.0])
kp_frc = np.ones(6) * 0.05
ki_frc = np.ones(6) * 0.05
robot_controller = RobotForcePositionController(
    kp_pos, kd_pos, kp_frc, ki_frc, robot_model, "wrist_3_link", np.array([0, 0, 0.05])
)

# record actual data
F_dir = 20.0 + 0.0 * np.sin(t)
F_act = F_dir.copy()

p_barrier = 0.1 * np.sin(t)

S = np.eye(6)
S[0, 0] = 0.0
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
        data.ctrl[0:6] = robot_controller.update(
            data.qpos[:6],
            data.qvel[:6],
            pos[step],
            orientation,
            vel[step],
            np.zeros(3),
            F,
            F_d,
            S,
        )
        data.ctrl[-1] = p_barrier[step]
        mujoco.mj_step(model, data)

        viewer.sync()
        step = step + 1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
