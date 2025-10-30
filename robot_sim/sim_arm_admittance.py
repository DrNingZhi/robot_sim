import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from robot_sim.robot_model import RobotModel
from robot_sim.controller import RobotPDController, RobotAdmittanceController3d
from robot_sim.keyboard_controller import KeyboardController
from robot_sim.contact_force import get_contact_force, show_contact_force

sim_mjcf_file = "model/ur5/ur5_interact.xml"
rob_mjcf_file = "model/ur5/ur5.xml"

# Load model and initialize data
model = mujoco.MjModel.from_xml_path(sim_mjcf_file)
data = mujoco.MjData(model)

dt = model.opt.timestep
dof = model.nv

robot_model = RobotModel(rob_mjcf_file)

target = np.eye(4)
target[:3, :3] = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()
target[:3, 3] = np.array([0.8, 0.0, 1.1])
q0 = robot_model.inverse_kinematics(
    target, "wrist_3_link", point_at_body=np.array([0, 0, 0.05]), smooth=False
)
data.qpos = np.hstack((q0, np.zeros(3)))
data.qvel = np.zeros(dof)

kp = np.array([100000.0, 100000.0, 100000.0, 1000.0, 1000.0, 10.0])
kd = np.array([50.0, 50.0, 50.0, 1.0, 1.0, 0.01])
robot_controller = RobotPDController(kp, kd)
tau_ref = robot_model.gravity_torque(q0)

keyboard_controller = KeyboardController()
keyboard_controller.start()

M = np.array([1.0, 1.0, 1.0])
B = np.array([0.0, 0.0, 0.0])
K = np.array([1000.0, 1000.0, 1000.0])
robot_admittance_controller = RobotAdmittanceController3d(
    dt, M, B, K, robot_model, "wrist_3_link", np.array([0, 0, 0.05])
)

T = 1000.0
num_step = int(T / dt)
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and step < num_step:
        step_start = time.time()

        f, p = get_contact_force(model, data, "end_effector")
        show_contact_force(viewer, f, p)

        delta_q, delta_qd = robot_admittance_controller.update(
            f,
            data.qpos[: robot_model.dof],
            data.qvel[: robot_model.dof],
            q0,
            np.zeros(robot_model.dof),
        )

        data.ctrl[: robot_model.dof] = robot_controller.update_with_feedforward(
            data.qpos[: robot_model.dof],
            data.qvel[: robot_model.dof],
            q0 + delta_q,
            np.zeros(robot_model.dof),
            tau_ref,
        )
        data.ctrl[-3:] = keyboard_controller.state()
        mujoco.mj_step(model, data)

        viewer.sync()
        step = step + 1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
