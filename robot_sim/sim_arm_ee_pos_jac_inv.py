import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from robot_sim.robot_model import RobotModel
from robot_sim.motion_planner import load_trajectory_data
from robot_sim.controller import RobotJacInvController, RobotJacTController

mjcf_file = "model/ur5/ur5.xml"

# Load model and initialize data
model = mujoco.MjModel.from_xml_path(mjcf_file)
data = mujoco.MjData(model)

dt = model.opt.timestep
dof = model.nv

# t, q, qd, qdd, pos, vel, acc = plan_ee_circle(1.0, 5.0, 1.0)
robot_model = RobotModel(mjcf_file)
t, q, qd, qdd, pos, vel, acc = load_trajectory_data("data/traj_1_5_1.txt", dof)
orientation = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()

# Initialize joint states
data.qpos = q[0]
data.qvel = qd[0]

# Initialize controller
kp = np.array([1000.0, 1000.0, 1000.0, 10.0, 10.0, 0.1])
kd = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.001])
robot_controller = RobotJacInvController(
    kp, kd, robot_model, "wrist_3_link", np.array([0, 0, 0.05])
)

# record actual data
pos_act = np.zeros((pos.shape))
mat_act = np.zeros((3, 3, len(t)))

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and step < len(t):
        step_start = time.time()

        pos_act[step], mat_act[:, :, step] = robot_model.forward_kinematics(
            data.qpos, body_name="wrist_3_link", point_at_body=np.array([0, 0, 0.05])
        )

        data.ctrl = robot_controller.update(
            data.qpos,
            data.qvel,
            pos[step],
            orientation,
            vel[step],
            np.zeros(3),
        )
        mujoco.mj_step(model, data)

        viewer.sync()
        step = step + 1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")  # 启用3D坐标轴
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", label="plan")
ax.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], "r--", label="act")
plt.axis("equal")
plt.legend()

plt.figure()
plt.subplot(121)
err = np.linalg.norm(pos - pos_act, axis=1)
print("最大位置误差：", np.max(err))
plt.plot(t, err)
plt.xlabel("t")
plt.ylabel("tracking error (m)")

plt.subplot(122)
pose_err = np.zeros(len(t))
for i in range(len(t)):
    R = orientation.T @ mat_act[:, :, i]
    w = Rotation.from_matrix(R).as_rotvec()
    pose_err[i] = np.linalg.norm(w)
print("最大角度误差：", np.max(pose_err))
plt.plot(t, pose_err)
plt.xlabel("t")
plt.ylabel("orientation error (rad)")

plt.show()
