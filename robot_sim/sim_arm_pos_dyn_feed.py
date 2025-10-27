import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt

from robot_sim.robot_model import RobotModel
from robot_sim.motion_planner import load_trajectory_data
from robot_sim.controller import RobotPDController

mjcf_file = "model/ur5/ur5.xml"

# Load model and initialize data
model = mujoco.MjModel.from_xml_path(mjcf_file)
data = mujoco.MjData(model)

dt = model.opt.timestep
dof = model.nv

# t, q, qd, qdd, pos, vel, acc = plan_ee_circle(1.0, 5.0, 1.0)
robot_model = RobotModel(mjcf_file)
t, q, qd, qdd, pos, vel, acc = load_trajectory_data("data/traj_1_5_1.txt", dof)

# Initialize joint states
data.qpos = q[0]
data.qvel = np.zeros(dof)

# Initialize controller
kp = np.array([1000.0, 1000.0, 1000.0, 10.0, 10.0, 0.1])
kd = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.001])
robot_controller = RobotPDController(kp, kd)

# record actual data
q_act = np.zeros((q.shape))
pos_act = np.zeros((pos.shape))

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and step < len(t):
        step_start = time.time()

        q_act[step] = data.qpos
        pos_act[step], _ = robot_model.forward_kinematics(
            q_act[step], body_name="wrist_3_link", point_at_body=np.array([0, 0, 0.05])
        )

        tau_ref = robot_model.inverse_dynamics(q[step], qd[step], qdd[step])
        data.ctrl = robot_controller.update_with_feedforward(
            data.qpos, data.qvel, q[step], qd[step], tau_ref
        )
        mujoco.mj_step(model, data)

        viewer.sync()
        step = step + 1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


plt.figure(constrained_layout=True)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.plot(t, q[:, i], "b-")
    plt.plot(t, q_act[:, i], "r--")
    plt.xlabel("t")
    plt.ylabel("q of Joint " + str(i + 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", label="plan")
ax.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], "r--", label="act")
plt.axis("equal")
plt.legend()

plt.figure()
err = np.linalg.norm(pos - pos_act, axis=1)
print("最大误差：", np.max(err))
plt.plot(t, err)
plt.xlabel("t")
plt.ylabel("tracking error")

plt.show()
