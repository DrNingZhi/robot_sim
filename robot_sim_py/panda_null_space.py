import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from robot_sim.robot_model import RobotModel
from robot_sim.controller import RobotPDController

mjcf_file = "model/panda/panda.xml"
model = mujoco.MjModel.from_xml_path(mjcf_file)
data = mujoco.MjData(model)

dt = model.opt.timestep
dof = model.nv
robot_model = RobotModel(mjcf_file)

# q0 = robot_model.rand_configuration()
# q0[4] = -2.8973
# print(q0)

q0 = np.array([0.9835, -0.0698, 1.9154, -1.0633, -2.8973, 0.4735, -2.3276])

# Initialize joint states
data.qpos = q0
data.qvel = np.zeros(dof)

# Initialize controller
kp = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
kd = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
robot_controller = RobotPDController(kp, kd)

step = 0
q = q0.copy()
reach_limit = False
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        if not reach_limit:
            qd_plan = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            null_space_proj = robot_model.null_space_projection(
                q, "panda_link7", point_on_body=np.array([0.0, 0.0, 0.1])
            )
            qd = null_space_proj @ qd_plan
            q_new = q + qd * dt
            if np.any(q_new > robot_model.joint_upper_limits) or np.any(
                q_new < robot_model.joint_lower_limits
            ):
                reach_limit = True
            else:
                q = q_new

        tau_ref = robot_model.gravity_torque(q)
        data.ctrl = robot_controller.update_with_feedforward(
            data.qpos, data.qvel, q, np.zeros(dof), tau_ref
        )
        mujoco.mj_step(model, data)

        viewer.sync()
        step = step + 1

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
