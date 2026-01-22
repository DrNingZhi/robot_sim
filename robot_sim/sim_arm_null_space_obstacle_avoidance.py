import numpy as np
import mujoco
import mujoco.viewer
import time
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from robot_sim.robot_model import RobotModel
from robot_sim.controller import RobotPDController
from robot_sim.collision import Collision, SphereFittingCollision
from robot_sim.auto_gradient import auto_gradient

mjcf_file = "model/panda/panda.xml"
model = mujoco.MjModel.from_xml_path(mjcf_file)
data = mujoco.MjData(model)

dt = model.opt.timestep
dof = model.nv
robot_model = RobotModel(mjcf_file, collision_detect_enable=True)

q0 = np.array([0.9835, -0.0698, 1.9154, -1.0633, -2.8973, 0.4735, -2.3276])

# Initialize joint states
data.qpos = q0
data.qvel = np.zeros(dof)

# Initialize controller
kp = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
kd = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
robot_controller = RobotPDController(kp, kd)

# define obstacle
p_obs_1 = np.array([0.0, 0.2, 0.8])
p_obs_2 = np.array([-0.6, 0.0, 0.2])
obs_move_period = 8.0
v_obs = (p_obs_2 - p_obs_1) / (obs_move_period / 2.0)

pose_obs = np.eye(4)
# obs = Collision(0, trimesh.primitives.Sphere(radius=0.05))
# obs.apply_transform(pose, level=1)
obs = SphereFittingCollision(0, trimesh.primitives.Sphere(radius=0.05), num_groups=1)


def calc_distance(q, obs):
    # dis = robot_model.collision_detection(q, [obs], level=1)
    dis = robot_model.sphere_fitting_collision_detection(q, [obs])
    return np.min(dis)


t = 0.0
q = q0.copy()
reach_limit = False
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # step_start = time.time()

        t_obs = t % obs_move_period
        if t_obs <= (obs_move_period / 2.0):
            p_obs = p_obs_1 + v_obs * t_obs
        else:
            p_obs = p_obs_2 - v_obs * (t_obs - obs_move_period / 2.0)
        pose_obs[:3, 3] = p_obs
        obs.apply_transform(pose_obs)

        qd_plan = 100.0 * auto_gradient(calc_distance, q, obs)
        null_space_proj = robot_model.null_space_projection(
            q, "panda_link7", point_on_body=np.array([0.0, 0.0, 0.1])
        )
        qd = null_space_proj @ qd_plan
        q_new = q + qd * dt
        if np.any(q_new > robot_model.joint_upper_limits) or np.any(
            q_new < robot_model.joint_lower_limits
        ):
            pass
        else:
            q = q_new

        tau_ref = robot_model.gravity_torque(q)
        data.ctrl = robot_controller.update_with_feedforward(
            data.qpos, data.qvel, q, np.zeros(dof), tau_ref
        )
        mujoco.mj_step(model, data)

        # visualize virtual obstacle
        # viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.05, 0.0, 0.0]),
            p_obs,
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
        )
        viewer.user_scn.ngeom = 1

        viewer.sync()
        t = t + dt

        # time_until_next_step = dt - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)
