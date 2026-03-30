import numpy as np
from scipy.spatial.transform import Rotation
from .robot_model import RobotModel
from .motion_planner import circular_trajectory


def plan_ee_circle(robot_model, T_init, T_motion, T_hold):
    dt = robot_model.dt
    dof = robot_model.dof

    # 生成末端圆形轨迹
    center = np.array([0.8, 0.0, 1.2])
    radius = 0.1
    rotation = Rotation.from_rotvec([0.0, np.pi / 2, 0.0])
    period = T_motion
    t, pos, vel, acc = circular_trajectory(
        center, radius, rotation.as_matrix(), dt, period
    )

    # 求解轨迹ik
    num_step = len(pos)
    quat = rotation.as_quat(scalar_first=True)
    quats = np.tile(quat, (num_step, 1))
    body_name = "wrist_3_link"
    point_at_body = np.array([0, 0, 0.05])
    ang_vel = np.zeros((num_step, 3))
    ang_acc = np.zeros((num_step, 3))
    q, qd, qdd = robot_model.trajectory_ik(
        pos, vel, acc, quats, ang_vel, ang_acc, body_name, point_at_body=point_at_body
    )

    # 扩展初始化和结束静止阶段
    init_steps = int(T_init / dt)
    hold_steps = int(T_hold / dt)

    t_motion = t + T_init
    t_init = np.linspace(0.0, T_init - dt, init_steps)
    t_hold = np.linspace(dt, T_hold, hold_steps)
    t_hold = t_hold + T_init + T_motion
    t = np.concatenate((t_init, t_motion, t_hold))

    q_init = np.tile(q[0], (init_steps, 1))
    q_hold = np.tile(q[-1], (hold_steps, 1))
    q = np.vstack((q_init, q, q_hold))

    qd_init = np.zeros((init_steps, dof))
    qd_hold = np.zeros((hold_steps, dof))
    qd = np.vstack((qd_init, qd, qd_hold))

    qdd_init = np.zeros((init_steps, dof))
    qdd_hold = np.zeros((hold_steps, dof))
    qdd = np.vstack((qdd_init, qdd, qdd_hold))

    pos_init = np.tile(pos[0], (init_steps, 1))
    pos_hold = np.tile(pos[-1], (hold_steps, 1))
    pos = np.vstack((pos_init, pos, pos_hold))

    vel_init = np.zeros((init_steps, 3))
    vel_hold = np.zeros((hold_steps, 3))
    vel = np.vstack((vel_init, vel, vel_hold))

    acc_init = np.zeros((init_steps, 3))
    acc_hold = np.zeros((hold_steps, 3))
    acc = np.vstack((acc_init, acc, acc_hold))

    return t, q, qd, qdd, pos, vel, acc


def save_trajectory_data(data, file_name):
    t, q, qd, qdd, pos, vel, acc = data
    data_ = np.hstack((t.reshape(-1, 1), q, qd, qdd, pos, vel, acc))
    np.savetxt(file_name, data_)


def load_trajectory_data(file_name, dof):
    data = np.loadtxt(file_name)
    if not data.shape[1] == 3 * dof + 10:
        raise ValueError(file_name + " has wrong size!")
    t = data[:, 0]
    q = data[:, 1 : (dof + 1)]
    qd = data[:, (dof + 1) : (2 * dof + 1)]
    qdd = data[:, (2 * dof + 1) : (3 * dof + 1)]
    pos = data[:, (3 * dof + 1) : (3 * dof + 4)]
    vel = data[:, (3 * dof + 4) : (3 * dof + 7)]
    acc = data[:, (3 * dof + 7) : (3 * dof + 10)]
    return t, q, qd, qdd, pos, vel, acc


def plan_for_force_control(robot_model, T_forward, T_contact, T_backward):
    dt = robot_model.dt
    dof = robot_model.dof
    center = np.array([0.8, 0.0, 1.2])
    radius = 0.1
    rotation = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()
    period = T_contact
    t, pos, vel, acc = circular_trajectory(center, radius, rotation, dt, period)

    forward_steps = int(T_forward / dt)
    backward_steps = int(T_backward / dt)

    t_contact = t + T_forward
    t_forward = np.linspace(0.0, T_forward - dt, forward_steps)
    t_backward = np.linspace(dt, T_backward, backward_steps) + T_forward + T_contact
    t = np.concatenate((t_forward, t_contact, t_backward))

    pos_forward = np.zeros([forward_steps, 3])
    vel_forward = np.zeros([forward_steps, 3])
    acc_forward = np.zeros([forward_steps, 3])
    pos_forward[:, 0], vel_forward[:, 0], acc_forward[:, 0] = interpolation(
        t_forward, T_forward, 0.1
    )
    pos_forward += pos[0]
    pos_forward[:, 0] -= 0.1

    pos_backward = np.zeros([forward_steps, 3])
    vel_backward = np.zeros([forward_steps, 3])
    acc_backward = np.zeros([forward_steps, 3])
    pos_backward[:, 0], vel_forward[:, 0], acc_forward[:, 0] = interpolation(
        t_backward - T_forward - T_contact, T_backward, -0.1
    )
    pos_backward += pos[-1]

    pos = np.vstack((pos_forward, pos, pos_backward))
    vel = np.vstack((vel_forward, vel, vel_backward))
    acc = np.vstack((acc_forward, acc, acc_backward))

    return t, pos, vel, acc
