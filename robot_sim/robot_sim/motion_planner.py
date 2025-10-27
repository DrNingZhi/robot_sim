import numpy as np
from scipy.spatial.transform import Rotation
from .robot_model import RobotModel


def interpolation(t, T, p):
    A = np.array(
        [
            [T**5, T**4, T**3],
            [5.0 * T**4, 4.0 * T**3, 3.0 * T**2],
            [20.0 * T**3, 12.0 * T**2, 6.0 * T],
        ]
    )
    B = np.array([p, 0.0, 0.0])
    a = np.linalg.inv(A) @ B
    x = a[0] * t**5 + a[1] * t**4 + a[2] * t**3
    xd = 5.0 * a[0] * t**4 + 4.0 * a[1] * t**3 + 3.0 * a[2] * t**2
    xdd = 20.0 * a[0] * t**3 + 12.0 * a[1] * t**2 + 6.0 * a[2] * t
    return x, xd, xdd


def circular_trajectory(center, radius, rotation, dt, period):
    """
    Parameters:
    ----------
    center: np.ndarray(3,)
        Center position of spatial circle
    radius: float
        Tadius of the spatial circle
    rotation: np.ndarray(3,3)
        Pose of the spatial circle, defined by rotation matrix.
        The primitive pose is and on x-y plane.
    dt: float
        Time step.
    period: float
        Period to accomplish the circle trajectory.

    Return:
    ----------
    t: np.ndarray(n,)
        Discrete time series of the trajectory.
    pos: np.ndarray(n,3)
        End-effector position series of the trajectory.
    vel: np.ndarray(n,3)
        End-effector velocity series of the trajectory.
    acc: np.ndarray(n,3)
        End-effector acceleration series of the trajectory.
    """

    num_steps = int(period / dt)
    t = np.linspace(0, period, num_steps + 1)
    theta, theta_d, theta_dd = interpolation(t, period, np.pi * 2)
    pos = np.zeros((num_steps + 1, 3))
    pos[:, 0] = radius * np.cos(theta)
    pos[:, 1] = radius * np.sin(theta)
    pos[:, 2] = 0.0
    vel = np.zeros((num_steps + 1, 3))
    vel[:, 0] = theta_d * radius * np.cos(theta + np.pi / 2)
    vel[:, 1] = theta_d * radius * np.sin(theta + np.pi / 2)
    vel[:, 2] = 0.0
    acc_t = np.zeros((num_steps + 1, 3))
    acc_n = np.zeros((num_steps + 1, 3))
    acc_t[:, 0] = theta_dd * radius * np.cos(theta + np.pi / 2)
    acc_t[:, 1] = theta_dd * radius * np.sin(theta + np.pi / 2)
    acc_t[:, 2] = 0.0
    acc_n[:, 0] = theta_d**2 * radius * np.cos(theta + np.pi)
    acc_n[:, 1] = theta_d**2 * radius * np.sin(theta + np.pi)
    acc_n[:, 2] = 0.0
    acc = acc_n + acc_t
    pos = center + (rotation @ pos.T).T
    vel = (rotation @ vel.T).T
    acc = (rotation @ acc.T).T
    return t, pos, vel, acc


def trajectory_ik(robot_model, pos, vel, acc, orientation, body_name, point_on_body):
    dof = robot_model.dof
    num_steps = len(pos)
    q = np.zeros((num_steps, dof))
    qd = np.zeros((num_steps, dof))
    qdd = np.zeros((num_steps, dof))
    for i in range(num_steps):
        target = np.eye(4)
        target[:3, 3] = pos[i]
        target[:3, :3] = orientation
        q[i] = robot_model.inverse_kinematics(
            target, body_name, q[i - 1], point_on_body, smooth=(not i == 0)
        )
        vel_a = np.array([vel[i, 0], vel[i, 1], vel[i, 2], 0.0, 0.0, 0.0])
        acc_a = np.array([acc[i, 0], acc[i, 1], acc[i, 2], 0.0, 0.0, 0.0])
        qd[i], qdd[i] = robot_model.inverse_kinematics_d(
            q[i], vel_a, acc_a, body_name, point_on_body
        )
        if i % 100 == 0:
            print("轨迹生成中，第" + str(i) + "步，共" + str(num_steps) + "步")
    return q, qd, qdd


def plan_ee_circle(robot_model, T_init, T_motion, T_hold):
    dt = robot_model.dt
    dof = robot_model.dof
    center = np.array([0.8, 0.0, 1.2])
    radius = 0.1
    rotation = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()
    period = T_motion
    t, pos, vel, acc = circular_trajectory(center, radius, rotation, dt, period)
    orientation = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()
    body_name = "wrist_3_link"
    point_at_body = np.array([0, 0, 0.05])
    q, qd, qdd = trajectory_ik(
        robot_model, pos, vel, acc, orientation, body_name, point_at_body
    )

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
