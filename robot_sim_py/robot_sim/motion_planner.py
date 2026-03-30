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
