import numpy as np


def rotation_matrix_from_z_axis(z_axis):
    z_axis = z_axis / np.linalg.norm(z_axis)
    z0 = np.array([0.0, 0.0, 1.0])
    if z_axis == z0:
        return np.eye(3)
    x_axis = np.cross(z0, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R = np.zeros(3, 3)
    R[:, 0] = x_axis
    R[:, 1] = y_axis
    R[:, 2] = z_axis
    return R
