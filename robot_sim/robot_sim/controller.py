import numpy as np
from scipy.spatial.transform import Rotation


class PDController:
    def __init__(self, kp, kd, limit=None):
        self.kp = kp
        self.kd = kd
        self.limit = limit

    def update(self, x, xd, x_ref, xd_ref=0.0):
        output = self.kp * (x_ref - x) + self.kd * (xd_ref - xd)
        if self.limit is not None:
            output = np.clip(output, -self.limit, self.limit)
        return output


class RobotPDController:
    def __init__(self, kp, kd, limit=None):
        self.dof = len(kp)
        self.joint_pd_controllers = []
        for i in range(self.dof):
            if limit is not None:
                pd_controller = PDController(kp[i], kd[i], limit[i])
            else:
                pd_controller = PDController(kp[i], kd[i])
            self.joint_pd_controllers.append(pd_controller)

    def update(self, q, qd, q_ref, qd_ref):
        tau = np.zeros(self.dof)
        for i in range(self.dof):
            tau[i] = self.joint_pd_controllers[i].update(
                q[i], qd[i], q_ref[i], qd_ref[i]
            )
        return tau

    def update_with_feedforward(self, q, qd, q_ref, qd_ref, tau_ref):
        tau = np.zeros(self.dof)
        for i in range(self.dof):
            tau[i] = self.joint_pd_controllers[i].update(
                q[i], qd[i], q_ref[i], qd_ref[i]
            )
        tau = tau + tau_ref
        return tau


class RobotJacInvController:
    def __init__(self, kp, kd, robot_model, ee_body_name, ee_point_on_body):
        self.kp = kp
        self.kd = kd
        self.robot_model = robot_model
        self.ee_body_name = ee_body_name
        self.ee_point_on_body = ee_point_on_body

    def update(self, q, qd, pos_ref, rotmat_ref, vel_ref, angvel_ref):
        pos, rotmat = self.robot_model.forward_kinematics(
            q, self.ee_body_name, self.ee_point_on_body
        )
        jac = self.robot_model.jacobian(q, self.ee_body_name, self.ee_point_on_body)

        p = pos_ref - pos
        R = rotmat.T @ rotmat_ref
        w = rotmat @ Rotation.from_matrix(R).as_rotvec()
        pos_err = np.hstack((p, w))
        q_err = np.linalg.pinv(jac) @ pos_err

        ee_vel_ref = np.hstack((vel_ref, angvel_ref))
        vel_err = ee_vel_ref - jac @ qd
        qd_err = np.linalg.pinv(jac) @ vel_err
        tau = self.kp * q_err + self.kd * qd_err
        return tau


class RobotJacTController:
    def __init__(self, kp, kd, robot_model, ee_body_name, ee_point_on_body):
        self.kp = kp
        self.kd = kd
        self.robot_model = robot_model
        self.ee_body_name = ee_body_name
        self.ee_point_on_body = ee_point_on_body

    def update(self, q, qd, pos_ref, rotmat_ref, vel_ref, angvel_ref):
        pos, rotmat = self.robot_model.forward_kinematics(
            q, self.ee_body_name, self.ee_point_on_body
        )
        jac = self.robot_model.jacobian(q, self.ee_body_name, self.ee_point_on_body)

        p = pos_ref - pos
        R = rotmat.T @ rotmat_ref
        w = rotmat @ Rotation.from_matrix(R).as_rotvec()
        pos_err = np.hstack((p, w))

        ee_vel_ref = np.hstack((vel_ref, angvel_ref))
        vel_err = ee_vel_ref - jac @ qd

        tau = jac.T @ (self.kp * pos_err + self.kd * vel_err)
        return tau

    def update_with_gravity_compensation(
        self, q, qd, pos_ref, rotmat_ref, vel_ref, angvel_ref
    ):
        # tau_ref = self.robot_model.gravity_torque(q)
        tau_ref = self.robot_model.inverse_dynamics(q, qd, np.zeros(6))
        pos, rotmat = self.robot_model.forward_kinematics(
            q, self.ee_body_name, self.ee_point_on_body
        )
        jac = self.robot_model.jacobian(q, self.ee_body_name, self.ee_point_on_body)

        p = pos_ref - pos
        R = rotmat.T @ rotmat_ref
        w = rotmat @ Rotation.from_matrix(R).as_rotvec()
        pos_err = np.hstack((p, w))

        ee_vel_ref = np.hstack((vel_ref, angvel_ref))
        vel_err = ee_vel_ref - jac @ qd

        tau = jac.T @ (self.kp * pos_err + self.kd * vel_err)
        return tau + tau_ref

    def update_with_feedforward(
        self, q, qd, pos_ref, rotmat_ref, vel_ref, angvel_ref, acc_ref, angacc_ref
    ):
        M = self.robot_model.inertia_matrix(q)
        jac = self.robot_model.jacobian(q, self.ee_body_name, self.ee_point_on_body)
        jac_inv = np.linalg.pinv(jac)
        jac_d = self.robot_model.jacobian_d(
            q, qd, self.ee_body_name, self.ee_point_on_body
        )
        jac_T_inv = np.linalg.pinv(jac.T)
        tau_compensation = self.robot_model.inverse_dynamics(
            q, qd, np.zeros(self.robot_model.dof)
        )
        pos, rotmat = self.robot_model.forward_kinematics(
            q, self.ee_body_name, self.ee_point_on_body
        )

        p = pos_ref - pos
        R = rotmat.T @ rotmat_ref
        w = rotmat @ Rotation.from_matrix(R).as_rotvec()
        pos_err = np.hstack((p, w))

        ee_vel_ref = np.hstack((vel_ref, angvel_ref))
        vel_err = ee_vel_ref - jac @ qd

        acc_d = np.hstack((acc_ref, angacc_ref))
        xdd_correction = self.kp * pos_err + self.kd * vel_err
        tau = (
            M @ jac_inv @ (acc_d + xdd_correction)
            - M @ jac_inv @ jac_d @ qd
            + tau_compensation
        )
        return tau


class RobotForceController:
    def __init__(self, kp, ki, robot_model, ee_body_name, ee_point_on_body):
        self.kp = kp
        self.ki = ki
        self.robot_model = robot_model
        self.ee_body_name = ee_body_name
        self.ee_point_on_body = ee_point_on_body
        self.F_err_int = np.zeros(6)

    def update(self, q, qd, F_d, F):
        tau_grav = self.robot_model.gravity_torque(q)
        jac = self.robot_model.jacobian(q, self.ee_body_name, self.ee_point_on_body)
        self.F_err_int += F_d - F
        tau = tau_grav + jac.T @ (self.kp * (F_d - F) + self.ki * self.F_err_int)
        return tau


class RobotForcePositionController:
    def __init__(
        self,
        kp_pos,
        kd_pos,
        kp_frc,
        ki_frc,
        robot_model,
        ee_body_name,
        ee_point_on_body,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_frc = kp_frc
        self.ki_frc = ki_frc
        self.robot_model = robot_model
        self.ee_body_name = ee_body_name
        self.ee_point_on_body = ee_point_on_body
        self.F_err_int = np.zeros(6)

    def update(self, q, qd, pos_ref, rotmat_ref, vel_ref, angvel_ref, F, F_d, S):
        tau_grav = self.robot_model.gravity_torque(q)
        jac = self.robot_model.jacobian(q, self.ee_body_name, self.ee_point_on_body)
        self.F_err_int += F_d - F
        tau = tau_grav + jac.T @ (
            (np.eye(6) - S) @ (self.kp_frc * (F_d - F) + self.ki_frc * self.F_err_int)
        )

        pos, rotmat = self.robot_model.forward_kinematics(
            q, self.ee_body_name, self.ee_point_on_body
        )
        p = pos_ref - pos
        R = rotmat.T @ rotmat_ref
        w = rotmat @ Rotation.from_matrix(R).as_rotvec()
        pos_err = np.hstack((p, w))

        ee_vel_ref = np.hstack((vel_ref, angvel_ref))
        vel_err = ee_vel_ref - jac @ qd

        tau += jac.T @ (S @ (self.kp_pos * pos_err + self.kd_pos * vel_err))
        return tau
