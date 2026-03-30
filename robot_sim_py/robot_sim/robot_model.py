import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import trimesh
import copy
import pyvista as pv
from .collision import Collision, SphereFittingCollision


class RobotModel:
    def __init__(self, model: str, collision_detect_enable=False):
        self.model = mujoco.MjModel.from_xml_path(model)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.dof = self.model.nv
        self.joint_lower_limits = self.model.jnt_range[:, 0]
        self.joint_upper_limits = self.model.jnt_range[:, 1]
        self.joint_ranges = self.joint_upper_limits - self.joint_lower_limits
        self.joint_mid = (self.joint_lower_limits + self.joint_upper_limits) / 2
        # self.collision_detector = None

    def rand_configuration(self):
        q = np.random.rand(self.dof) * self.joint_ranges + self.joint_lower_limits
        return q

    def forward_kinematics(self, q, body_name, point_at_body=np.zeros(3)):
        self.data.qpos = q.copy()
        mujoco.mj_kinematics(self.model, self.data)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        p = self.data.xpos[body_id].copy()
        R = self.data.xmat[body_id].reshape((3, 3)).copy()
        p = p + R @ point_at_body
        return [p, R]

    def inverse_kinematics(
        self, target, body_name, q_init=None, point_at_body=np.zeros(3), smooth=True
    ):
        if q_init is None:
            q_init = np.zeros(self.dof)

        res = minimize(
            self.ik_cost,
            q_init,
            args=(target, body_name, q_init, point_at_body, smooth),
        )
        q = res.x

        err = self.ik_cost(q, target, body_name, q_init, point_at_body, False)
        if err > 1e-3:
            print("Warning: IK err is larger, " + str(err))
        return q

    def ik_cost(self, q, target, body_name, q_init, point_at_body, smooth):
        p, R = self.forward_kinematics(q, body_name, point_at_body)
        p_tar = target[:3, 3]
        R_tar = target[:3, :3]
        cost1 = np.linalg.norm(p - p_tar) ** 2
        cost2 = np.linalg.norm(Rotation.from_matrix(R @ R_tar.T).as_rotvec()) ** 2

        # convert joint limits to cost
        q_normalize = (q - self.joint_lower_limits) / self.joint_ranges * 2.0 - 1.0
        y1 = q_normalize - 0.8
        y2 = -0.8 - q_normalize
        y1[y1 < 0] = 0.0
        y2[y2 < 0] = 0.0
        y = (1.0 / 0.2**12) * (y1**12 + y2**12)
        cost3 = np.sum(y)

        if not smooth:
            return 1000000.0 * cost1 + 3282.8 * cost2 + 1.0 * cost3

        cost3 = np.linalg.norm(q - q_init) ** 2 / self.dof
        return 1000000.0 * cost1 + 3282.8 * cost2 + 32.8 * cost3

    def jacobian(self, q, body_name, point_on_body=np.zeros(3)):
        point, _ = self.forward_kinematics(q, body_name, point_on_body)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        self.data.qpos = q.copy()
        mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.dof))
        jacr = np.zeros((3, self.dof))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, body_id)
        Jac = np.vstack((jacp, jacr))
        return Jac

    def jacobian_d(self, q, qd, body_name, point_on_body=np.zeros(3)):
        point, _ = self.forward_kinematics(q, body_name, point_on_body)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        self.data.qpos = q.copy()
        self.data.qpos = qd.copy()
        mujoco.mj_forward(self.model, self.data)

        jacp_d = np.zeros((3, self.dof))
        jacr_d = np.zeros((3, self.dof))
        mujoco.mj_jacDot(self.model, self.data, jacp_d, jacr_d, point, body_id)
        Jac_d = np.vstack((jacp_d, jacr_d))
        return Jac_d

    def inverse_kinematics_d(self, q, vel, acc, body_name, point_on_body=np.zeros(3)):
        Jac = self.jacobian(q, body_name, point_on_body)
        qd = np.linalg.pinv(Jac) @ vel
        Jac_d = self.jacobian_d(q, qd, body_name, point_on_body)
        qdd = np.linalg.pinv(Jac) @ (acc - Jac_d @ qd)
        return qd, qdd

    def inverse_dynamics(self, q, qd, qdd):
        self.data.qpos = q.copy()
        self.data.qvel = qd.copy()
        self.data.qacc = qdd.copy()
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse.copy()
        return tau

    def forward_dynamics(self, q, qd, tau):
        self.data.qpos = q.copy()
        self.data.qvel = qd.copy()
        self.data.qacc = np.zeros(self.dof)
        mujoco.mj_step(self.model, self.data)
        acc = self.data.qacc.copy()
        return acc

    def gravity_torque(self, q):
        self.data.qpos = q.copy()
        self.data.qvel = np.zeros(self.dof)
        self.data.qacc = np.zeros(self.dof)
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse.copy()
        return tau

    def inertia_matrix(self, q):
        self.data.qpos = q.copy()
        mujoco.mj_step(self.model, self.data)
        M_matrix = np.zeros((self.dof, self.dof))
        mujoco.mj_fullM(self.model, M_matrix, self.data.qM)
        return M_matrix

    def null_space_projection(self, q, body_name, point_on_body=np.zeros(3)):
        if self.dof <= 6:
            print("WARNING: there is no null space for non-redundant manipulator!")
        J = self.jacobian(q, body_name, point_on_body)
        P = np.eye(self.dof) - np.linalg.pinv(J) @ J
        return P

    def fast_ik(self, target, body_name, q_init, point_at_body=np.zeros(3)):
        pos_err_threshold = 1e-6
        rot_err_threshold = 0.017453e-3
        max_iter = 50
        lamb = 0.7

        p_tar = target[:3, 3].copy()
        R_tar = target[:3, :3].copy()

        q = q_init.copy()
        iter = 0
        while True:
            p, R = self.forward_kinematics(q, body_name, point_at_body)

            del_p = p_tar - p
            del_R = R.T @ R_tar
            del_w = R @ Rotation.from_matrix(del_R).as_rotvec()

            pos_err = np.linalg.norm(del_p)
            rot_err = np.linalg.norm(del_w)

            # print(f"iter: {iter}, pos_err: {pos_err:.6f}, rot_err: {rot_err:.6f}")

            if (
                pos_err < pos_err_threshold and rot_err < rot_err_threshold
            ) or iter >= max_iter:
                break

            del_ee = np.concatenate((del_p, del_w))

            J = self.jacobian(q, body_name, point_at_body)

            J_pinv = np.linalg.inv(J.T @ J + 0.001 * np.eye(self.dof)) @ J.T
            # J_pinv = np.linalg.pinv(J)

            del_q = J_pinv @ del_ee
            q += lamb * del_q
            q = np.clip(q, self.joint_lower_limits, self.joint_upper_limits)

            iter += 1

        return q

    def trajectory_ik(
        self,
        pos,  # num_step x 3
        vel,  # num_step x 3
        acc,  # num_step x 3
        quat,  # num_step x 4 (quat, wxyz)
        ang_vel,  # num_step x 3
        ang_acc,  # num_step x 3
        body_name,
        q_init=None,
        point_at_body=np.zeros(3),
        use_smooth=False,
    ):
        if q_init is None:
            q_init = self.joint_mid

        num_steps = len(pos)
        q = np.zeros((num_steps, self.dof))
        qd = np.zeros((num_steps, self.dof))
        qdd = np.zeros((num_steps, self.dof))
        for i in range(num_steps):
            target = np.eye(4)
            target[:3, 3] = pos[i]
            target[:3, :3] = Rotation.from_quat(quat[i], scalar_first=True).as_matrix()
            q[i] = self.fast_ik(target, body_name, q_init, point_at_body)
            if use_smooth:
                q_init = q[i]

            vel_a = np.hstack((vel[i], ang_vel[i]))
            acc_a = np.hstack((acc[i], ang_acc[i]))
            qd[i], qdd[i] = self.inverse_kinematics_d(
                q[i], vel_a, acc_a, body_name, point_at_body
            )
            # ratio = 100.0 * (i + 1) / num_steps
            # print(f"轨迹生成完成{ratio:.2f}%")
        return q, qd, qdd
