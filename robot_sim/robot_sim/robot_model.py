import numpy as np
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


class RobotModel:
    def __init__(self, model: str):
        self.model = mujoco.MjModel.from_xml_path(model)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.dof = self.model.nv
        self.joint_lower_limits = self.model.jnt_range[:, 0]
        self.joint_upper_limits = self.model.jnt_range[:, 1]

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
            bounds=self.model.jnt_range,
            args=(target, body_name, q_init, point_at_body, smooth),
        )
        q = res.x

        err = self.ik_cost(q, target, body_name, q_init, point_at_body, smooth)
        if err > 1e-3:
            print("Warning: IK err is larger, " + str(err))
        return q

    def ik_cost(self, q, target, body_name, q_init, point_at_body, smooth):
        p, R = self.forward_kinematics(q, body_name, point_at_body)
        p_tar = target[:3, 3]
        R_tar = target[:3, :3]
        cost1 = np.linalg.norm(p - p_tar) ** 2
        cost2 = np.linalg.norm(Rotation.from_matrix(R @ R_tar.T).as_rotvec()) ** 2
        if not smooth:
            return 1000000.0 * cost1 + 3282.8 * cost2

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

    def gravity_torque(self, q):
        self.data.qpos = q.copy()
        self.data.qvel = np.zeros(self.dof)
        self.data.qacc = np.zeros(self.dof)
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse
        return tau

    def inverse_dynamics(self, q, qd, qdd):
        self.data.qpos = q.copy()
        self.data.qvel = qd.copy()
        self.data.qacc = qdd.copy()
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse
        return tau

    def inertia_matrix(self, q):
        self.data.qpos = q.copy()
        mujoco.mj_step(self.model, self.data)
        M_matrix = np.zeros((self.dof, self.dof))
        mujoco.mj_fullM(self.model, M_matrix, self.data.qM)
        return M_matrix
