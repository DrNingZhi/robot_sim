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

        self.collision_detect_enable = collision_detect_enable
        if self.collision_detect_enable:
            self.init_collision()
            self.init_sphere_fitting_collision()

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

    def null_space_projection(self, q, body_name, point_on_body=np.zeros(3)):
        if self.dof <= 6:
            print("WARNING: there is no null space for non-redundant manipulator!")
        J = self.jacobian(q, body_name, point_on_body)
        P = np.eye(self.dof) - np.linalg.pinv(J) @ J
        return P

    def init_collision(self):
        self.collisions = [[] for _ in range(self.model.nbody - 1)]
        for i in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[i])
            if body_id == 0:
                continue  # body 0 (worldbody) is not considered

            geom_type = self.model.geom_type[i]
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = self.model.geom_size[i][0]
                mesh = trimesh.primitives.Sphere(r)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r = self.model.geom_size[i][0]
                L = self.model.geom_size[i][1] * 2
                mesh = trimesh.primitives.Capsule(r, L)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r = self.model.geom_size[i][0]
                L = self.model.geom_size[i][1] * 2
                mesh = trimesh.primitives.Cylinder(r, L)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                L = self.model.geom_size[i] * 2
                mesh = trimesh.primitives.Box(L, T)
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = self.model.geom_dataid[i]
                vertex_start_id = self.model.mesh_vertadr[mesh_id]
                vertex_num = self.model.mesh_vertnum[mesh_id]
                face_start_id = self.model.mesh_faceadr[mesh_id]
                face_num = self.model.mesh_facenum[mesh_id]
                vertex = self.model.mesh_vert[
                    vertex_start_id : (vertex_start_id + vertex_num)
                ]
                faces = self.model.mesh_face[face_start_id : (face_start_id + face_num)]
                mesh = trimesh.Trimesh(vertex, faces)
            else:
                print(f"Geom id {i} type is not supported! It will be ignored!")
                mesh = None
            if mesh is not None:
                self.collisions[body_id - 1].append(Collision(i, mesh))

    def update_collision_pose(self, q, level):
        self.data.qpos = q.copy()
        mujoco.mj_kinematics(self.model, self.data)
        for i in range(len(self.collisions)):
            for j in range(len(self.collisions[i])):
                geom_id = self.collisions[i][j].id
                p = self.data.geom_xpos[geom_id].copy()
                R = self.data.geom_xmat[geom_id].reshape((3, 3)).copy()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = p
                self.collisions[i][j].apply_transform(T, level)

    def self_collision_detection(
        self, q, level, exclude_adjacent_links=True, exclude_body_pairs=[]
    ):
        results = []
        distances = []
        self.update_collision_pose(q, level)
        num_body = self.model.nbody - 1
        for i in range(num_body - 1):
            for j in range(i + 1, num_body):
                if exclude_adjacent_links and abs(i - j) == 1:
                    continue

                body_name1 = self.model.body(i + 1).name
                body_name2 = self.model.body(j + 1).name
                if [body_name1, body_name2] in exclude_body_pairs or [
                    body_name2,
                    body_name1,
                ] in exclude_body_pairs:
                    continue

                res = dict()
                res["body1"] = body_name1
                res["body2"] = body_name2
                res["min_dis"] = self.body_pair_collision_detection(
                    self.collisions[i], self.collisions[j], level
                )
                results.append(res)
                distances.append(res["min_dis"])
        min_dis = np.min(np.array(distances))
        return min_dis, results

    def body_pair_collision_detection(self, collisions1, collisions2, level):
        distances = []
        for i in range(len(collisions1)):
            for j in range(len(collisions2)):
                dis = collisions1[i].collision_detection(collisions2[j], level)
                distances.append(dis)
        return np.min(np.array(distances))

    def collision_detection(self, q, objects: list, level):
        self.update_collision_pose(q, level)
        num_links = len(self.collisions)
        num_objs = len(objects)
        distances = np.zeros((num_objs, num_links))
        for i in range(num_objs):
            for j in range(num_links):
                distances[i, j] = self.body_pair_collision_detection(
                    [objects[i]], self.collisions[j], level
                )
        return distances

    def show_collision(self, q, level, objects=[], obj_tforms=[]):
        plotter = pv.Plotter()
        R0 = np.eye(3)
        p0 = np.zeros(3)
        plotter.add_arrows(p0, R0[:, 0] * 0.1, color="red")
        plotter.add_arrows(p0, R0[:, 1] * 0.1, color="green")
        plotter.add_arrows(p0, R0[:, 2] * 0.1, color="blue")

        self.data.qpos = q.copy()
        mujoco.mj_kinematics(self.model, self.data)
        for i in range(len(self.collisions)):
            for j in range(len(self.collisions[i])):
                geom_id = self.collisions[i][j].id
                p = self.data.geom_xpos[geom_id].copy()
                R = self.data.geom_xmat[geom_id].reshape((3, 3)).copy()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = p
                if level == 0:
                    self.collisions[i][j].show(plotter, T, show_bounding_sphere=True)
                elif level == 1:
                    self.collisions[i][j].show(
                        plotter, T, show_bounding_box_oriented=True
                    )
                elif level == 2:
                    self.collisions[i][j].show(plotter, T, show_convex_hull=True)
                elif level == 3:
                    self.collisions[i][j].show(plotter, T)
                else:
                    raise ValueError("Error level! Please give 1~3!")

        for i in range(len(objects)):
            if level == 0:
                objects[i].show(plotter, obj_tforms[i], show_bounding_sphere=True)
            elif level == 1:
                objects[i].show(plotter, obj_tforms[i], show_bounding_box_oriented=True)
            elif level == 2:
                objects[i].show(plotter, obj_tforms[i], show_convex_hull=True)
            elif level == 3:
                objects[i].show(plotter, obj_tforms[i])
            else:
                raise ValueError("Error level! Please give 1~3!")
        plotter.show()

    def init_sphere_fitting_collision(self):
        self.sphere_fitting_collisions = [[] for _ in range(self.model.nbody - 1)]
        for i in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[i])
            if body_id == 0:
                continue  # body 0 (worldbody) is not considered

            geom_type = self.model.geom_type[i]
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = self.model.geom_size[i][0]
                mesh = trimesh.primitives.Sphere(r)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r = self.model.geom_size[i][0]
                L = self.model.geom_size[i][1] * 2
                mesh = trimesh.primitives.Capsule(r, L)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r = self.model.geom_size[i][0]
                L = self.model.geom_size[i][1] * 2
                mesh = trimesh.primitives.Cylinder(r, L)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                L = self.model.geom_size[i] * 2
                mesh = trimesh.primitives.Box(L, T)
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = self.model.geom_dataid[i]
                vertex_start_id = self.model.mesh_vertadr[mesh_id]
                vertex_num = self.model.mesh_vertnum[mesh_id]
                face_start_id = self.model.mesh_faceadr[mesh_id]
                face_num = self.model.mesh_facenum[mesh_id]
                vertex = self.model.mesh_vert[
                    vertex_start_id : (vertex_start_id + vertex_num)
                ]
                faces = self.model.mesh_face[face_start_id : (face_start_id + face_num)]
                mesh = trimesh.Trimesh(vertex, faces)
            else:
                print(f"Geom id {i} type is not supported! It will be ignored!")
                mesh = None
            if mesh is not None:
                sph_fit = SphereFittingCollision(i, mesh)
                sph_fit.cluster()
                self.sphere_fitting_collisions[body_id - 1].append(sph_fit)

    def update_sphere_fitting_collision_pose(self, q):
        self.data.qpos = q.copy()
        mujoco.mj_kinematics(self.model, self.data)
        for i in range(len(self.sphere_fitting_collisions)):
            for j in range(len(self.sphere_fitting_collisions[i])):
                geom_id = self.sphere_fitting_collisions[i][j].id
                p = self.data.geom_xpos[geom_id].copy()
                R = self.data.geom_xmat[geom_id].reshape((3, 3)).copy()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = p
                self.sphere_fitting_collisions[i][j].apply_transform(T)

    def self_sphere_fitting_collision_detection(
        self, q, exclude_adjacent_links=True, exclude_body_pairs=[]
    ):
        results = []
        distances = []
        self.update_sphere_fitting_collision_pose(q)
        num_body = self.model.nbody - 1
        for i in range(num_body - 1):
            for j in range(i + 1, num_body):
                if exclude_adjacent_links and abs(i - j) == 1:
                    continue

                body_name1 = self.model.body(i + 1).name
                body_name2 = self.model.body(j + 1).name
                if [body_name1, body_name2] in exclude_body_pairs or [
                    body_name2,
                    body_name1,
                ] in exclude_body_pairs:
                    continue

                res = dict()
                res["body1"] = body_name1
                res["body2"] = body_name2
                res["min_dis"] = self.body_pair_sphere_fitting_collision_detection(
                    self.sphere_fitting_collisions[i], self.sphere_fitting_collisions[j]
                )
                results.append(res)
                distances.append(res["min_dis"])
        min_dis = np.min(np.array(distances))
        return min_dis, results

    def body_pair_sphere_fitting_collision_detection(self, collisions1, collisions2):
        distances = []
        for i in range(len(collisions1)):
            for j in range(len(collisions2)):
                dis = collisions1[i].collision_detection(collisions2[j])
                distances.append(dis)
        return np.min(np.array(distances))

    def sphere_fitting_collision_detection(self, q, objects: list):
        self.update_sphere_fitting_collision_pose(q)
        num_links = len(self.sphere_fitting_collisions)
        num_objs = len(objects)
        distances = np.zeros((num_objs, num_links))
        for i in range(num_objs):
            for j in range(num_links):
                distances[i, j] = self.body_pair_sphere_fitting_collision_detection(
                    [objects[i]], self.sphere_fitting_collisions[j]
                )
        return distances

    def show_sphere_fitting_collision(self, q, objects=[]):
        plotter = pv.Plotter()
        R0 = np.eye(3)
        p0 = np.zeros(3)
        plotter.add_arrows(p0, R0[:, 0] * 0.1, color="red")
        plotter.add_arrows(p0, R0[:, 1] * 0.1, color="green")
        plotter.add_arrows(p0, R0[:, 2] * 0.1, color="blue")

        self.update_sphere_fitting_collision_pose(q)
        for i in range(len(self.sphere_fitting_collisions)):
            for j in range(len(self.sphere_fitting_collisions[i])):
                self.sphere_fitting_collisions[i][j].show_spheres(plotter)

        for i in range(len(objects)):
            objects[i].show_spheres(plotter)
        plotter.show()
