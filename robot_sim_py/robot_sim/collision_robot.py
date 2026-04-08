import numpy as np
import trimesh
import pyvista as pv
import mujoco
import time

from .collision import CollisionDetectionMethod, Collision


class CollisionDetectionConfig:
    def __init__(self):
        # 仅用于自碰撞，是否剔除相邻关节，默认剔除
        self.exclude_adjacent_bodies = True

        # 包含或剔除body，同时作用于自碰撞和避障
        self.enable_include_bodies = False
        self.include_bodies = []
        self.enable_exclude_bodies = False
        self.exclude_bodies = []

        # 包含或剔除body对，仅用于自碰撞
        self.enable_include_body_pairs = False
        self.include_body_pairs = []
        self.enable_exclude_body_pairs = False
        self.exclude_body_pairs = []


class RobotCollisionDetector:
    def __init__(
        self,
        robot_model,
        method: CollisionDetectionMethod,
        config: CollisionDetectionConfig,
    ):
        self.robot_model = robot_model
        self.method = method
        self.config = config
        self.collisions = [
            [] for _ in range(self.robot_model.model.nbody - 1)
        ]  # body 0 (worldbody) is not considered
        self.init_collision()

    def init_collision(self):
        for i in range(self.robot_model.model.ngeom):
            body_id = int(self.robot_model.model.geom_bodyid[i])
            if body_id == 0:
                continue  # body 0 (worldbody) is not considered

            body_name = self.robot_model.model.body(body_id).name
            if self.config.enable_include_bodies:
                if body_name not in self.config.include_bodies:
                    continue
            if self.config.enable_include_body_pairs:
                if not any(
                    body_name in sublist for sublist in self.config.include_body_pairs
                ):
                    continue
            if self.config.enable_exclude_bodies:
                if body_name in self.config.exclude_bodies:
                    continue
            if self.config.enable_exclude_body_pairs:
                pass

            geom_type = self.robot_model.model.geom_type[i]
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = self.robot_model.model.geom_dataid[i]
                vertex_start_id = self.robot_model.model.mesh_vertadr[mesh_id]
                vertex_num = self.robot_model.model.mesh_vertnum[mesh_id]
                face_start_id = self.robot_model.model.mesh_faceadr[mesh_id]
                face_num = self.robot_model.model.mesh_facenum[mesh_id]
                vertex = self.robot_model.model.mesh_vert[
                    vertex_start_id : (vertex_start_id + vertex_num)
                ]
                faces = self.robot_model.model.mesh_face[
                    face_start_id : (face_start_id + face_num)
                ]
                mesh = trimesh.Trimesh(vertex, faces)
            else:
                print(
                    f"Only mesh is supported! Geom id {i} type is not mesh! It will be ignored!"
                )
                mesh = None
            if mesh is not None:
                self.collisions[body_id - 1].append(Collision(i, mesh, self.method))

    def update_collision_pose(self, q):
        self.robot_model.data.qpos = q.copy()
        mujoco.mj_kinematics(self.robot_model.model, self.robot_model.data)
        for i in range(len(self.collisions)):
            for j in range(len(self.collisions[i])):
                geom_id = self.collisions[i][j].id
                p = self.robot_model.data.geom_xpos[geom_id].copy()
                R = self.robot_model.data.geom_xmat[geom_id].reshape((3, 3)).copy()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = p
                self.collisions[i][j].apply_transform(T)

    def self_collision_detection(self, q):
        results = []
        distances = []
        self.update_collision_pose(q)
        num_body = self.robot_model.model.nbody - 1
        for i in range(num_body - 1):
            if len(self.collisions[i]) == 0:
                continue
            for j in range(i + 1, num_body):
                if len(self.collisions[j]) == 0:
                    continue

                body_id1 = i + 1
                body_id2 = j + 1
                body1 = self.robot_model.model.body(body_id1)
                body2 = self.robot_model.model.body(body_id2)
                if self.config.exclude_adjacent_bodies:
                    if body1.parentid == body_id2 or body2.parentid == body_id1:
                        continue

                body_name1 = body1.name
                body_name2 = body2.name
                if self.config.enable_include_body_pairs:
                    if [
                        body_name1,
                        body_name2,
                    ] not in self.config.include_body_pairs and [
                        body_name2,
                        body_name1,
                    ] not in self.config.include_body_pairs:
                        continue
                if self.config.enable_exclude_body_pairs:
                    if [body_name1, body_name2] in self.config.exclude_body_pairs or [
                        body_name2,
                        body_name1,
                    ] in self.config.exclude_body_pairs:
                        continue

                res = dict()
                res["body1"] = body_name1
                res["body2"] = body_name2
                res["min_dis"] = self.body_pair_collision_detection(
                    self.collisions[i], self.collisions[j]
                )
                results.append(res)
                distances.append(res["min_dis"])

        min_dis = np.min(np.array(distances))
        return min_dis, results

    def body_pair_collision_detection(self, collisions1, collisions2):
        distances = []
        for i in range(len(collisions1)):
            for j in range(len(collisions2)):
                dis = collisions1[i].collision_detection(collisions2[j])
                distances.append(dis)
        return np.min(np.array(distances))

    def collision_detection(self, q, objects: list):
        self.update_collision_pose(q)
        num_links = len(self.collisions)
        num_objs = len(objects)
        distances = np.zeros((num_objs, num_links))
        for i in range(num_objs):
            for j in range(num_links):
                if len(self.collisions[j]) == 0:
                    distances[i, j] = 1.0e10
                else:
                    distances[i, j] = self.body_pair_collision_detection(
                        [objects[i]], self.collisions[j]
                    )
        return distances

    def show_collision(self, q, objects=[]):
        plotter = pv.Plotter()
        R0 = np.eye(3)
        p0 = np.zeros(3)
        plotter.add_arrows(p0, R0[:, 0] * 0.1, color="red")
        plotter.add_arrows(p0, R0[:, 1] * 0.1, color="green")
        plotter.add_arrows(p0, R0[:, 2] * 0.1, color="blue")
        self.update_collision_pose(q)

        for i in range(len(self.collisions)):
            for j in range(len(self.collisions[i])):
                self.collisions[i][j].show(plotter)
        for i in range(len(objects)):
            objects[i].show(plotter)

        plotter.show()
