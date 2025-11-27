import numpy as np
import trimesh
import pyvista as pv
import copy


class Collision:
    def __init__(self, id, arg1, arg2=None):
        """
        Define a triangle mesh for collision detection.

        Parameters:
            option1:
                arg1: trimesh.primitives.Trimesh
                arg2: None
            option2:
                arg1: vertices: np.ndarray(n,3)
                arg2: faces: np.ndarray(n,3)
        """
        self.id = id
        if arg2 is not None:
            self.vertices = arg1
            self.faces = arg2
            self.trimesh = trimesh.Trimesh(self.vertices, self.faces)
        else:
            self.trimesh = arg1
            self.vertices = self.trimesh.vertices
            self.faces = self.trimesh.faces

        # if not self.trimesh.is_watertight:
        #     trimesh.repair.fill_holes(self.trimesh)
        #     trimesh.repair.broken_faces(self.trimesh)
        #     trimesh.repair.fix_inversion(self.trimesh)
        #     trimesh.repair.fix_normals(self.trimesh)
        #     trimesh.repair.fix_winding(self.trimesh)
        #     print("mesh is not watertight! Repair it!")

        # if not self.trimesh.is_watertight:
        #     raise ValueError("mesh is still not watertight!")

        bounding_sphere = self.trimesh.bounding_sphere
        center = np.mean(bounding_sphere.vertices, axis=0)
        radius = np.mean(np.linalg.norm(bounding_sphere.vertices - center, axis=1))
        self.bounding_sphere = np.hstack((center, radius))

        self.obb = self.trimesh.bounding_box_oriented
        self.convex_hull = self.trimesh.convex_hull

    def apply_transform(self, tform, level):
        if level == 0:
            self.bounding_sphere_t = self.bounding_sphere.copy()
            self.bounding_sphere_t[:3] += tform[:3, 3]
        elif level == 1:
            self.obb_t = self.obb.copy().apply_transform(tform)
        elif level == 2:
            self.convex_hull_t = self.convex_hull.copy().apply_transform(tform)
        elif level == 3:
            self.trimesh_t = self.trimesh.copy().apply_transform(tform)
        else:
            raise ValueError("Error level! Please give 0~3!")

    def show(
        self,
        plotter,
        transform=np.eye(4),
        show_ori_mesh=True,
        show_convex_hull=False,
        show_bounding_box=False,
        show_bounding_box_oriented=False,
        show_bounding_sphere=False,
    ):
        self.apply_transform(transform, 3)
        if show_ori_mesh:
            pv_mesh = self.trimesh_to_pv(self.trimesh_t)
            plotter.add_mesh(pv_mesh, show_edges=True)
        if show_convex_hull:
            convex_mesh = self.trimesh_to_pv(self.trimesh_t.convex_hull)
            plotter.add_mesh(convex_mesh, show_edges=True, opacity=0.5)
        if show_bounding_box:
            bounding_box = self.trimesh_to_pv(self.trimesh_t.bounding_box)
            plotter.add_mesh(bounding_box, show_edges=True, opacity=0.5)
        if show_bounding_box_oriented:
            bounding_box_oriented = self.trimesh_to_pv(
                self.trimesh_t.bounding_box_oriented
            )
            plotter.add_mesh(bounding_box_oriented, show_edges=True, opacity=0.5)
        if show_bounding_sphere:
            bounding_sphere = self.trimesh_to_pv(self.trimesh_t.bounding_sphere)
            plotter.add_mesh(bounding_sphere, show_edges=False, opacity=0.5)

    @staticmethod
    def trimesh_to_pv(mesh):
        vertices = mesh.vertices
        faces = mesh.faces
        pv_faces = np.hstack((np.ones((len(faces), 1), dtype=int) * 3, faces))
        pv_mesh = pv.PolyData(vertices, pv_faces)
        return pv_mesh

    def collision_detection(self, mesh, level):
        """
        collision detection between self and another Collision object
        level = 1: use obb
        level = 2: use convex_hull
        level = 3: use complete mesh
        """
        if level == 0:
            cdis = np.linalg.norm(
                self.bounding_sphere_t[:3] - mesh.bounding_sphere_t[:3]
            )
            dis = cdis - self.bounding_sphere_t[3] - mesh.bounding_sphere_t[3]
            return dis
        if level == 1:
            collision_manager1 = trimesh.collision.CollisionManager()
            collision_manager1.add_object("mesh1", self.obb_t)
            collision_manager2 = trimesh.collision.CollisionManager()
            collision_manager2.add_object("mesh2", mesh.obb_t)
            dis = collision_manager1.min_distance_other(collision_manager2)
            return dis
        elif level == 2:
            collision_manager1 = trimesh.collision.CollisionManager()
            collision_manager1.add_object("mesh1", self.convex_hull_t)
            collision_manager2 = trimesh.collision.CollisionManager()
            collision_manager2.add_object("mesh2", mesh.convex_hull_t)
            dis = collision_manager1.min_distance_other(collision_manager2)
            return dis
        elif level == 3:
            collision_manager1 = trimesh.collision.CollisionManager()
            collision_manager1.add_object("mesh1", self.trimesh_t)
            collision_manager2 = trimesh.collision.CollisionManager()
            collision_manager2.add_object("mesh2", mesh.trimesh_t)
            dis = collision_manager1.min_distance_other(collision_manager2)
            return dis
        else:
            raise ValueError("Error level! Please give 1~3!")
