import numpy as np
import trimesh
import pyvista as pv
import copy
import time
from enum import Enum
from scipy.spatial.transform import Rotation

from .collision_utils import CollisionGenerator, CollisionVisualizer, CollisionDetector
from .transform_utils import rotation_matrix_from_z_axis


class CollisionDetectionMethod(Enum):
    BoundingSphere = 0
    BoundingBox = 1
    BoundingCylinder = 2
    OrientedBoundingBox = 3

    SphereByBoundingCylinder = 4
    SphereByBoundingBox = 5
    SphereByConvexHull = 6
    SphereByOriginalMesh = 7

    ConvexHull = 8
    OriginalMesh = 9


class Collision:
    def __init__(
        self,
        id,
        mesh,
        method: CollisionDetectionMethod,
    ):
        """
        Define a triangle mesh for collision detection.

        Parameters:
            id: id
            mesh: trimesh.primitives.Trimesh
            method: CollisionDetectionMethod
        """
        self.id = id
        self.trimesh = mesh
        self.vertices = self.trimesh.vertices
        self.faces = self.trimesh.faces
        self.method = method

        # if not self.trimesh.is_watertight:
        #     trimesh.repair.fill_holes(self.trimesh)
        #     trimesh.repair.broken_faces(self.trimesh)
        #     trimesh.repair.fix_inversion(self.trimesh)
        #     trimesh.repair.fix_normals(self.trimesh)
        #     trimesh.repair.fix_winding(self.trimesh)
        #     print("mesh is not watertight! Repair it!")

        # if not self.trimesh.is_watertight:
        #     raise ValueError("mesh is still not watertight!")

        # 原始碰撞检测模型
        self.bounding_sphere = None  # [球心，半径]
        self.bounding_box = None  # (2,3)，[min, max]坐标
        self.bounding_cylinder = None  # [起点，终点，半径]
        self.oriented_bounding_box = None  # [中心， 半边长， 旋转矩阵]
        self.spheres = None  # [球心，半径]
        self.convex_hull = None  # Trimesh
        self.original_mesh = self.trimesh

        # 施加空间运动后的碰撞检测模型（最终用于显示和检测）
        self.bounding_sphere_t = None  # [球心，半径]
        self.bounding_box_t = None  # (2,3)，[min, max]坐标
        self.bounding_cylinder_t = None  # [起点，终点，半径]
        self.oriented_bounding_box_t = None  # [中心， 半边长， 旋转矩阵]
        self.spheres_t = None  # [球心，半径]
        self.convex_hull_t = None  # Trimesh
        self.original_mesh_t = self.trimesh.copy()

        self.initialize()

    def initialize(self):
        match self.method:
            case CollisionDetectionMethod.BoundingSphere:
                self.bounding_sphere = CollisionGenerator.get_bounding_sphere(
                    self.trimesh
                )
                self.bounding_sphere_t = copy.deepcopy(self.bounding_sphere)
                return
            case CollisionDetectionMethod.BoundingBox:
                self.bounding_box = self.trimesh.bounds.copy()
                self.bounding_box_t = copy.deepcopy(self.bounding_box)
                return
            case CollisionDetectionMethod.BoundingCylinder:
                self.bounding_cylinder = CollisionGenerator.get_bounding_cylinder(
                    self.trimesh
                )
                self.bounding_cylinder_t = copy.deepcopy(self.bounding_cylinder)
                return
            case CollisionDetectionMethod.OrientedBoundingBox:
                self.oriented_bounding_box = (
                    CollisionGenerator.get_oriented_bounding_box(self.trimesh)
                )
                self.oriented_bounding_box_t = copy.deepcopy(self.oriented_bounding_box)
                return
            case CollisionDetectionMethod.SphereByBoundingCylinder:
                self.bounding_cylinder = CollisionGenerator.get_bounding_cylinder(
                    self.trimesh
                )
                self.bounding_cylinder_t = copy.deepcopy(self.bounding_cylinder)
                self.spheres = CollisionGenerator.get_spheres_from_cylinder(
                    self.bounding_cylinder
                )
                self.spheres_t = copy.deepcopy(self.spheres)
                return
            case CollisionDetectionMethod.SphereByBoundingBox:
                self.oriented_bounding_box = (
                    CollisionGenerator.get_oriented_bounding_box(self.trimesh)
                )
                self.oriented_bounding_box_t = copy.deepcopy(self.oriented_bounding_box)
                self.spheres = CollisionGenerator.get_spheres_from_obb(
                    self.oriented_bounding_box
                )
                self.spheres_t = copy.deepcopy(self.spheres)
                return
            case CollisionDetectionMethod.SphereByConvexHull:
                self.convex_hull = self.trimesh.convex_hull
                self.convex_hull_t = self.convex_hull.copy()
                self.spheres = CollisionGenerator.get_spheres_from_mesh(
                    self.convex_hull
                )
                self.spheres_t = copy.deepcopy(self.spheres)
                return
            case CollisionDetectionMethod.SphereByOriginalMesh:
                self.spheres = CollisionGenerator.get_spheres_from_mesh(self.trimesh)
                self.spheres_t = copy.deepcopy(self.spheres)
                return
            case CollisionDetectionMethod.ConvexHull:
                self.convex_hull = self.trimesh.convex_hull
                self.convex_hull_t = self.convex_hull.copy()
                return
            case CollisionDetectionMethod.OriginalMesh:
                return

    def apply_transform(self, tform):
        translation = tform[:3, 3]
        rotation = tform[:3, :3]
        self.original_mesh_t = self.original_mesh.copy().apply_transform(tform)
        match self.method:
            case CollisionDetectionMethod.BoundingSphere:
                self.bounding_sphere_t = copy.deepcopy(self.bounding_sphere)
                self.bounding_sphere_t[0] = (
                    translation + rotation @ self.bounding_sphere_t[0]
                )
                return
            case CollisionDetectionMethod.BoundingBox:
                self.bounding_box_t = self.original_mesh_t.bounds.copy()
                return
            case CollisionDetectionMethod.BoundingCylinder:
                self.bounding_cylinder_t = copy.deepcopy(self.bounding_cylinder)
                self.bounding_cylinder_t[0] = (
                    translation + rotation @ self.bounding_cylinder_t[0]
                )
                self.bounding_cylinder_t[1] = (
                    translation + rotation @ self.bounding_cylinder_t[1]
                )
                return
            case CollisionDetectionMethod.OrientedBoundingBox:
                self.oriented_bounding_box_t = copy.deepcopy(self.oriented_bounding_box)
                self.oriented_bounding_box_t[0] = (
                    translation + rotation @ self.oriented_bounding_box_t[0]
                )
                self.oriented_bounding_box_t[2] = (
                    rotation @ self.oriented_bounding_box_t[2]
                )
                return
            case CollisionDetectionMethod.SphereByBoundingCylinder:
                self.bounding_cylinder_t = copy.deepcopy(self.bounding_cylinder)
                self.bounding_cylinder_t[0] = (
                    translation + rotation @ self.bounding_cylinder_t[0]
                )
                self.bounding_cylinder_t[1] = (
                    translation + rotation @ self.bounding_cylinder_t[1]
                )
                self.spheres_t = copy.deepcopy(self.spheres)
                self.spheres_t[0] = translation + (rotation @ self.spheres_t[0].T).T
                return
            case CollisionDetectionMethod.SphereByBoundingBox:
                self.oriented_bounding_box_t = copy.deepcopy(self.oriented_bounding_box)
                self.oriented_bounding_box_t[0] = (
                    translation + rotation @ self.oriented_bounding_box_t[0]
                )
                self.oriented_bounding_box_t[2] = (
                    rotation @ self.oriented_bounding_box_t[2]
                )
                self.spheres_t = copy.deepcopy(self.spheres)
                self.spheres_t[0] = translation + (rotation @ self.spheres_t[0].T).T
                return
            case CollisionDetectionMethod.SphereByConvexHull:
                self.convex_hull_t = self.convex_hull.copy().apply_transform(tform)
                self.spheres_t = copy.deepcopy(self.spheres)
                self.spheres_t[0] = translation + (rotation @ self.spheres_t[0].T).T
                return
            case CollisionDetectionMethod.SphereByOriginalMesh:
                self.spheres_t = copy.deepcopy(self.spheres)
                self.spheres_t[0] = translation + (rotation @ self.spheres_t[0].T).T
                return
            case CollisionDetectionMethod.ConvexHull:
                self.convex_hull_t = self.convex_hull.copy().apply_transform(tform)
                return
            case CollisionDetectionMethod.OriginalMesh:
                return

    def show(self, plotter):
        plotter.add_mesh(
            CollisionVisualizer.trimesh_to_pv(self.original_mesh_t),
            show_edges=True,
            opacity=0.8,
        )
        match self.method:
            case CollisionDetectionMethod.BoundingSphere:
                CollisionVisualizer.add_sphere(plotter, self.bounding_sphere_t, 0.2)
                return
            case CollisionDetectionMethod.BoundingBox:
                plotter.add_mesh(
                    pv.Box(bounds=self.bounding_box_t.flatten(order="F")),
                    opacity=0.2,
                )
                return
            case CollisionDetectionMethod.BoundingCylinder:
                CollisionVisualizer.add_cylinder(plotter, self.bounding_cylinder_t, 0.2)
                return
            case CollisionDetectionMethod.OrientedBoundingBox:
                CollisionVisualizer.add_obb(plotter, self.oriented_bounding_box_t, 0.2)
                return
            case CollisionDetectionMethod.SphereByBoundingCylinder:
                CollisionVisualizer.add_cylinder(plotter, self.bounding_cylinder_t, 0.2)
                CollisionVisualizer.add_spheres(plotter, self.spheres_t, 0.5, True)
                return
            case CollisionDetectionMethod.SphereByBoundingBox:
                CollisionVisualizer.add_obb(plotter, self.oriented_bounding_box_t, 0.2)
                CollisionVisualizer.add_spheres(plotter, self.spheres_t, 0.5, True)
                return
            case CollisionDetectionMethod.SphereByConvexHull:
                plotter.add_mesh(
                    CollisionVisualizer.trimesh_to_pv(self.convex_hull_t),
                    opacity=0.5,
                    show_edges=True,
                )
                CollisionVisualizer.add_spheres(plotter, self.spheres_t, 0.5, True)
                return
            case CollisionDetectionMethod.SphereByOriginalMesh:
                CollisionVisualizer.add_spheres(plotter, self.spheres_t, 0.5, True)
                return
            case CollisionDetectionMethod.ConvexHull:
                plotter.add_mesh(
                    CollisionVisualizer.trimesh_to_pv(self.convex_hull_t),
                    opacity=0.5,
                    show_edges=True,
                )
                return
            case CollisionDetectionMethod.OriginalMesh:
                return

    def collision_detection(self, collision):
        if not self.method == collision.method:
            raise TypeError(
                "Collision detection methods are different. It is not supported!"
            )

        if self.method == CollisionDetectionMethod.BoundingSphere:
            dis = CollisionDetector.sphere_sphere(
                self.bounding_sphere_t, collision.bounding_sphere_t
            )
        elif self.method == CollisionDetectionMethod.BoundingBox:
            dis = CollisionDetector.box_box(
                self.bounding_box_t, collision.bounding_box_t
            )
        elif self.method == CollisionDetectionMethod.BoundingCylinder:
            dis = CollisionDetector.cylinder_cylinder(
                self.bounding_cylinder_t, collision.bounding_cylinder_t
            )
        elif self.method == CollisionDetectionMethod.OrientedBoundingBox:
            dis = CollisionDetector.obb_obb(
                self.oriented_bounding_box_t, collision.oriented_bounding_box_t
            )
        elif self.method == CollisionDetectionMethod.ConvexHull:
            dis = CollisionDetector.mesh_mesh(
                self.convex_hull_t, collision.convex_hull_t
            )
        elif self.method == CollisionDetectionMethod.OriginalMesh:
            dis = CollisionDetector.mesh_mesh(
                self.original_mesh_t, collision.original_mesh_t
            )
        else:  # all spheres cases
            dis = CollisionDetector.spheres_spheres(self.spheres_t, collision.spheres_t)
        return dis
