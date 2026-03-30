import numpy as np
import trimesh
import pyvista as pv
import copy
import time


def trimesh_to_pv(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    pv_faces = np.hstack((np.ones((len(faces), 1), dtype=int) * 3, faces))
    pv_mesh = pv.PolyData(vertices, pv_faces)
    return pv_mesh


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
            pv_mesh = trimesh_to_pv(self.trimesh_t)
            plotter.add_mesh(pv_mesh, show_edges=True)
        if show_convex_hull:
            convex_mesh = trimesh_to_pv(self.trimesh_t.convex_hull)
            plotter.add_mesh(convex_mesh, show_edges=True, opacity=0.5)
        if show_bounding_box:
            bounding_box = trimesh_to_pv(self.trimesh_t.bounding_box)
            plotter.add_mesh(bounding_box, show_edges=True, opacity=0.5)
        if show_bounding_box_oriented:
            bounding_box_oriented = trimesh_to_pv(self.trimesh_t.bounding_box_oriented)
            plotter.add_mesh(bounding_box_oriented, show_edges=True, opacity=0.5)
        if show_bounding_sphere:
            bounding_sphere = trimesh_to_pv(self.trimesh_t.bounding_sphere)
            plotter.add_mesh(bounding_sphere, show_edges=False, opacity=0.5)

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


class SphereFittingCollision:
    def __init__(self, id, arg1, arg2=None, num_groups=10):
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

        self.points = None
        self.uniform_sampling()

        self.grouped_points = None
        self.sphere_centers = None
        self.sphere_radius = None
        self.cluster(num_groups)

    def uniform_sampling(self, resolution=None):
        bounds = self.trimesh.bounds
        lower_bounds = bounds[0]
        upper_bounds = bounds[1]
        sizes = upper_bounds - lower_bounds
        if resolution is None:
            resolution = np.min(sizes) / 10.0

        upper = np.ceil(upper_bounds / resolution, dtype=float) * resolution + 1.0e-6
        lower = np.floor(lower_bounds / resolution, dtype=float) * resolution

        x = np.arange(lower[0], upper[0], resolution)
        y = np.arange(lower[1], upper[1], resolution)
        z = np.arange(lower[2], upper[2], resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        points = np.stack([X, Y, Z], axis=-1)
        points = points.reshape(-1, 3)
        in_mesh = self.trimesh.contains(points)
        self.points = points[in_mesh]

    def cluster(self, num_groups=10, max_step=1000, convergence_threshold=1e-9):
        num_points = len(self.points)
        indices = np.random.choice(num_points, num_groups, replace=False)
        cluster_centers = self.points[indices]
        grouped_points = [None for _ in range(num_groups)]
        last_inertia = 0.0
        for i in range(max_step):
            diff = np.sum(
                (self.points[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :])
                ** 2,
                axis=2,
            )
            group_ids = np.argmin(diff, axis=1)
            for j in range(num_groups):
                grouped_points[j] = self.points[group_ids == j]
                cluster_centers[j] = np.mean(grouped_points[j], axis=0)
            inertia = self.calc_inertia(grouped_points, cluster_centers)
            # print(f"iter: {i}, inertia: {inertia}")
            if abs(last_inertia - inertia) < inertia * convergence_threshold:
                break
            last_inertia = inertia

        self.grouped_points = grouped_points
        self.sphere_centers = np.zeros((num_groups, 3))
        self.sphere_radius = np.zeros(num_groups)
        for i in range(num_groups):
            convex_mesh = trimesh.convex.convex_hull(self.grouped_points[i])
            bounding_sphere = convex_mesh.bounding_sphere
            center = np.mean(bounding_sphere.vertices, axis=0)
            radius = np.mean(np.linalg.norm(bounding_sphere.vertices - center, axis=1))
            self.sphere_centers[i] = center
            self.sphere_radius[i] = radius

    def calc_inertia(self, grouped_points, cluster_centers):
        iner = 0.0
        for i in range(len(grouped_points)):
            iner += np.sum(np.square(grouped_points[i] - cluster_centers[i]))
        return iner

    def show(
        self,
        show_mesh=True,
        show_points=True,
        show_cluster=True,
        show_spheres=True,
    ):
        plotter = pv.Plotter()
        if show_mesh:
            plotter.add_mesh(trimesh_to_pv(self.trimesh), show_edges=True, opacity=0.2)
        if show_points:
            plotter.add_points(self.points, color="r")
        if show_cluster:
            n = len(self.grouped_points)
            colors = np.random.rand(n, 3)
            for i in range(n):
                plotter.add_points(self.grouped_points[i], color=colors[i])
        if show_spheres:
            n = len(self.sphere_centers)
            for i in range(n):
                sphere = pv.Sphere(
                    radius=self.sphere_radius[i],
                    center=self.sphere_centers[i],
                )
                plotter.add_mesh(
                    sphere, color="lightblue", show_edges=True, opacity=0.2
                )
        plotter.show()

    def apply_transform(self, tform):
        self.sphere_centers_t = self.sphere_centers.copy()
        p = tform[:3, 3]
        R = tform[:3, :3]
        self.sphere_centers_t = p + (R @ self.sphere_centers_t.T).T

    def collision_detection(self, sphere_fitting_obj):
        center1 = self.sphere_centers_t
        center2 = sphere_fitting_obj.sphere_centers_t
        cdis = np.linalg.norm(
            center1[:, np.newaxis, :] - center2[np.newaxis, :, :], axis=2
        )

        radius1 = self.sphere_radius
        radius2 = sphere_fitting_obj.sphere_radius
        rdis = radius1[:, np.newaxis] + radius2[np.newaxis, :]

        dis = np.min(cdis - rdis)
        return dis

    def show_spheres(self, plotter):
        n = len(self.sphere_centers_t)
        for i in range(n):
            sphere = pv.Sphere(
                radius=self.sphere_radius[i],
                center=self.sphere_centers_t[i],
            )
            plotter.add_mesh(sphere, color="lightblue", show_edges=True, opacity=0.5)
