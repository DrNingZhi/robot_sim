import numpy as np
import pyvista as pv
import trimesh


class CollisionGenerator:
    @staticmethod
    def get_bounding_sphere(mesh):
        bounding_sphere = mesh.bounding_sphere
        center = np.mean(bounding_sphere.vertices, axis=0)
        distances = bounding_sphere.vertices - center
        radius = np.mean(np.linalg.norm(distances, axis=1))
        return [center, radius]

    @staticmethod
    def get_bounding_cylinder(mesh):
        bounding_cylinder = mesh.bounding_cylinder
        bounding_cylinder_dict = bounding_cylinder.to_dict()
        radius = bounding_cylinder_dict["radius"]
        height = bounding_cylinder_dict["height"]
        center = np.array(bounding_cylinder_dict["transform"])[:3, 3]
        direction = bounding_cylinder.direction
        p1 = center - height / 2 * direction
        p2 = center + height / 2 * direction
        return [p1, p2, radius]

    @staticmethod
    def get_oriented_bounding_box(mesh):
        obb = mesh.bounding_box_oriented
        obb_dict = obb.to_dict()
        T = np.array(obb_dict["transform"])
        center = T[:3, 3]
        rotation = T[:3, :3]
        extent = np.array(obb_dict["extents"]) / 2
        return [center, extent, rotation]

    @staticmethod
    def get_spheres_from_cylinder(bounding_cylinder):
        p1, p2, radius = bounding_cylinder
        length = np.linalg.norm(p1 - p2)
        a = radius * 2 / np.sqrt(3.0)
        num = int(np.ceil(length / a))
        if num == 1:
            centers = np.array([(p1 + p2) / 2])
        else:
            dis = np.linspace(a / 2, length - a / 2, num)
            direction = (p2 - p1) / np.linalg.norm(p2 - p1)
            centers = p1 + dis[:, None] * direction
        radius_arg = np.ones((num, 1)) * radius
        return [centers, radius_arg]

    @staticmethod
    def get_spheres_from_obb(obb):
        center = obb[0]
        size = obb[1] * 2
        rotation = obb[2]
        radius = np.min(size) / 2.0

        # 球之间的距离取r/sqrt(3) ~ 1r/sqrt(3)之间
        def calc_coordinate(L):
            num = np.floor((L - (radius * 2.0)) / (radius / np.sqrt(3.0))) + 1
            if num == 1:
                if (L - (radius * 2.0)) > 0.01:
                    num = 2
            if num == 1:
                x = np.array([0.0])
            else:
                x = np.linspace(-L / 2.0 + radius, L / 2.0 - radius, int(num))
            return x

        x_coord = calc_coordinate(size[0])
        y_coord = calc_coordinate(size[1])
        z_coord = calc_coordinate(size[2])
        X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord, indexing="ij")
        centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        centers_t = center + (rotation @ centers.T).T
        radius_arg = np.ones((centers.shape[0], 1)) * radius
        return [centers_t, radius_arg]

    @staticmethod
    def get_spheres_from_mesh(mesh, size=None):
        print("spheres are generating...")
        # 均匀采样
        bounds = mesh.bounds
        lower_bounds = bounds[0]
        upper_bounds = bounds[1]
        sizes = upper_bounds - lower_bounds
        if size is None:
            size = np.min(sizes) / 1.5
        resolution = min(np.min(sizes) / 10.0, size / 10.0)

        upper = np.ceil(upper_bounds / resolution, dtype=float) * resolution + 1.0e-6
        lower = np.floor(lower_bounds / resolution, dtype=float) * resolution

        x = np.arange(lower[0], upper[0], resolution)
        y = np.arange(lower[1], upper[1], resolution)
        z = np.arange(lower[2], upper[2], resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        points = np.stack([X, Y, Z], axis=-1)
        points = points.reshape(-1, 3)
        in_mesh = mesh.contains(points)
        points = points[in_mesh]

        # k-means聚类
        num_points = len(points)
        num_groups = int(mesh.volume / (np.pi * (size / 2) ** 3 * 4 / 3))
        indices = np.random.choice(num_points, num_groups, replace=False)
        cluster_centers = points[indices]
        grouped_points = [None for _ in range(num_groups)]
        last_inertia = 0.0
        while True:
            diff = np.sum(
                (points[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            group_ids = np.argmin(diff, axis=1)
            for j in range(num_groups):
                grouped_points[j] = points[group_ids == j]
                cluster_centers[j] = np.mean(grouped_points[j], axis=0)
            # 计算惯性
            inertia = 0.0
            for i in range(num_groups):
                inertia += np.sum(np.square(grouped_points[i] - cluster_centers[i]))
            # print(f"iter: {i}, inertia: {inertia}")
            if abs(last_inertia - inertia) < inertia * 1e-9:
                break
            last_inertia = inertia

        # 获得外接球
        sphere_centers = np.zeros((num_groups, 3))
        sphere_radius = np.zeros((num_groups, 1))
        for i in range(num_groups):
            convex_mesh = trimesh.convex.convex_hull(grouped_points[i])
            sphere_centers[i], sphere_radius[i, 0] = (
                CollisionGenerator.get_bounding_sphere(convex_mesh)
            )
        print("spheres are generated.")
        return [sphere_centers, sphere_radius]


class CollisionVisualizer:
    @staticmethod
    def trimesh_to_pv(mesh):
        vertices = mesh.vertices
        faces = mesh.faces
        pv_faces = np.hstack((np.ones((len(faces), 1), dtype=int) * 3, faces))
        pv_mesh = pv.PolyData(vertices, pv_faces)
        return pv_mesh

    @staticmethod
    def add_sphere(plotter, sphere, opacity, show_edges=False):
        plotter.add_mesh(
            pv.Sphere(
                radius=sphere[1],
                center=sphere[0],
            ),
            opacity=opacity,
            show_edges=show_edges,
        )

    @staticmethod
    def add_cylinder(plotter, cylinder, opacity, show_edges=False):
        height = np.linalg.norm(cylinder[0] - cylinder[1])
        center = (cylinder[0] + cylinder[1]) / 2
        direction = cylinder[1] - cylinder[0]
        radius = cylinder[2]
        plotter.add_mesh(
            pv.Cylinder(center, direction, radius, height),
            opacity=opacity,
            show_edges=show_edges,
        )

    @staticmethod
    def add_obb(plotter, obb, opacity, show_edges=False):
        transform = np.eye(4)
        transform[:3, 3] = obb[0]
        transform[:3, :3] = obb[2]
        plotter.add_mesh(
            CollisionVisualizer.trimesh_to_pv(
                trimesh.primitives.Box(
                    extents=obb[1] * 2,
                    transform=transform,
                )
            ),
            opacity=opacity,
            show_edges=show_edges,
        )

    @staticmethod
    def add_spheres(plotter, spheres, opacity, show_edges=False):
        pv_spheres = pv.MultiBlock()
        num = len(spheres[0])
        for i in range(num):
            pv_spheres.append(pv.Sphere(radius=spheres[1][i], center=spheres[0][i]))
        plotter.add_mesh(pv_spheres, opacity=opacity, show_edges=show_edges)


class CollisionDetector:
    @staticmethod
    def sphere_sphere(sphere1, sphere2):
        cdis = np.linalg.norm(sphere1[0] - sphere2[0])
        dis = cdis - sphere1[1] - sphere2[1]
        return dis

    @staticmethod
    def box_box(box1, box2):
        box1_min = box1[0]
        box1_max = box1[1]
        box2_min = box2[0]
        box2_max = box2[1]
        dx = max(0, max(box1_min[0], box2_min[0]) - min(box1_max[0], box2_max[0]))
        dy = max(0, max(box1_min[1], box2_min[1]) - min(box1_max[1], box2_max[1]))
        dz = max(0, max(box1_min[2], box2_min[2]) - min(box1_max[2], box2_max[2]))

        overlap_x = max(
            0, min(box1_max[0], box2_max[0]) - max(box1_min[0], box2_min[0])
        )
        overlap_y = max(
            0, min(box1_max[1], box2_max[1]) - max(box1_min[1], box2_min[1])
        )
        overlap_z = max(
            0, min(box1_max[2], box2_max[2]) - max(box1_min[2], box2_min[2])
        )

        if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
            return -min(overlap_x, overlap_y, overlap_z)
        return np.sqrt(dx**2 + dy**2 + dz**2)

    @staticmethod
    def cylinder_cylinder(cylinder1, cylinder2):
        # 仅计算轴线距离，不考虑端面
        P1 = cylinder1[0]
        P2 = cylinder2[0]
        d1 = cylinder1[1] - cylinder1[0]
        d2 = cylinder2[1] - cylinder2[0]
        cross_d = np.cross(d1, d2)
        norm_cross = np.linalg.norm(cross_d)

        vec = P2 - P1
        if norm_cross < 1e-8:  # 平行
            cdis = np.linalg.norm(np.cross(vec, d1)) / np.linalg.norm(d1)
        else:
            cdis = abs(np.dot(vec, cross_d)) / norm_cross
        dis = cdis - cylinder1[2] - cylinder2[2]
        return dis

    @staticmethod
    def obb_obb(obb1, obb2):
        c1 = obb1[0]
        e1 = obb1[1]
        r1 = obb1[2]
        c2 = obb2[0]
        e2 = obb2[1]
        r2 = obb2[2]

        t = c2 - c1  # 中心位移矢量

        # 获取 15 条测试轴
        # 轴 1-3: OBB1 的局部轴 (r1 的列)
        # 轴 4-6: OBB2 的局部轴 (r2 的列)
        axes_a = r1.T  # 每行是一个轴
        axes_b = r2.T
        # 轴 7-15: 边与边的叉积轴
        cross_axes = np.cross(axes_a[:, None, :], axes_b[None, :, :]).reshape(-1, 3)

        # 组合所有待测轴并归一化
        all_axes = np.vstack([axes_a, axes_b, cross_axes])
        norms = np.linalg.norm(all_axes, axis=1, keepdims=True)
        # 过滤平行边产生的无效轴
        valid_mask = (norms > 1e-6).flatten()
        test_axes = all_axes[valid_mask] / norms[valid_mask]

        # 2. 计算投影和间隙
        # 中心在轴上的投影距离
        dist_c = np.abs(np.dot(test_axes, t))

        # 计算两个 OBB 在各轴上的投影半径
        # r = sum(|axis · u_i| * extent_i)
        r_a = np.sum(np.abs(np.dot(test_axes, axes_a.T)) * e1, axis=1)
        r_b = np.sum(np.abs(np.dot(test_axes, axes_b.T)) * e2, axis=1)

        # 3. 计算所有轴上的间隙
        gaps = dist_c - (r_a + r_b)
        return np.max(gaps)

    @staticmethod
    def spheres_spheres(sphere1, sphere2):
        c1 = sphere1[0]
        c2 = sphere2[0]
        r1 = sphere1[1]
        r2 = sphere2[1]
        diff = c1[:, np.newaxis, :] - c2[np.newaxis, :, :]  # 形状为 (n, m, 3)
        dist_matrix = np.linalg.norm(diff, axis=2)  # 形状为 (n, m)
        d = r1 + r2.T  # 结果形状为 (n, m)
        dis = np.min(dist_matrix - d)
        return dis

    @staticmethod
    def mesh_mesh(mesh1, mesh2):
        collision_manager1 = trimesh.collision.CollisionManager()
        collision_manager1.add_object("mesh1", mesh1)
        collision_manager2 = trimesh.collision.CollisionManager()
        collision_manager2.add_object("mesh2", mesh2)
        dis = collision_manager1.min_distance_other(collision_manager2)
        return dis
