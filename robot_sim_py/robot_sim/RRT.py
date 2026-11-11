import numpy as np


class RRT:
    def __init__(self, ranges, step_size, sample_step=None):
        """
        Initialize RRT searching space and obstacles. n is the space dimension.

        Parameters
        ----------
        ranges: np.ndarray(n,2)
            Space range, each row is the lower and upper limits of one dimensionality.
        step_size: float
            Step size used to extend the tree towards sampled points.
        sample_step: float, optional
            Sampling interval along an edge for collision checking.
            Defaults to step_size / 10.
        """
        self.ranges = ranges
        self.step_size = step_size
        self.dimension = len(ranges)
        if sample_step is None:
            sample_step = step_size / 10
        self.sample_step = sample_step

        self.vertex_list = []
        self.parent_list = []

    def add_obstacles(self, obstacles, buffer):
        """
        setup the obstacles.

        Parameters
        ----------
        obstacles: np.ndarray(m,n)
            List of nodes of obstacles. Should has the same dimensionality as the space.
        buffer: float
            Buffer distance considered from obstacles
        """
        self.obstacles = obstacles
        self.buffer = buffer

    def plan(self, start, target, max_iter=10000, goal_sample_rate=0.05):
        """
        Plan the path.

        Parameters
        ----------
        start: np.ndarray(n,)
            start position
        target: np.ndarray(n,)
            target position
        max_iter: int
            Maximum number of sampling iterations.
        goal_sample_rate: float
            Probability of sampling the target point directly.

        Returns
        -------
        np.ndarray(k,n)
            Path from start to target.
        """
        cp = self.check_points(np.vstack((start, target)))
        if len(cp) < 2:
            raise ValueError(
                "start or target point is out of range or too close to obstacles!"
            )

        self.vertex_list = np.array([start])
        self.parent_list = np.array([-1], dtype=int)

        # 起点可以直接连到目标
        if np.linalg.norm(start - target) < self.step_size and self.check_edge(
            start, target
        ):
            return np.vstack((start, target))

        for _ in range(max_iter):
            # 随机采样，以一定概率直接采样目标点
            if np.random.rand() < goal_sample_rate:
                sample = target
            else:
                sample = self.ranges[:, 0] + np.random.rand(self.dimension) * (
                    self.ranges[:, 1] - self.ranges[:, 0]
                )

            # 找到树上离采样点最近的顶点
            distances = np.linalg.norm(self.vertex_list - sample, axis=1)
            nearest_id = int(np.argmin(distances))
            nearest = self.vertex_list[nearest_id]

            # 从最近顶点向采样点方向扩展一步
            direction = sample - nearest
            length = np.linalg.norm(direction)
            if length < 1e-12:
                continue
            step = min(self.step_size, length)
            new_point = nearest + direction * (step / length)

            # 检查新边是否发生碰撞
            if not self.check_edge(nearest, new_point):
                continue

            # 新顶点加入树
            self.vertex_list = np.vstack((self.vertex_list, new_point))
            self.parent_list = np.hstack((self.parent_list, nearest_id))
            new_id = len(self.vertex_list) - 1

            # 新顶点足够接近目标时，尝试直接连到目标
            if (
                np.linalg.norm(new_point - target) < self.step_size
                and self.check_edge(new_point, target)
            ):
                self.vertex_list = np.vstack((self.vertex_list, target))
                self.parent_list = np.hstack((self.parent_list, new_id))

                results = [self.vertex_list[-1]]
                parent_id = self.parent_list[-1]
                while parent_id >= 0:
                    results.append(self.vertex_list[parent_id])
                    parent_id = self.parent_list[parent_id]

                return np.array(results[::-1])

        raise ValueError("Cannot find a path!")

    def check_points(self, points):
        """
        check the points in the space range and not too close to obstacles
        """
        lower_check = points >= self.ranges[:, 0]
        upper_check = points <= self.ranges[:, 1]
        row_valid = np.all(lower_check & upper_check, axis=1)

        diff = points[:, np.newaxis, :] - self.obstacles[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        valid = np.all(distances > self.buffer, axis=1)
        new_points = points[row_valid & valid]
        return new_points

    def check_edge(self, from_point, to_point):
        """
        Check whether the straight edge between two points is collision free,
        by sampling points along the edge with interval sample_step.
        """
        length = np.linalg.norm(to_point - from_point)
        num = int(length / self.sample_step) + 2
        points = np.linspace(from_point, to_point, num)
        return len(self.check_points(points)) == num
