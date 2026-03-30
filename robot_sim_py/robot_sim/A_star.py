import numpy as np


class AStar:
    def __init__(self, ranges, resolution):
        """
        Initialize A* searching space and obstacles. n is the space dimension.

        Parameters
        ----------
        ranges: np.ndarray(n,2)
            Space range, each row is the lower and upper limits of one dimensionality.
        resolution: float
            Resolution used to discretize the space.
        """
        self.ranges = ranges
        self.resolution = resolution
        self.dimension = len(ranges)

        self.open_list = []
        self.open_cost = []
        self.open_parent = []

        self.close_list = []
        self.close_cost = []
        self.close_parent = []

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

    def plan(self, start, target):
        """
        Plan the path.

        Parameters
        ----------
        start: np.ndarray(n,)
            start position
        target: np.ndarray(n,)
            target position
        """
        cp = self.check_points(np.vstack((start, target)))
        if len(cp) < 2:
            raise ValueError(
                "start or target point is out of range or too close to obstacles!"
            )

        self.close_list = np.array([start])
        h0 = np.sum(np.abs(target - start))
        self.close_cost = np.array([[0.0, h0, h0]])
        self.close_parent = np.array([-1], dtype=int)

        start_neighboring = self.generate_new_points(start)
        self.open_list = self.check_points(start_neighboring)
        self.open_parent = np.zeros(len(self.open_list), dtype=int)
        self.open_cost = np.zeros((len(self.open_list), 3))
        self.open_cost[:, 0] = (
            np.linalg.norm(self.open_list - self.close_list[self.open_parent], axis=1)
            + self.close_cost[self.open_parent, 0]
        )
        self.open_cost[:, 1] = np.sum(np.abs(self.open_list - target), axis=1)
        self.open_cost[:, 2] = self.open_cost[:, 0] + self.open_cost[:, 1]

        while True:
            # 从open list取cost最小点，加入close list
            id = np.argmin(self.open_cost[:, 2])
            new_point = self.open_list[id]
            new_cost = self.open_cost[id]
            new_parent = self.open_parent[id]
            self.close_list = np.vstack((self.close_list, new_point))
            self.close_cost = np.vstack((self.close_cost, new_cost))
            self.close_parent = np.hstack((self.close_parent, new_parent))
            new_id = len(self.close_list) - 1
            self.open_list = np.delete(self.open_list, id, axis=0)
            self.open_cost = np.delete(self.open_cost, id, axis=0)
            self.open_parent = np.delete(self.open_parent, id)

            # 新的相邻点
            neighboring = self.generate_new_points(new_point)
            new_neighboring = self.check_points(neighboring)

            # 找到并删去已经在close_list中的点
            distances_close = np.linalg.norm(
                (self.close_list[:, np.newaxis, :] - new_neighboring), axis=2
            )
            min_distances = np.min(distances_close, axis=0)
            new_neighboring = new_neighboring[min_distances > self.resolution / 10]

            # 找到已经在open_list中的点
            distances_open = np.linalg.norm(
                (self.open_list[:, np.newaxis, :] - new_neighboring), axis=2
            )
            points_in_open_list_id = np.where(
                np.min(distances_open, axis=1) < self.resolution / 10
            )[0]

            # 更新已经在open_list中的点的cost和parent
            new_cost_G = new_cost[0] + np.linalg.norm(
                new_point - self.open_list[points_in_open_list_id], axis=1
            )
            old_cost_G = self.open_cost[points_in_open_list_id, 0]
            need_update_id = points_in_open_list_id[new_cost_G < old_cost_G]

            self.open_cost[need_update_id, 0] = new_cost_G[new_cost_G < old_cost_G]
            self.open_cost[:, 2] = self.open_cost[:, 0] + self.open_cost[:, 1]
            self.open_parent[need_update_id] = new_id

            # 其他未在open_list的新点
            not_in_open_list = np.min(distances_open, axis=0) > self.resolution / 10
            new_neighboring = new_neighboring[not_in_open_list]

            # 更新新点的cost和parent，并加入open list
            new_neighboring_cost = np.zeros((len(new_neighboring), 3))
            new_neighboring_cost[:, 0] = new_cost[0] + np.linalg.norm(
                new_point - new_neighboring, axis=1
            )
            new_neighboring_cost[:, 1] = np.sum(
                np.abs(new_neighboring - target), axis=1
            )
            new_neighboring_cost[:, 2] = (
                new_neighboring_cost[:, 0] + new_neighboring_cost[:, 1]
            )
            new_neighboring_parent = np.ones(len(new_neighboring), dtype=int) * new_id
            self.open_list = np.vstack((self.open_list, new_neighboring))
            self.open_cost = np.vstack((self.open_cost, new_neighboring_cost))
            self.open_parent = np.hstack((self.open_parent, new_neighboring_parent))

            # 判断是否无法找到路径
            if len(self.open_list) == 0:
                raise ValueError("Cannot find a path!")

            # 判断是否到达目标点附近
            diff = np.linalg.norm(target - self.open_list, axis=1)
            if np.min(diff) < self.resolution * np.sqrt(self.dimension) / 2:
                target_id = np.argmin(diff)
                break

        results = [self.open_list[target_id]]
        parent_id = self.open_parent[target_id]
        while parent_id >= 0:
            results.append(self.close_list[parent_id])
            parent_id = self.close_parent[parent_id]

        return np.array(results)

    def check_points(self, points):
        """
        check the points in the space range and not too close to obstacles
        """
        lower_check = points >= self.ranges[:, 0]  # (m, n), 判断 B[k, i] ≥ A[i, 0]
        upper_check = points <= self.ranges[:, 1]  # (m, n), 判断 B[k, i] ≤ A[i, 1]
        row_valid = np.all(lower_check & upper_check, axis=1)  # (m,)

        diff = points[:, np.newaxis, :] - self.obstacles[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        valid = np.all(distances > self.resolution + self.buffer, axis=1)
        new_points = points[row_valid & valid]
        return new_points

    def generate_new_points(self, point):
        """
        Generate new neighboring points from one point(n,)
        """
        states = [-1, 0, 1]
        grids = np.meshgrid(*([states] * self.dimension), indexing="ij")
        neighboring = np.stack(grids, axis=-1).reshape(-1, self.dimension)
        new_neighboring = self.resolution * np.delete(
            neighboring, int((3**self.dimension - 1) / 2), axis=0
        )
        return point + new_neighboring
