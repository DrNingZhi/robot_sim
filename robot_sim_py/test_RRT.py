from robot_sim.RRT import RRT
import numpy as np
import matplotlib.pyplot as plt


def obstacle(start, length, direction, resolution):
    num = int(length / resolution) + 1
    lengths = np.linspace(0.0, length, num)
    return start + direction * lengths[:, np.newaxis]


ranges = np.array([[-1.0, 1.0], [-1.0, 1.0]])
step_size = 0.1
rrt = RRT(ranges, step_size)


obstacle1 = obstacle(np.array([-0.3, 0.25]), 1.0, np.array([0.0, -1.0]), 0.01)
obstacle2 = obstacle(np.array([0.3, -0.25]), 1.0, np.array([0.0, 1.0]), 0.01)
obstacles = np.vstack((obstacle1, obstacle2))
buffer = 0.05
rrt.add_obstacles(obstacles, buffer)

start = np.array([-0.9, 0.0])
target = np.array([0.9, 0.0])
results = rrt.plan(start, target)

# 画出探索树
for i in range(1, len(rrt.vertex_list)):
    child = rrt.vertex_list[i]
    parent = rrt.vertex_list[rrt.parent_list[i]]
    plt.plot([parent[0], child[0]], [parent[1], child[1]], color="lightgray", linewidth=0.5, zorder=1)

plt.scatter(obstacles[:, 0], obstacles[:, 1], s=4)
plt.plot(results[:, 0], results[:, 1], zorder=2)
plt.scatter(*start, c="green", zorder=3)
plt.scatter(*target, c="red", zorder=3)
plt.axis("equal")
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()
