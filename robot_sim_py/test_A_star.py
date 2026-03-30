from robot_sim.A_star import AStar
import numpy as np
import matplotlib.pyplot as plt


def obstacle(start, length, direction, resolution):
    num = int(length / resolution) + 1
    lengths = np.linspace(0.0, length, num)
    return start + direction * lengths[:, np.newaxis]


ranges = np.array([[-1.0, 1.0], [-1.0, 1.0]])
resolution = 0.01
A_star = AStar(ranges, resolution)


obstacle1 = obstacle(np.array([-0.3, 0.25]), 1.0, np.array([0.0, -1.0]), resolution)
obstacle2 = obstacle(np.array([0.3, -0.25]), 1.0, np.array([0.0, 1.0]), resolution)
obstacles = np.vstack((obstacle1, obstacle2))
buffer = 0.05
A_star.add_obstacles(obstacles, buffer)

start = np.array([-0.9, 0.0])
target = np.array([0.9, 0.0])
results = A_star.plan(start, target)

plt.scatter(obstacles[:, 0], obstacles[:, 1])
plt.plot(results[:, 0], results[:, 1])
plt.axis("equal")
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()
