import numpy as np
from robot_sim.robot_model import RobotModel
import time
from robot_sim.collision import Collision, CollisionDetectionMethod
from robot_sim.collision_robot import RobotCollisionDetector, CollisionDetectionConfig
import trimesh

# method = CollisionDetectionMethod.BoundingSphere
# method = CollisionDetectionMethod.BoundingBox
# method = CollisionDetectionMethod.BoundingCylinder
# method = CollisionDetectionMethod.OrientedBoundingBox
# method = CollisionDetectionMethod.SphereByBoundingCylinder
# method = CollisionDetectionMethod.SphereByBoundingBox
# method = CollisionDetectionMethod.SphereByConvexHull
# method = CollisionDetectionMethod.SphereByOriginalMesh
# method = CollisionDetectionMethod.ConvexHull
method = CollisionDetectionMethod.OriginalMesh

config = CollisionDetectionConfig()  # 默认都考虑

mjcf_file = "model/ur5/ur5.xml"
robot_model = RobotModel(mjcf_file)

robot_collision_detector = RobotCollisionDetector(robot_model, method, config)

# q = np.zeros(robot_model.dof)
q = np.random.rand(robot_model.dof)
# q = np.array([[0.6900721, 0.95876795, 0.06526024, 0.2282442, 0.66531103, 0.38351114]])

t0 = time.time()
min_dis, results = robot_collision_detector.self_collision_detection(q)
t1 = time.time()

print("碰撞检测耗时：", t1 - t0)
print("最小距离：", min_dis)
print("碰撞检测信息：", len(results))
print("碰撞检测信息：", results)

robot_collision_detector.show_collision(q)
