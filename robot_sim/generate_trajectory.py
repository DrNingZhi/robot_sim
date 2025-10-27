from robot_sim.motion_planner import plan_ee_circle, save_trajectory_data
from robot_sim.robot_model import RobotModel


mjcf_file = "model/ur5/ur5.xml"
robot_model = RobotModel(mjcf_file)

for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
    data = plan_ee_circle(robot_model, 1.0, i, 1.0)
    data_file = "data/traj_1_" + str(i) + "_1.txt"
    save_trajectory_data(data, "data/traj_1_1_1.txt")
