clear
clc

addpath(genpath(pwd));

robot = importrobot('ur5.urdf');
robot.DataFormat = 'row';

ee_link_name = 'wrist_3_link';
num_samples = 10000;
p = fun_workspace_fk(robot, ee_link_name, num_samples);

show(robot);
hold on
scatter3(p(:, 1), p(:, 2), p(:, 3), 'b')
set(gca, 'fontsize', 20);
axis([-1,1,-1,1,-1,1]);
set(gcf, 'color', [1 1 1]);