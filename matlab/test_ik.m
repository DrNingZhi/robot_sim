clear
clc
addpath(genpath(pwd));

robot = importrobot('ur5.urdf');
robot.DataFormat = 'row';
joint_ranges = fun_get_joint_ranges(robot);

q=randomConfiguration(robot)
ee_link_name = 'wrist_3_link';
T = getTransform(robot, q, ee_link_name);
pos = T(1:3, 4)'

% ranges = [-1, -1, -1;
%             1, 1, 1] * 0.5;
% resolution = 0.1;
% points = fun_space_sampling(ranges, resolution);
% n = ceil(rand * size(points, 1));
% pos = points(n, :)
q_init = zeros(1, 6);
[success, q2] = fun_ik_oriented_pos(robot, ee_link_name, pos, [1,0,0], q_init, joint_ranges)


T2 = getTransform(robot, q2, ee_link_name);
pos2 = T2(1:3, 4)
% err = norm(T2(1:3, 4) - pos(:))

show(robot, q2)
hold on
scatter3(pos(1), pos(2), pos(3))