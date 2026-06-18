clear
clc
addpath(genpath(pwd));
robot = importrobot('ur5.urdf');
robot.DataFormat = 'row';
show(robot);