clear
clc
addpath(genpath(pwd));

robot = importrobot('ur5.urdf');
robot.DataFormat = 'row';

mode = 'visualization'; % 'analysis' or 'visualization'

if strcmp(mode, 'analysis')
    dof = length(homeConfiguration(robot));
    q_init = zeros(1, dof);
    ee_link_name = 'wrist_3_link';

    direction = [0, 0, 1];

    % workspace sampling
    ranges = [-1, -1, -1;
            1, 1, 1];
    resolution = 0.1;
    points = fun_space_sampling(ranges, resolution);
    num_samples = size(points, 1);
    disp(['Totally ', num2str(num_samples), ' points will be check.']);

    % initialize workspace data
    workspace_data = struct;
    for i = 1:num_samples
        workspace_data(i).position = points(i, :);
        workspace_data(i).reachability = 0;
        workspace_data(i).solution = zeros(1, dof);
    end

    % joint ranges   
    joint_ranges = fun_get_joint_ranges(robot);
    disp('Joint ranges: ');
    disp(joint_ranges);

    % workspace analysis
    disp('Oriented workspace analysis by inverse kinematics: ');
    t0 = datetime('now');
    fun_parfor_progress(num_samples, 0, t0);
    parfor i=1:num_samples
        [success, q] = fun_ik_oriented_pos(robot, ee_link_name, points(i, :), direction, q_init, joint_ranges);
        if success 
            workspace_data(i).reachability = 1;
            workspace_data(i).solution = q; 
        end
        fun_parfor_progress(num_samples, i, t0);
    end
    fun_parfor_progress(num_samples, -1, t0);

    file_name = 'data/workspace/data_oriented_workspace_ik3.mat';
    save(file_name, 'workspace_data');
    disp("Results have been saved in: " + file_name);

elseif strcmp(mode, 'visualization')
    data = load('data/workspace/data_oriented_workspace_ik3.mat', 'workspace_data');
    workspace_data = data.workspace_data;

    points = vertcat(workspace_data.position);
    reachability = vertcat(workspace_data.reachability);
    q = vertcat(workspace_data.solution);
    p = points(reachability==1, :);
    q = q(reachability==1, :);

    show(robot, q(1, :));
    hold on
    scatter3(p(:, 1), p(:, 2), p(:, 3), 'b')
    set(gca, 'fontsize', 20);
    axis([-1,1,-1,1,-1,1]);
    set(gcf, 'color', [1 1 1]);

else
    disp('Invalid analysis type');
end