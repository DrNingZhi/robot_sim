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

    % workspace sampling
    ranges = [0, 0, -1;
            1, 1, 1];
    resolution = 0.1;
    points = fun_space_sampling(ranges, resolution);
    N_dir = 50;
    directions = fun_spherical_discretization(N_dir);
    N_p = size(points, 1);
    disp(['Totally ', num2str(N_p * N_dir), ' points will be check.']);

    % initialize workspace data
    workspace_data = struct;
    for i = 1:N_p
        workspace_data(i).position = points(i, :);
        workspace_data(i).reachability = 0;
        workspace_data(i).direction_reachability = struct;
        workspace_data(i).dexterity = 0;
        for j = 1:N_dir
            workspace_data(i).direction_reachability(j).direction = directions(j,:);
            workspace_data(i).direction_reachability(j).reachability = 0;
            workspace_data(i).direction_reachability(j).solution = zeros(1, dof);
        end
    end

    % joint ranges   
    joint_ranges = fun_get_joint_ranges(robot);
    disp('Joint ranges: ');
    disp(joint_ranges);

    % workspace analysis
    disp('Reachable workspace analysis by inverse kinematics: ');
    t0 = datetime('now');
    fun_parfor_progress(N_p, 0, t0);
    parfor i=1:N_p
        for j=1:N_dir
            [success, q] = fun_ik_oriented_pos(robot, ee_link_name, points(i, :), directions(j, :), q_init, joint_ranges);
            if success 
                workspace_data(i).reachability = 1;
                workspace_data(i).direction_reachability(j).reachability = 1;
                workspace_data(i).direction_reachability(j).solution = q;
            end
        end
        fun_parfor_progress(N_p, i, t0);
    end
    fun_parfor_progress(N_p, -1, t0);
    
    for i=1:N_p
        workspace_data(i).dexterity = sum(vertcat(workspace_data(i).direction_reachability.reachability)) / N_dir;
    end

    file_name = 'data/workspace/data_dexterity_workspace_ik.mat';
    save(file_name, 'workspace_data');
    disp("Results have been saved in: " + file_name);

elseif strcmp(mode, 'visualization')
    data = load('data/workspace/data_dexterity_workspace_ik.mat', 'workspace_data');
    workspace_data = data.workspace_data;

    points = vertcat(workspace_data.position);
    dexterity = vertcat(workspace_data.dexterity);
    reachability = vertcat(workspace_data.reachability);
    p = points(reachability==1, :);
    dexterity = dexterity(reachability==1);
    colors = zeros(length(dexterity), 3);
    for i = 1:length(dexterity)
        colors(i,:) = fun_color_mapping(dexterity(i));
    end
    show(robot);
    hold on
    scatter3(p(:, 1), p(:, 2), p(:, 3), 50, colors);
    set(gca, 'fontsize', 20);
    axis([-1,1,-1,1,-1,1]);
    set(gcf, 'color', [1 1 1]);
else
    disp('Invalid analysis type');
end