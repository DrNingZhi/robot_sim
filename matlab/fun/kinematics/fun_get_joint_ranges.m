function joint_ranges = fun_get_joint_ranges(robot)
    dof = length(homeConfiguration(robot));
    joint_ranges = zeros(dof, 2);
    n = 1;
    for i=1:robot.NumBodies;
        if strcmp(robot.Bodies{i}.Joint.Type, 'revolute') || strcmp(robot.Bodies{i}.Joint.Type, 'prismatic')
            joint_ranges(n,:) = robot.Bodies{i}.Joint.PositionLimits;
            n = n + 1;
        end
    end
end