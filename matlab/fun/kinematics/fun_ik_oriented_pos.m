function [success, q] = fun_ik_oriented_pos(robot, ee_link_name, pos, direction, q_init, joint_ranges)
    gik = generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs',{'position', 'aiming', 'joint'});
    positionConst = constraintPositionTarget(ee_link_name);
    positionConst.TargetPosition = pos;
    jointConst = constraintJointBounds(robot);
    jointConst.Bounds = joint_ranges;
    aimConst = constraintAiming(ee_link_name);
    aimConst.TargetPoint = pos + direction;
    [q, ~] = gik(q_init, positionConst, aimConst, jointConst);

    % check solution
    direction = direction / norm(direction);
    T = getTransform(robot, q, ee_link_name);
    pos_err = norm(T(1:3, 4) - pos(:));
    dir_err = acos(dot(T(1:3, 3), direction(:)) / norm(direction));
    if pos_err <= 0.001 && dir_err <= pi / 180
        success = 1;
    else
        success = 0;
    end
end