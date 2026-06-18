function [success, q] = fun_ik_pos(robot, ee_link_name, pos, q_init, joint_ranges)
    gik = generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs',{'position', 'joint'});
    positionConst = constraintPositionTarget(ee_link_name);
    positionConst.TargetPosition = pos;
    jointConst = constraintJointBounds(robot);
    jointConst.Bounds = joint_ranges;
    [q, ~] = gik(q_init, positionConst, jointConst);
    
    % check solution
    T = getTransform(robot, q, ee_link_name);
    err = norm(T(1:3, 4) - pos(:));
    if err <= 0.001
        success = 1;
    else
        success = 0;
    end
end