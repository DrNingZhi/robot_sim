function points = fun_workspace_fk(robot, ee_link_name, num_samples)
    points = zeros(num_samples, 3);
    disp('Workspace analysis by forward kinematics: ')
    t0 = datetime('now');
    fun_parfor_progress(num_samples, 0, t0);
    parfor i=1:num_samples
        q = randomConfiguration(robot);
        T = getTransform(robot, q, ee_link_name);
        points(i, :) = T(1:3, 4);
        fun_parfor_progress(num_samples, i, t0);
    end
    fun_parfor_progress(num_samples, -1, t0);
end