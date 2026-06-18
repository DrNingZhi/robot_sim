function points = fun_space_sampling(ranges, resolution, dis_limits)

    if nargin < 3
        dis_limits = 0;  % 默认值
    end

    x_min = ranges(1, 1);
    x_max = ranges(2, 1);
    y_min = ranges(1, 2);
    y_max = ranges(2, 2);
    z_min = ranges(1, 3);
    z_max = ranges(2, 3);

    x = [flip(0:-resolution:x_min), resolution:resolution:x_max];
    y = [flip(0:-resolution:y_min), resolution:resolution:y_max];
    z = [flip(0:-resolution:z_min), resolution:resolution:z_max];
    [X, Y, Z] = ndgrid(x, y, z);
    points = [X(:), Y(:), Z(:)];

    if ~(dis_limits == 0)
        dis = vecnorm(points - dis_limits.ref_point, 2, 2);
        points = points(dis <= dis_limits.dis_threshold, :);

end