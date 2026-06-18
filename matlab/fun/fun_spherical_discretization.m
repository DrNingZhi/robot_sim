function dir = fun_spherical_discretization(N)
    if nargin < 1 || N <= 0 || mod(N,1)~=0
        error('N 必须是正整数');
    end

    % 黄金角
    phi = pi * (3 - sqrt(5));   % ≈ 2.399963

    % 索引
    i = (0:N-1)';

    % z 坐标
    z = 1 - (2*i)/(N-1);

    % 极角
    theta = i * phi;

    % 笛卡尔坐标
    r = sqrt(1 - z.^2);
    X = r .* cos(theta);
    Y = r .* sin(theta);
    Z = z;
    dir = [X, Y, Z];
end