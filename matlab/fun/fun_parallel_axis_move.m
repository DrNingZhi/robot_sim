function I = fun_parallel_axis_move(I0,m,p)
%平行移轴，p是列向量，m是重量
I=I0+m*(p'*p*eye(3)-p*p');
end