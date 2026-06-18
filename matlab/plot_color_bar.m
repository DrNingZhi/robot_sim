clear
clc
addpath(genpath(pwd));

a = 0:0.01:1;
colors = zeros(length(a),3);
for i=1:length(a)
    colors(i,:) = fun_color_mapping(a(i));
    plot([0.1, 0.2], [a(i), a(i)], 'Color',colors(i,:), 'LineWidth',5);
    hold on
end

axis([0, 1, -0.1, 1.1])
