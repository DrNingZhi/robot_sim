function color = fun_color_mapping(p)
    % 红橙黄绿青蓝紫
    color_ref = [255, 0, 0;
                 255, 127, 0;
                 255, 255, 0;
                 0, 255, 0;
                 0, 255, 255;
                 0, 0, 255;
                 139, 0, 255] / 255;
    use_ref_color_num = 7;

    if p >= 1
        color = color_ref(1, :);
        return;
    end

    if p <= 0
        color = color_ref(use_ref_color_num, :);
        return;
    end

    spacing = 1 / (use_ref_color_num - 1);
    idx = use_ref_color_num - ceil(p / spacing);
    p_mod = mod(p, spacing);

    if p_mod == 0
        p_mod = spacing;
    end

    color = (p_mod / spacing) * (color_ref(idx, :) - color_ref(idx + 1, :)) + color_ref(idx + 1, :);
end