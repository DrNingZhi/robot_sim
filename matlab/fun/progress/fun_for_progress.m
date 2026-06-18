function fun_for_progress(N, i, t0)

w = 50;

percent = num2str(round(100 * i / N));

start_blank = repmat(' ', 1, 5 - length(percent));

num_equal = round(w * i / N);

if num_equal == w
    num_equal = w - 1;
end

a1 = repmat('=', 1, num_equal);

a2 = repmat(' ', 1, w - num_equal - 1);

% Estimated
% Estimated remaining time

if nargin < 3 || isempty(t0)

    if i == 1
        disp([start_blank, percent, '% [', a1, '>', a2, ']']);
    else
        back_space = repmat(char(8), 1, w + 10);
        disp([back_space, start_blank, percent, '% [', a1, '>', a2, ']', ]);
    end

else
    current_time = datetime('now');
    use_time = seconds(current_time - t0);
    remaining_time = use_time * (N - i) / i;
    remaining_time_str = num2str(remaining_time);

    if length(remaining_time_str) > 6
        remaining_time_str = remaining_time_str(1:6);
    else
        remaining_time_str = [remaining_time_str, repmat(' ', 1, 6 - length(remaining_time_str))];
    end

    if i == 1
        disp([start_blank, percent, '% [', a1, '>', a2, '] Estimated remaining time: ', remaining_time_str, ' s.']);
    else
        back_space = repmat(char(8), 1, w + 10 + 36);
        disp([back_space, start_blank, percent, '% [', a1, '>', a2, '] Estimated remaining time: ', remaining_time_str, ' s.']);
    end

end