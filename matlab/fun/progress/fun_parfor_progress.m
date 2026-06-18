function fun_parfor_progress(N, i, t0)

    if i == 0

        if exist('temp_file.txt')
            delete('temp_file.txt');
        end

        f = fopen('temp_file.txt', 'a');
        fprintf(f, '%d\n', 0.001);
        fclose(f);

        if nargin < 3 || isempty(t0)

            fun_for_progress(N, 1);
        else

            fun_for_progress(N, 1, t0);

        end

    elseif i == -1
        delete('temp_file.txt');

        if nargin < 3 || isempty(t0)

            fun_for_progress(N, N);
        else

            fun_for_progress(N, N, t0);

        end

    else
        f = fopen('temp_file.txt', 'a');
        fprintf(f, '%d\n', 1);
        fclose(f);

        f = fopen('temp_file.txt', 'r');
        progress = fscanf(f, '%f');
        fclose(f);
        aa = length(progress);

        if nargin < 3 || isempty(t0)

            fun_for_progress(N, aa);
        else

            fun_for_progress(N, aa, t0);

        end

    end

end
