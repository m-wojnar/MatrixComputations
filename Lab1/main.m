function main()

    fig = figure;

    min_size = 1;
    max_size = 12;
    num_rep = 5;
    n = max_size - min_size + 1;
    l = [3, 6, 9];

    global op;
    global op_index;
    global index;

    op = zeros(n, size(l, 2)); % number of operations for each matrix size
    op_index = 1;

    % colors for plots
    newcolors = [1, 0, 0
                 0, 1, 0
                 0, 0, 1];

    colororder(newcolors)

    % time plots
    for i = 1:1:size(l, 2)
        [sizes, results] = timer(2^min_size, 2^max_size, 2^l(i), 2, num_rep, 42); % wykonywanie mnożeń
        mean = results(:, 1);
        std = results(:, 2);

        loglog(sizes, mean, 's', 'MarkerFaceColor', newcolors(i,:), 'DisplayName', ['l = ', num2str(2^l(i))]);
        hold on;

        op_index = op_index + 1;
    end

    op = op / num_rep;

    xlim([sizes(1), sizes(end)]);
    xticks(sizes);
    xlabel('Matrix size (k x k)');
    ylabel('Time [s]');
    legend('Location', 'northwest');

    grid on;
    set(gca, 'XMinorGrid', 'off', 'YMinorGrid', 'off');

    hold off;
    saveas(fig, 'Time results.svg');

    % floating-point operations plots
    for i = 1:1:size(l, 2)
        loglog(sizes, op(:, i), 's', 'MarkerFaceColor', newcolors(i,:), 'DisplayName', ['l = ', num2str(2^l(i))]);
        hold on;
    end

    xlim([sizes(1), sizes(end)]);
    xticks(sizes);
    xlabel('Matrix size (k x k)');
    ylabel('Floating-points operations');
    legend('Location', 'northwest');

    grid on;
    set(gca, 'XMinorGrid', 'off', 'YMinorGrid', 'off');

    hold off;
    saveas(fig, 'Float results.svg');
end

% Function for measuring matrix multiplication time
function [sizes, results] = timer(min_size, max_size, l, multiplier, num_rep, seed)

    global index;
    index = 1;

    rng(seed);

    sizes = [];
    results = [];

    size = min_size;

    while size <= max_size
        multiplication_time = [];

        for i = 1:num_rep
            A = rand(size);
            B = rand(size);

            tic
            C = matmul(A, B, l);
            multiplication_time(end+1) = toc;

            assert(all(all(abs(C - (A*B)) < 1e-8)));
        end

        results(end+1, :) = [mean(multiplication_time), std(multiplication_time)];

        sizes(end+1) = size;
        size = size * multiplier;

        index = index + 1;
    end
end

% Matrix multiplication function
function C = matmul(A, B, l)

    if size(A, 1) <= l
        C = classic_matmul(A, B);
    else
        C = binet_matmul(A, B, l);
    end
end

% Classic matrix multiplication
function C = classic_matmul(A, B)

    global op;
    global op_index;
    global index;

    new_op = 0; % number of floating-point operations

    [n, m] = size(A);
    [m, k] = size(B);
    C = zeros(n, k);

    for a = 1:1:n % rows in C
        for b = 1:1:k % columns in C
            sum = 0;

            for c = 1:1:m % columns in A and rows in B
                sum = sum + A(a, c) * B(c, b);
                new_op = new_op + 2; % one addition and one multiplication
            end

            C(a, b) = sum;
        end
    end

    op(index, op_index) = op(index, op_index) + new_op;
end

% Binet matrix recursive multiplication
function C = binet_matmul(A, B, l)

    global op;
    global op_index;
    global index;

    [n, m] = size(A);
    A11 = A(1:n/2, 1:m/2);
    A12 = A(1:n/2, m/2+1:end);
    A21 = A(n/2+1:end, 1:m/2);
    A22 = A(n/2+1:end, m/2+1:end);

    [m, k] = size(B);
    B11 = B(1:m/2, 1:k/2);
    B12 = B(1:m/2, k/2+1:end);
    B21 = B(m/2+1:end, 1:k/2);
    B22 = B(m/2+1:end, k/2+1:end);


    C = [matmul(A11, B11, l) + matmul(A12, B21, l), matmul(A11, B12, l) + matmul(A12, B22, l);
         matmul(A21, B11, l) + matmul(A22, B21, l), matmul(A21, B12, l) + matmul(A22, B22, l)];

    op(index, op_index) = op(index, op_index) + n*k;
end