function main()

    fig = figure;

    min_size = 1;
    max_size = 11;
    num_rep = 5;
    n = max_size - min_size + 1;
    l = [6];

    global op;
    global op_index;
    global index;

    op = zeros(n, size(l, 2)); % number of operations for each matrix size
    op_index = 1;

    % colors for plots
    newcolors = [0, 0, 1
                 0, 1, 0
                 1, 0, 0];

    colororder(newcolors)

    % time plots
    for i = 1:1:size(l, 2)
        [sizes, results] = timer(2^min_size, 2^max_size, 2^l(i), 2, num_rep, 42);
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

    grid on;
    set(gca, 'XMinorGrid', 'off', 'YMinorGrid', 'off');

    hold off;
    saveas(fig, 'Float results.svg');
end

% Function for measuring matrix inversion time
function [sizes, results] = timer(min_size, max_size, l, multiplier, num_rep, seed)

    global index;
    index = 1;

    rng(seed);

    sizes = [];
    results = [];

    size = min_size;

    while size <= max_size
        inversion_time = [];

        for i = 1:num_rep
            A = rand(size);

            tic
            B = inversion(A, l);
            inversion_time(end+1) = toc;
            
            if size <= 256
                assert(all(all(abs(eye(size) - (A*B)) < 1e-3)));
            end
        end

        results(end+1, :) = [mean(inversion_time), std(inversion_time)];

        sizes(end+1) = size;
        size = size * multiplier;

        index = index + 1;
    end
end

% Matrix inversion function
function B = inversion(A, l)

    global op;
    global op_index;
    global index;

    n = size(A, 1);

    if n == 1
        B = 1 / A;

        op(index, op_index) = op(index, op_index) + 1;
    else
        A11 = A(1:n/2, 1:n/2);
        A12 = A(1:n/2, n/2+1:end);
        A21 = A(n/2+1:end, 1:n/2);
        A22 = A(n/2+1:end, n/2+1:end);

        A11_inv = inversion(A11, l);
        A12_inv = inversion(A12, l);

        S22 = A22 - matmul(matmul(A21, A11_inv, l), A12, l);
        S22_inv = inversion(S22, l);

        B11 = matmul(A11_inv, eye(n/2) + matmul(matmul(matmul(A12, S22_inv, l), A21, l), A11_inv, l), l);
        B12 = -matmul(matmul(A11_inv, A12, l), S22_inv, l);
        B21 = -matmul(matmul(S22_inv, A21, l), A11_inv, l);
        B22 = S22_inv;

        B = [B11, B12;
             B21, B22];

        op(index, op_index) = op(index, op_index) + 4*(n/2)*(n/2);
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

    [n, m] = size(A);
    [m, k] = size(B);
    C = zeros(n, k);

    for a = 1:1:n % rows in C
        for b = 1:1:k % columns in C
            sum = 0;

            for c = 1:1:m % columns in A and rows in B
                sum = sum + A(a, c) * B(c, b);
            end

            C(a, b) = sum;
        end
    end

    op(index, op_index) = op(index, op_index) + 2*n*k*m;
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