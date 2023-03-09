function main()
    fig = figure;

    for l = [2^2, 2^4, 2^6]
        [sizes, results] = timer(2^1, 2^9, l, 2, 5, 42);
        mean = results(:, 1);
        std = results(:, 2);

        loglog(sizes, mean, 'DisplayName', ['l = ', num2str(l)]);
        hold on;
    end
    
    xlim([sizes(1), sizes(end)]);
    xlabel('Matrix size (k x k)');
    ylabel('Time [s]');
    
    title('Classic + Binet matrix multiplication');
    legend('Location', 'northwest');
    
    grid on;
    set(gca, 'XMinorGrid', 'off', 'YMinorGrid', 'off');
    
    hold off;
    saveas(fig, 'Time results.pdf');

    % TODO number of floating point calculations
end

function [sizes, results] = timer(min_size, max_size, l, multiplier, num_rep, seed)
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
    end
end

function C = matmul(A, B, l)
    if size(A, 1) <= l
        C = classic_matmul(A, B);
    else
        C = binet_matmul(A, B, l);
    end
end

function C = classic_matmul(A, B)
    % TODO classic matrix multiplication
    C = A * B;
end

function C = binet_matmul(A, B, l)
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
end
