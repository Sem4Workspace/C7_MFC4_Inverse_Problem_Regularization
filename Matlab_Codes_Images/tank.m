%% Real-World Image Reconstruction — Pseudoinverse Baseline
% Demonstrates pseudoinverse reconstruction on a real image and why it fails.
%
% *Forward Model:*
%
% $$y = A\,x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$
%
% *Pseudoinverse:*
%
% $$x^{\dagger} = A^{\dagger}y = V\Sigma^{-1}U^T y$$
%
% This fails for ill-conditioned $A$ because small $\sigma_i$ amplify noise.
%
% *Error Metrics:*
%
% $$\text{MSE} = \frac{1}{N}\sum(x_i^{\text{rec}}-x_i^{\text{true}})^2, \quad
% \text{Relative Error} = \frac{\|x^{\text{rec}}-x^{\text{true}}\|}{\|x^{\text{true}}\|}$$

%% 1. Load Image
img = imread('296059.jpg');
if ndims(img) == 3
    img = rgb2gray(img);
end
x_full = double(img);
x_full = x_full / max(x_full(:));

[rows, cols] = size(x_full);
fprintf('Full image shape: %d x %d\n', rows, cols);

%% 2. Gaussian Kernel
% *Gaussian PSF:*
%
% $$h(i,j) = \frac{1}{Z}\exp\!\left(-\frac{i^2+j^2}{2\sigma_k^2}\right)$$

kernel_size = 9;
sigma_k = 2.0;

half = floor(kernel_size / 2);
[xx, yy] = meshgrid(-half:half, -half:half);
kernel = exp(-(xx.^2 + yy.^2) / (2 * sigma_k^2));
kernel = kernel / sum(kernel(:));

%% 3. Extract Patch and Build Explicit Matrix
x_true = x_full;
n = rows;

fprintf('Building explicit convolution matrix (%dx%d)...\n', n*n, n*n);
N = n * n;
A = zeros(N, N);
for i = 1:N
    basis = zeros(n, n);
    basis(i) = 1.0;
    conv_result = imfilter(basis, kernel, 'symmetric', 'same');
    A(:, i) = conv_result(:);
end
fprintf('Explicit A shape: %d x %d\n', size(A));

%% 4. Generate Observation
% $$y = Ax + \varepsilon$$

Ax = imfilter(x_true, kernel, 'symmetric', 'same');
noise_std = 0.01;
rng(42);
eps = noise_std * randn(rows, cols);
y = Ax + eps;

x_vec = x_true(:);
y_vec = y(:);

%% 5. SVD and Pseudoinverse
% *SVD:*
%
% $$A = U\Sigma V^T$$
%
% *Pseudoinverse reconstruction:*
%
% $$x^{\dagger} = V\Sigma^{-1}U^T y$$

[U, S_mat, V] = svd(A, 'econ');
s = diag(S_mat);
fprintf('Condition number of A: %.2e\n', s(1)/s(end));

x_pinv = V * (U' * y_vec ./ s);
x_pinv_img = reshape(x_pinv, rows, cols);

%% 6. Visualisation
figure;
subplot(1,3,1); imshow(x_true, []);     title('Original');
subplot(1,3,2); imshow(y, []);           title('Blurred + Noisy');
subplot(1,3,3); imshow(x_pinv_img, []); title('Pseudoinverse');

figure;
semilogy(s, 'o-');
xlabel('Index');
ylabel('Singular value (log scale)');
title('Singular Value Decay of A');
grid on;

%% 7. Error Analysis
mse_pinv = mean((x_pinv - x_vec).^2);
rel_err = norm(x_pinv - x_vec) / norm(x_vec);

fprintf('\n%s\n', repmat('=', 1, 50));
fprintf('PSEUDOINVERSE RECONSTRUCTION ERROR ANALYSIS\n');
fprintf('%s\n', repmat('=', 1, 50));
fprintf('Relative Error: %.6f\n', rel_err);
fprintf('MSE: %.8f\n', mse_pinv);
fprintf('Condition number: %.2e\n', s(1)/s(end));
fprintf('Image size: %dx%d\n', rows, cols);
fprintf('Kernel: %dx%d, sigma=%.1f\n', kernel_size, kernel_size, sigma_k);
