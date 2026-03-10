%% Toy Example — Why Inverse Problems Need Regularization
% This script demonstrates on a small 16x16 image why naive inversion fails
% and regularization is necessary.
%
% *Forward Model:*
%
% $$y = A\,x + \varepsilon$$
%
% *Pseudoinverse solution:*
%
% $$x^{\dagger} = A^{\dagger}y = V\Sigma^{-1}U^T y = \sum_{i=1}^{r}\frac{u_i^T y}{\sigma_i}v_i$$
%
% When singular values $\sigma_i$ are small, noise is amplified:
%
% $$\frac{u_i^T \varepsilon}{\sigma_i} \gg 1 \implies \text{reconstruction blows up}$$
%
% *Error Metrics:*
%
% $$\text{MSE} = \frac{1}{N}\sum(x_i^{\text{rec}}-x_i^{\text{true}})^2, \quad
% \text{Relative Error} = \frac{\|x^{\text{rec}}-x^{\text{true}}\|}{\|x^{\text{true}}\|}$$

%% 1. Load Toy Image
img = imread('toy16x16.png');
if ndims(img) == 3
    img = rgb2gray(img);
end
x = double(img);
x = x / max(x(:));

n = size(x, 1);
fprintf('Image size: %d x %d\n', n, n);

figure;
imshow(x, []);
title('Toy Image (16x16)');

%% 2. Gaussian Blur Kernel
% *Gaussian PSF:*
%
% $$h(i,j) = \frac{1}{Z}\exp\!\left(-\frac{i^2+j^2}{2\sigma_k^2}\right)$$

kernel_size = 7;
sigma_k = 1.5;

half = floor(kernel_size / 2);
[xx, yy] = meshgrid(-half:half, -half:half);
kernel = exp(-(xx.^2 + yy.^2) / (2 * sigma_k^2));
kernel = kernel / sum(kernel(:));

%% 3. Build Blur Matrix and Generate Observation
x_vec = x(:);
N = n * n;
A = zeros(N, N);
for i = 1:N
    basis = zeros(n, n);
    basis(i) = 1.0;
    blurred = imfilter(basis, kernel, 'symmetric', 'same');
    A(:, i) = blurred(:);
end
fprintf('Matrix A shape: %d x %d\n', size(A));

Ax = A * x_vec;

noise_level = 0.01;
rng(42);
noise = noise_level * randn(N, 1);
y = Ax + noise;

figure;
imshow(reshape(y, n, n), []);
title('Blurred + Noisy Image y = Ax + \epsilon');

%% 4. Pseudoinverse Reconstruction
% *SVD decomposition:*
%
% $$A = U\Sigma V^T$$
%
% *Naive inversion:*
%
% $$x^{\dagger} = V\Sigma^{-1}U^T y$$

[U, S_mat, V] = svd(A, 'econ');
s = diag(S_mat);

x_pinv = V * (U' * y ./ s);
x_pinv_img = reshape(x_pinv, n, n);

figure;
subplot(1,2,1);
imshow(x_pinv_img, []);
title('Pseudoinverse Reconstruction (FAILS)');

subplot(1,2,2);
semilogy(s, 'o-');
xlabel('Index');
ylabel('Singular value');
title('Singular Value Decay of A');
grid on;

%% 5. Error Analysis
mse_pinv = mean((x_pinv - x_vec).^2);
rel_err = norm(x_pinv - x_vec) / norm(x_vec);

fprintf('\n%s\n', repmat('=', 1, 50));
fprintf('PSEUDOINVERSE RECONSTRUCTION ERROR ANALYSIS\n');
fprintf('%s\n', repmat('=', 1, 50));
fprintf('Relative Error: %.6f\n', rel_err);
fprintf('MSE: %.8f\n', mse_pinv);
fprintf('Condition number: %.2e\n', s(1)/s(end));
fprintf('Image size: %dx%d\n', n, n);
fprintf('Kernel: %dx%d, sigma=%.1f\n', kernel_size, kernel_size, sigma_k);
fprintf('Smallest singular value: %.2e\n', s(end));
fprintf('Largest singular value: %.2e\n', s(1));
