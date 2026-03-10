%% Forward Operator — Image Degradation Model
% This script demonstrates the forward model for image deblurring.
%
% *The Forward Model:*
%
% $$y = A\,x + \varepsilon$$
%
% where $A$ is the blur operator (convolution with a Gaussian PSF) and
% $\varepsilon \sim \mathcal{N}(0,\sigma^2 I)$ is additive noise.
%
% *Gaussian Point Spread Function (PSF):*
%
% $$h(i,j) = \frac{1}{Z}\exp\!\left(-\frac{i^2+j^2}{2\sigma_k^2}\right), \quad Z = \sum_{i,j}h(i,j)$$
%
% *Convolution as matrix-vector product:*
%
% $$y = h * x + \varepsilon$$
%
% In matrix form: $A\,\text{vec}(x)$ where $A$ is block-Toeplitz.
%
% *Noise model:*
%
% $$\varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2), \quad \delta = \|\varepsilon\|_F$$

%% 1. Load Image
img = imread('296059.jpg');
if ndims(img) == 3
    img = rgb2gray(img);
end
x = double(img);
x = x / max(x(:));

[rows, cols] = size(x);
fprintf('Image size: %d x %d\n', rows, cols);

%% 2. Define Gaussian Kernel
kernel_size = 9;
sigma_k = 2.0;

half = floor(kernel_size / 2);
[xx, yy] = meshgrid(-half:half, -half:half);
kernel = exp(-(xx.^2 + yy.^2) / (2 * sigma_k^2));
kernel = kernel / sum(kernel(:));

%% 3. Forward Blur + Noise
noise_std = 0.01;
rng(42);

Ax = imfilter(x, kernel, 'symmetric', 'same');
eps = noise_std * randn(rows, cols);
y = Ax + eps;

%% 4. Visualisation
figure;
subplot(2,3,1); imshow(x, []);   title('Clean Image x');
subplot(2,3,2); imshow(Ax, []);  title('Blurred Image Ax');
subplot(2,3,3); imshow(eps, []); title('Noise \epsilon');
subplot(2,3,4); imshow(y, []);   title('Observed y = Ax + \epsilon');
subplot(2,3,5); imagesc(kernel); colormap(hot); colorbar;
                title('Gaussian Kernel (PSF)');
subplot(2,3,6); histogram(eps(:), 60, 'Normalization', 'pdf');
                title('Noise Distribution'); xlabel('Amplitude');

%% 5. Analysis
noise_norm = norm(eps(:));
signal_norm = norm(Ax(:));
relative_noise = noise_norm / signal_norm;

fprintf('\n%s\n', repmat('=', 1, 50));
fprintf('FORWARD MODEL ANALYSIS\n');
fprintf('%s\n', repmat('=', 1, 50));
fprintf('Image size: %dx%d\n', rows, cols);
fprintf('Kernel size: %dx%d\n', kernel_size, kernel_size);
fprintf('Kernel/Image ratio: %.2f (should be < 0.3)\n', kernel_size/rows);
fprintf('Noise norm: %.6f\n', noise_norm);
fprintf('Signal norm: %.6f\n', signal_norm);
fprintf('Relative noise level: %.6f\n', relative_noise);
