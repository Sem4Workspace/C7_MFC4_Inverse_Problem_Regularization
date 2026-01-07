from data_input import load_image
from forward_model import forward_blur
from baseline_pseudoinverse import (
    build_convolution_matrix,
    pseudoinverse_reconstruction
)
from diagnostics import compute_diagnostics
import matplotlib.pyplot as plt
import numpy as np

# Load data
img = load_image()

# Select small patch
patch = img[100:116, 100:116]

# Forward model
y_img, kernel = forward_blur(patch)
# --- Visualize blur kernel (forward operator PSF) ---
plt.figure()
plt.imshow(kernel, cmap='hot')
plt.colorbar()
plt.title("Gaussian Blur Kernel (Forward Operator)")
plt.show()


from scipy.signal import convolve2d

# --- Compute noise explicitly ---
blurred_clean = convolve2d(patch, kernel, mode='same', boundary='symm')
noise = y_img - blurred_clean

# --- Visualize noise ε ---
plt.figure()
plt.imshow(noise, cmap='gray')
plt.colorbar()
plt.title("Additive Noise ε")
plt.show()


# --- Noise distribution ---
plt.figure()
plt.hist(noise.flatten(), bins=50, density=True)
plt.title("Noise Distribution (ε)")
plt.xlabel("Amplitude")
plt.ylabel("Density")
plt.show()


# Build system
A = build_convolution_matrix(kernel, patch.shape[0])
# --- Visualize forward operator matrix A ---
plt.figure(figsize=(6,6))
plt.imshow(A, cmap='gray')
plt.colorbar()
plt.title("Forward Operator Matrix A")
plt.show()


# Baseline pseudoinverse
x_hat, S = pseudoinverse_reconstruction(A, y_img.flatten())
# --- Singular value decay (ill-posedness proof) ---
plt.figure()
plt.semilogy(S, 'o-')
plt.xlabel("Index")
plt.ylabel("Singular value (log scale)")
plt.title("Singular Value Decay of A")
plt.grid(True)
plt.show()


# Diagnostics
diag = compute_diagnostics(S)
print("Diagnostics:", diag)

# Visualize
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(patch, cmap='gray')
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(y_img, cmap='gray')
plt.title("Blurred + Noise")

plt.subplot(1,3,3)
plt.imshow(x_hat.reshape(patch.shape), cmap='gray')
plt.title("Pseudoinverse")

plt.show()



# --- Summary visualization ---
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(patch, cmap='gray')
plt.title("Clean Signal x")

plt.subplot(1,3,2)
plt.imshow(y_img, cmap='gray')
plt.title("Blurred + Noise y")

plt.subplot(1,3,3)
plt.imshow(noise, cmap='gray')
plt.title("Noise ε")

plt.show()
