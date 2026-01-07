"""
Phase 1 + Baseline Inversion
---------------------------
1. Load full-size user image
2. Apply forward operator y = A x + ε (operator form)
3. Diagnostics (noise stats)
4. Extract patch
5. Build explicit A for patch
6. SVD + pseudoinverse reconstruction
7. Visualize reconstruction failure

This establishes ill-posedness BEFORE regularization.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from numpy.linalg import svd


# ---------------------------------------------------
# Gaussian kernel (PSF)
# ---------------------------------------------------
def gaussian_kernel(size=9, sigma=2.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


# ---------------------------------------------------
# Forward operator (implicit A)
# ---------------------------------------------------
def forward_blur(image, kernel, noise_std=0.01):
    blurred = convolve2d(image, kernel, mode="same", boundary="symm")
    noise = noise_std * np.random.randn(*blurred.shape)
    noisy = blurred + noise
    return blurred, noise, noisy


# ---------------------------------------------------
# Build explicit convolution matrix A (PATCH ONLY)
# ---------------------------------------------------
def build_convolution_matrix(kernel, patch_size):
    k = kernel.shape[0]
    pad = k // 2
    N = patch_size * patch_size
    A = np.zeros((N, N))

    for i in range(N):
        basis = np.zeros((patch_size, patch_size))
        basis.flat[i] = 1.0
        conv = convolve2d(basis, kernel, mode="same", boundary="symm")
        A[:, i] = conv.flatten()

    return A


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------------------
    # 1. Load FULL image
    # ---------------------------------------------------
    image_path = r"C:\Desktop\Sem 4\inverse-problems-regularization\data\BSDS300\images\test\296059.jpg"   # <-- change path
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Image not found")

    img = img.astype(np.float64) / 255.0
    x_full = img


    # ---------------------------------------------------
    # 2. Forward model (FULL image)
    # ---------------------------------------------------
    kernel = gaussian_kernel(size=9, sigma=2.0)
    Ax_full, eps_full, y_full = forward_blur(x_full, kernel, noise_std=0.01)


    # ---------------------------------------------------
    # 3. Forward model visualization
    # ---------------------------------------------------
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(x_full, cmap="gray")
    plt.title("Clean Image x")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(Ax_full, cmap="gray")
    plt.title("Blurred Image Ax")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(eps_full, cmap="gray")
    plt.title("Noise ε")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(y_full, cmap="gray")
    plt.title("Observed Image y = Ax + ε")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(kernel, cmap="hot")
    plt.title("Gaussian Kernel (PSF)")
    plt.colorbar(fraction=0.046)

    plt.subplot(2, 3, 6)
    plt.hist(eps_full.flatten(), bins=60, density=True)
    plt.title("Noise Distribution")
    plt.xlabel("Amplitude")

    plt.tight_layout()
    plt.show()


    # ---------------------------------------------------
    # 4. Diagnostics
    # ---------------------------------------------------
    print("\n===== Forward Model Diagnostics =====")
    print(f"Full image size      : {x_full.shape}")
    print(f"Kernel size          : {kernel.shape}")
    print(f"Noise std (measured) : {np.std(eps_full):.5f}")


    # ===================================================
    # BASELINE INVERSE PROBLEM (PATCH-BASED)
    # ===================================================

    # ---------------------------------------------------
    # 5. Extract patch
    # ---------------------------------------------------
    patch_size = 16
    x_patch = x_full[100:100+patch_size, 100:100+patch_size]
    y_patch = y_full[100:100+patch_size, 100:100+patch_size]

    x_vec = x_patch.flatten()
    y_vec = y_patch.flatten()


    # ---------------------------------------------------
    # 6. Build explicit A for patch
    # ---------------------------------------------------
    A = build_convolution_matrix(kernel, patch_size)


    # ---------------------------------------------------
    # 7. SVD of A
    # ---------------------------------------------------
    U, S, Vt = svd(A, full_matrices=False)
    cond_number = S[0] / S[-1]

    print("\n===== Inverse Problem Diagnostics =====")
    print(f"A matrix size        : {A.shape}")
    print(f"Condition number    : {cond_number:.2e}")
    print(f"Smallest singular   : {S[-1]:.2e}")


    # ---------------------------------------------------
    # 8. Pseudoinverse reconstruction
    # ---------------------------------------------------
    S_inv = np.diag(1.0 / S)
    x_hat = Vt.T @ S_inv @ U.T @ y_vec
    x_hat_img = x_hat.reshape(patch_size, patch_size)


    # ---------------------------------------------------
    # 9. Reconstruction visualization
    # ---------------------------------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(x_patch, cmap="gray")
    plt.title("Clean Patch x")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(y_patch, cmap="gray")
    plt.title("Blurred + Noisy Patch y")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(x_hat_img, cmap="gray")
    plt.title("Pseudoinverse Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


    # ---------------------------------------------------
    # 10. Singular value decay plot (ILL-POSEDNESS PROOF)
    # ---------------------------------------------------
    plt.figure()
    plt.semilogy(S, 'o-')
    plt.xlabel("Index")
    plt.ylabel("Singular value (log scale)")
    plt.title("Singular Value Decay of A")
    plt.grid(True)
    plt.show()


    # ---------------------------------------------------
    # 11. Reconstruction error
    # ---------------------------------------------------
    rel_error = np.linalg.norm(x_patch - x_hat_img) / np.linalg.norm(x_patch)
    print(f"Relative reconstruction error: {rel_error:.4f}")
