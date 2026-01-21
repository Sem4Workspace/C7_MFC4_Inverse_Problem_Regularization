import numpy as np
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    idx = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-0.5 * (idx / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def blur_matrix(n: int, sigma: float, kernel_radius: int = 10) -> np.ndarray:
    ksize = 2 * kernel_radius + 1
    k = gaussian_kernel(ksize, sigma)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            shift = (i - j) % n
            A[i, j] = k[shift] if shift < ksize else 0.0
    return A 
