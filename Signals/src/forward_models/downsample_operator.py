import numpy as np


def downsample_matrix(n: int, factor: int = 2) -> np.ndarray:
    m = n // factor
    A = np.zeros((m, n))
    for i in range(m):
        A[i, i * factor] = 1.0
    return A
