import numpy as np
from numpy.linalg import svd


def rank_deficient_matrix(n: int, rank: int) -> np.ndarray:
    U, _, Vt = svd(np.random.randn(n, n), full_matrices=False)
    s = np.linspace(1.0, 0.1, n)
    s[rank:] = 0.0
    return (U * s) @ Vt
