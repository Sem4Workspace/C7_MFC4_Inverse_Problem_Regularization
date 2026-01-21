import numpy as np
from numpy.linalg import svd, norm
from reconstruction.spectral_filters import tikhonov_filter


def l_curve(A: np.ndarray, y: np.ndarray, lambdas: np.ndarray):
    """Compute L-curve data (residual and solution norms) for regularization parameter selection."""
    U, s, Vt = svd(A, full_matrices=False)
    residual_norms = []
    solution_norms = []
    for lam in lambdas:
        filt = tikhonov_filter(s, lam)
        x_hat = (Vt.T * filt) @ (U.T @ y)
        residual_norms.append(norm(A @ x_hat - y))
        solution_norms.append(norm(x_hat))
    return np.array(residual_norms), np.array(solution_norms)
