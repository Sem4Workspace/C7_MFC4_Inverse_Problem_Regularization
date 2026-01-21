"""
Non-Stationary Iterated Tikhonov (NSIT) Regularization

NSIT improves the solution step by step using regularized corrections
of the residual, where the regularization strength decreases over iterations,
allowing stable recovery of low-frequency components first and gradually
introducing finer details while controlling noise.

Mathematical Foundation:

NSIT update rule:
    x_n = x_{n-1} + F_n * r_{n-1}

where:
    r_{n-1} = y_δ - A*x_{n-1}  (residual)
    F_n = (A^T*A + α_n*I)^{-1} * A^T  (regularized inverse)

Key insight: 
    - Early iterations (large α_n): Suppress noise, recover smooth components
    - Late iterations (small α_n): Add detail while controlling amplification
    - Iteration count = regularization parameter via Morozov stopping rule
"""

import numpy as np
from numpy.linalg import svd, norm


def reconstruct_nsit(A: np.ndarray, y: np.ndarray, alpha_schedule, 
                     stopping_rule='morozov', noise_level: float = None,
                     tau: float = 1.0, max_iter: int = 100,
                     return_history: bool = False):
    """
    Non-Stationary Iterated Tikhonov (NSIT) reconstruction.
    
    Iteratively refines solution using regularized corrections with
    decreasing regularization parameter.
    
    Parameters:
    -----------
    A : np.ndarray
        Forward operator matrix (m × n)
    y : np.ndarray
        Measurement vector (length m)
    alpha_schedule : callable or array
        Regularization parameters for each iteration
        If callable: alpha_n = alpha_schedule(n)
        If array: alpha_n = alpha_schedule[n]
    stopping_rule : str, default='morozov'
        'morozov' - Stop when ||y - Ax_n|| ≈ τ*δ*||y||
        'none' - Run for all iterations (use max_iter)
        'relative_error' - Stop when error plateaus (requires y_true)
    noise_level : float, optional
        Noise level δ (required for 'morozov' stopping)
    tau : float, default=1.0
        Safety factor for Morozov principle
    max_iter : int, default=100
        Maximum number of iterations
    return_history : bool, default=False
        If True, return (solution, history_dict)
        
    Returns:
    --------
    x : np.ndarray
        Reconstructed solution
    history : dict (optional)
        Contains 'x': solution at each iteration
                'residuals': ||y - Ax_n||
                'errors': ||x_n - x||  (if available)
                'alphas': regularization parameters used
                'stopping_iter': iteration where stopped
    """
    m, n = A.shape
    x = np.zeros(n)
    
    history = {
        'x': [x.copy()],
        'residuals': [],
        'alphas': []
    }
    
    # Set up stopping criterion
    if stopping_rule == 'morozov' and noise_level is None:
        raise ValueError("noise_level required for Morozov stopping rule")
    
    if stopping_rule == 'morozov':
        target_residual = tau * noise_level * norm(y)
    
    # Precompute A^T and A^T*y for efficiency
    AtA = A.T @ A
    Aty = A.T @ y
    
    for iter_count in range(max_iter):
        # Get regularization parameter for this iteration
        if callable(alpha_schedule):
            alpha_n = alpha_schedule(iter_count)
        else:
            if iter_count >= len(alpha_schedule):
                alpha_n = alpha_schedule[-1]  # Use last value if array exhausted
            else:
                alpha_n = alpha_schedule[iter_count]
        
        # Compute residual
        r = y - A @ x
        residual_norm = norm(r)
        
        # Update: x_n = x_{n-1} + (A^T*A + α_n*I)^{-1} * A^T * r
        correction = np.linalg.solve(AtA + alpha_n * np.eye(n), A.T @ r)
        x = x + correction
        
        # Store history
        history['residuals'].append(residual_norm)
        history['alphas'].append(alpha_n)
        history['x'].append(x.copy())
        
        # Check stopping criteria
        if stopping_rule == 'morozov' and residual_norm <= target_residual:
            history['stopping_iter'] = iter_count
            break
    else:
        history['stopping_iter'] = max_iter - 1
    
    if return_history:
        return x, history
    return x


def decreasing_schedule(alpha_0: float, rate: str = 'sqrt'):
    """
    Generate decreasing alpha schedule.
    
    Parameters:
    -----------
    alpha_0 : float
        Initial regularization parameter
    rate : str
        'sqrt' - α_n = α_0 / sqrt(n+1)
        'linear' - α_n = α_0 / (n+1)
        'exp' - α_n = α_0 * 0.9^n
        'power' - α_n = α_0 / (n+1)^1.5
        
    Returns:
    --------
    schedule : callable
        Function that takes iteration number n and returns α_n
    """
    schedules = {
        'sqrt': lambda n: alpha_0 / np.sqrt(n + 1),
        'linear': lambda n: alpha_0 / (n + 1),
        'exp': lambda n: alpha_0 * (0.9 ** n),
        'power': lambda n: alpha_0 / ((n + 1) ** 1.5),
        'log': lambda n: alpha_0 / np.log(n + 2),
    }
    
    if rate not in schedules:
        raise ValueError(f"rate must be one of {list(schedules.keys())}")
    
    return schedules[rate]


def nsit_with_morozov(A: np.ndarray, y: np.ndarray, noise_level: float,
                      alpha_0: float = None, schedule_type: str = 'sqrt',
                      tau: float = 1.0, max_iter: int = 100):
    """
    NSIT with automatic stopping using Morozov discrepancy principle.
    
    Stop when: ||y - Ax_n|| ≈ τ*δ*||y||
    
    Parameters:
    -----------
    A : np.ndarray
        Forward operator
    y : np.ndarray
        Measurements
    noise_level : float
        Relative noise level δ
    alpha_0 : float, optional
        Initial regularization (default: auto-compute)
    schedule_type : str
        Type of decreasing schedule
    tau : float
        Safety factor (typically 1.0-1.2)
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    x : np.ndarray
        Reconstructed solution
    history : dict
        Convergence history
    """
    # Auto-select initial alpha if not provided
    if alpha_0 is None:
        U, s, _ = svd(A, full_matrices=False)
        alpha_0 = s.max() ** 2  # Start with largest singular value squared
    
    schedule = decreasing_schedule(alpha_0, rate=schedule_type)
    
    x, history = reconstruct_nsit(
        A, y, schedule,
        stopping_rule='morozov',
        noise_level=noise_level,
        tau=tau,
        max_iter=max_iter,
        return_history=True
    )
    
    return x, history


def nsit_with_lcurve(A: np.ndarray, y: np.ndarray, 
                     alpha_0: float = None, schedule_type: str = 'sqrt',
                     max_iter: int = 100):
    """
    NSIT with L-curve stopping (when residual/solution norm ratio changes).
    
    Parameters:
    -----------
    A : np.ndarray
        Forward operator
    y : np.ndarray
        Measurements
    alpha_0 : float, optional
        Initial regularization
    schedule_type : str
        Type of schedule
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    x : np.ndarray
        Reconstructed solution
    history : dict
        With 'L_curve_point' indicating where stopped
    """
    if alpha_0 is None:
        U, s, _ = svd(A, full_matrices=False)
        alpha_0 = s.max() ** 2
    
    schedule = decreasing_schedule(alpha_0, rate=schedule_type)
    
    x, history = reconstruct_nsit(
        A, y, schedule,
        stopping_rule='none',
        max_iter=max_iter,
        return_history=True
    )
    
    # Find L-curve corner (maximum curvature)
    if len(history['residuals']) > 2:
        residuals = np.array(history['residuals'])
        solution_norms = [norm(xi) for xi in history['x']]
        
        # Compute curvature
        log_res = np.log10(residuals)
        log_norms = np.log10(solution_norms)
        
        # Simple curvature: second derivative
        d1 = np.diff(log_res)
        d2_approx = np.diff(d1)
        
        if len(d2_approx) > 0:
            corner_idx = np.argmax(np.abs(d2_approx)) + 1
            x = history['x'][corner_idx]
            history['stopping_iter'] = corner_idx
    
    return x, history


def spectral_view_nsit(A: np.ndarray, y: np.ndarray, true_x: np.ndarray = None,
                       alpha_0: float = None, schedule_type: str = 'sqrt',
                       max_iter: int = 20):
    """
    NSIT reconstruction with spectral (SVD) analysis.
    
    Returns solution plus spectral filter evolution showing how
    different frequency components are recovered over iterations.
    
    Parameters:
    -----------
    A : np.ndarray
        Forward operator
    y : np.ndarray
        Measurements  
    true_x : np.ndarray, optional
        True solution (for error analysis)
    alpha_0 : float, optional
        Initial regularization
    schedule_type : str
        Schedule type
    max_iter : int
        Number of iterations to track
        
    Returns:
    --------
    x : np.ndarray
        Final reconstruction
    history : dict
        With 'spectral_filters': filter factors over iterations
                'singular_values': σ_i of A
                'errors': relative error at each step
    """
    if alpha_0 is None:
        U, s, _ = svd(A, full_matrices=False)
        alpha_0 = s.max() ** 2
    
    U, s, Vt = svd(A, full_matrices=False)
    
    schedule = decreasing_schedule(alpha_0, rate=schedule_type)
    
    x, history = reconstruct_nsit(
        A, y, schedule,
        stopping_rule='none',
        max_iter=max_iter,
        return_history=True
    )
    
    # Compute spectral filters
    spectral_filters = []
    errors_per_iter = []
    
    for n in range(len(history['alphas'])):
        alphas_used = history['alphas'][:n+1]
        
        # For each singular value, compute cumulative filter
        filter_n = np.ones_like(s)
        for alpha_j in alphas_used:
            filter_n *= alpha_j / (s**2 + alpha_j)
        
        # Recovery filter: 1 - filter_n (how much is recovered, not damped)
        recovery = 1 - filter_n
        spectral_filters.append(recovery)
        
        if true_x is not None:
            error = np.linalg.norm(true_x - history['x'][n+1]) / np.linalg.norm(true_x)
            errors_per_iter.append(error)
    
    history['spectral_filters'] = np.array(spectral_filters)
    history['singular_values'] = s
    if errors_per_iter:
        history['errors'] = np.array(errors_per_iter)
    
    return x, history
