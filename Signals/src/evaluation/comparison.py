import numpy as np
from typing import Dict, List, Optional
from reconstruction import pseudoinverse, tikhonov, tsvd, nsit
from evaluation.error_metrics import mse, psnr, relative_error


def compare_methods(
    A: np.ndarray,
    y: np.ndarray,
    x_true: np.ndarray,
    tikh_lambdas: List[float],
    tsvd_ks: List[int],
    nsit_strategies: Optional[List[str]] = None,
    nsit_max_iters: int = 50,
) -> Dict:
    """
    Compare multiple regularization methods.
    
    Parameters
    ----------
    A : np.ndarray
        Forward operator
    y : np.ndarray
        Measurements
    x_true : np.ndarray
        True solution
    tikh_lambdas : list
        Tikhonov regularization parameters to test
    tsvd_ks : list
        TSVD truncation values to test
    nsit_strategies : list, optional
        NSIT parameter adaptation strategies to test
    nsit_max_iters : int
        Maximum iterations for NSIT
        
    Returns
    -------
    results : dict
        Comprehensive comparison results
    """
    if nsit_strategies is None:
        nsit_strategies = ["residual_norm", "hybrid"]
    
    results = {
        "pseudoinverse": {},
        "tikhonov": [],
        "tsvd": [],
        "nsit": [],
    }
    
    # Pseudoinverse baseline
    x_pinv = pseudoinverse.reconstruct(A, y)
    results["pseudoinverse"] = {
        "mse": mse(x_true, x_pinv),
        "psnr": psnr(x_true, x_pinv),
        "rel_error": relative_error(x_true, x_pinv),
    }
    
    # Tikhonov with different lambdas
    for lam in tikh_lambdas:
        x_hat = tikhonov.reconstruct(A, y, lam)
        results["tikhonov"].append({
            "lambda": float(lam),
            "mse": mse(x_true, x_hat),
            "psnr": psnr(x_true, x_hat),
            "rel_error": relative_error(x_true, x_hat),
        })
    
    # TSVD with different truncation values
    for k in tsvd_ks:
        x_hat = tsvd.reconstruct(A, y, k)
        results["tsvd"].append({
            "k": int(k),
            "mse": mse(x_true, x_hat),
            "psnr": psnr(x_true, x_hat),
            "rel_error": relative_error(x_true, x_hat),
        })
    
    # NSIT with different strategies
    for strategy in nsit_strategies:
        x_hat, history = nsit.reconstruct(
            A,
            y,
            alpha_strategy=strategy,
            max_iters=nsit_max_iters,
            verbose=False,
        )
        results["nsit"].append({
            "strategy": strategy,
            "mse": mse(x_true, x_hat),
            "psnr": psnr(x_true, x_hat),
            "rel_error": relative_error(x_true, x_hat),
            "iterations": history["iterations"],
            "converged": history["converged"],
            "final_residual": history["residual_norms"][-1],
            "final_alpha": history["alphas"][-1],
        })
    
    return results


def compare_methods_extended(
    A: np.ndarray,
    y: np.ndarray,
    x_true: np.ndarray,
    tikh_lambdas: List[float],
    tsvd_ks: List[int],
    nsit_alpha_inits: Optional[List[float]] = None,
    nsit_strategies: Optional[List[str]] = None,
    nsit_max_iters: int = 50,
    return_solutions: bool = False,
) -> Dict:
    """
    Extended comparison with detailed NSIT analysis.
    
    Parameters
    ----------
    A : np.ndarray
        Forward operator
    y : np.ndarray
        Measurements
    x_true : np.ndarray
        True solution
    tikh_lambdas : list
        Tikhonov lambdas
    tsvd_ks : list
        TSVD k values
    nsit_alpha_inits : list, optional
        Initial alpha values for NSIT
    nsit_strategies : list, optional
        NSIT strategies
    nsit_max_iters : int
        Max iterations
    return_solutions : bool
        Include reconstructed solutions
        
    Returns
    -------
    results : dict
        Extended results with detailed histories
    """
    if nsit_alpha_inits is None:
        nsit_alpha_inits = [1.0]
    if nsit_strategies is None:
        nsit_strategies = ["residual_norm", "hybrid"]
    
    results = {
        "pseudoinverse": {},
        "tikhonov": [],
        "tsvd": [],
        "nsit": [],
    }
    
    # Pseudoinverse
    x_pinv = pseudoinverse.reconstruct(A, y)
    results["pseudoinverse"] = {
        "mse": mse(x_true, x_pinv),
        "psnr": psnr(x_true, x_pinv),
        "rel_error": relative_error(x_true, x_pinv),
    }
    if return_solutions:
        results["pseudoinverse"]["solution"] = x_pinv
    
    # Tikhonov
    for lam in tikh_lambdas:
        x_hat = tikhonov.reconstruct(A, y, lam)
        entry = {
            "lambda": float(lam),
            "mse": mse(x_true, x_hat),
            "psnr": psnr(x_true, x_hat),
            "rel_error": relative_error(x_true, x_hat),
        }
        if return_solutions:
            entry["solution"] = x_hat
        results["tikhonov"].append(entry)
    
    # TSVD
    for k in tsvd_ks:
        x_hat = tsvd.reconstruct(A, y, k)
        entry = {
            "k": int(k),
            "mse": mse(x_true, x_hat),
            "psnr": psnr(x_true, x_hat),
            "rel_error": relative_error(x_true, x_hat),
        }
        if return_solutions:
            entry["solution"] = x_hat
        results["tsvd"].append(entry)
    
    # NSIT with different settings
    for alpha_init in nsit_alpha_inits:
        for strategy in nsit_strategies:
            x_hat, history = nsit.reconstruct(
                A,
                y,
                alpha_init=alpha_init,
                alpha_strategy=strategy,
                max_iters=nsit_max_iters,
                verbose=False,
            )
            
            entry = {
                "alpha_init": float(alpha_init),
                "strategy": strategy,
                "mse": mse(x_true, x_hat),
                "psnr": psnr(x_true, x_hat),
                "rel_error": relative_error(x_true, x_hat),
                "iterations": history["iterations"],
                "converged": history["converged"],
                "final_residual": history["residual_norms"][-1],
                "final_alpha": history["alphas"][-1],
                "history": history,
            }
            if return_solutions:
                entry["solution"] = x_hat
            
            results["nsit"].append(entry)
    
    return results
