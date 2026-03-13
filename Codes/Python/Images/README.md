# Inverse Problems & Regularization

A comprehensive study of regularization techniques for solving ill-posed inverse problems, with applications to image deblurring.

## Overview

This project explores why naive solutions (pseudoinverse) fail for ill-posed inverse problems and demonstrates several modern regularization approaches:

- **Notebook 0**: Conceptual foundation — why the pseudoinverse is unstable
- **Notebook 1**: Forward problem setup and data generation
- **tikhonov.ipynb**: Tikhonov regularization with parameter selection
- **tsvd.ipynb**: Truncated Singular Value Decomposition (TSVD)
- **nsit_morozpv.ipynb**: Nested Semi-Iterative Tikhonov (NSIT) with Morozov discrepancy principle
- **fnsit.ipynb**: Fast Nested Semi-Iterative Tikhonov (NSIT) with Morozov discrepancy principle
- **llm_image_regularization.ipynb**: Uses LLMguidance to get regularization parameters.

## Problem Formulation

We solve the inverse problem:
```
y = A x + ε
```

where:
- `x` is the true image (unknown)
- `A` is the blur operator (forward model)
- `y` is the observed noisy image
- `ε` is measurement noise

The forward operator `A` is implemented as a 2D Gaussian convolution (point spread function).

## Methods

### 1. Naive Pseudoinverse (Unstable)
```
x_pinv = A^+ y
```
Amplifies noise due to small singular values → **fails on ill-posed problems**.

### 2. Tikhonov Regularization
```
x_λ = argmin ||Ax - y||² + λ||x||²
```
Gradient descent solution with penalty parameter `λ`. Balances data fidelity and solution smoothness.

### 3. Truncated SVD (TSVD)
```
x_k = Σ(i=1 to k) (u_i^T y / σ_i) v_i
```
Ignores small singular values by truncating at index `k`. Effective but requires manual parameter selection.

### 4. Nested Semi-Iterative Tikhonov (NSIT) + Morozov
Combines:
- Inner iterations with decaying Tikhonov parameter
- Morozov discrepancy principle for stopping: `||Ax - y|| ≈ τ·δ`

where `δ` is the noise level and `τ ≈ 1.05` is a safety factor.

### 4. Fast Nested Semi-Iterative Tikhonov (FNSIT) + Morozov
Faster than NSIT reconstruction

### 6. LLM Guided Regularization Parameters
Uses Gemini API Key to get regularization parameters from LLM.
## Key Findings 

- **Ill-posedness**: Small singular values cause noise amplification
- **Regularization**: Essential to stabilize solutions
- **Parameter tuning**: Critical—too small → noise dominates; too large → over-smoothing
- **Morozov principle**: Automatically balances fitting and regularization

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**:
   - Place test image at `images/296059.jpg`
   - Adjust image path in notebooks if needed

3. **Run notebooks** in sequence:
   ```
   Notebook 0 → Notebook 1 → {tikhonov, tsvd, nsit_morozpv}
   ```

4. **Modify parameters** as needed:
   - Noise level: `noise_std`
   - Kernel size & sigma: `gaussian_kernel()`
   - Regularization parameter: `lam`, `alpha0`, `q`

## File Structure

```
inverse-problems-regularization/
images/
      ├── requirements.txt          # Python dependencies
      ├── README.md                 # This file
      ├── notebook0.ipynb           # Phase 0: Pseudoinverse instability
      ├── notebook1.ipynb           # Phase 1: Forward problem setup
      ├── tikhonov.ipynb            # Tikhonov regularization
      ├── tsvd.ipynb                # Truncated SVD
      ├── nsit_morozpv.ipynb        # NSIT + Morozov discrepancy
      ├── fnsi.ipynb                # FNSIT Reconstruction
      ├── llm_image_reconstruction.ipynb  # LLM-Guided Regularization Parameters
      └──── 296059.jpg            # Test image
```

## Dependencies

- **numpy**: Numerical computations
- **scipy**: Signal processing (convolution, SVD)
- **matplotlib**: Visualization
- **jupyter/ipython**: Interactive notebooks

## References

- Tikhonov, A. N. (1963). "On the solution of incorrectly stated problems..."
- Morozov, V. A. (1984). "Methods for Solving Incorrectly Posed Problems"
- Hansen, P. C. (2010). "Discrete Inverse Problems: Insight and Algorithms"

## Notes

- All experiments use 16×16 patches for explicit matrix construction
- Full-image deblurring uses implicit operators (convolution)
- Results depend on noise level, kernel parameters, and regularization strength

---

**Course**: Semester 4 - Mathematics for computing 4 | **Topic**: Inverse Problems & Regularization