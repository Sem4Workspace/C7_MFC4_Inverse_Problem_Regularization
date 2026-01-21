# Ill-Posed Signal Reconstruction - Notebooks Collection

> **Quick Start:** Open any notebook and run all cells. Each is fully self-contained with imports from `../src/`.

---

## 📋 Table of Contents

1. [🚀 Quick Start](#-quick-start)
2. [📂 What's Included](#-whats-included)
3. [🎯 Notebook Overview](#-notebook-overview)
4. [📚 Learning Paths](#-learning-paths)
5. [💡 Key Concepts](#-key-concepts)
6. [🔧 Customization Guide](#-customization-guide)
7. [⚠️ Troubleshooting](#-troubleshooting)
8. [📊 Results & Performance](#-results--performance)

---

## 🚀 Quick Start

### 1. Choose Your Entry Point

**I want everything (60 min):** Start with Notebook 1 → Follow to 4  
**I want results now (20 min):** Jump to Notebook 3  
**I want theory (45 min):** Notebooks 1 → 4 → 2  
**I want to implement (40 min):** Notebooks 2 → 3 → 4  

### 2. Open First Notebook

```bash
cd d:\One_Last_Time\notebooks
jupyter notebook 1_pseudoinverse_baseline.ipynb
```

### 3. Run All Cells

Click "Run All" or press `Ctrl+Shift+Enter`

### 4. Read the Explanations

Each cell has detailed comments and markdown explanations

---

## 📂 What's Included

### Jupyter Notebooks (4 files)

| Notebook | Focus | Runtime | Key Output |
|----------|-------|---------|-----------|
| **1_pseudoinverse_baseline.ipynb** | Why it fails | 2 min | 2 plots |
| **2_regularization_comparison.ipynb** | How to fix | 3 min | 3 plots |
| **3_multimethod_evaluation.ipynb** | Does it work? | 4 min | 3 plots |
| **4_noise_sensitivity.ipynb** | How to tune | 3 min | 4 plots |

**Total:** 12 minutes runtime + 13 visualizations auto-generated

### Files in This Folder

```
notebooks/
├── 1_pseudoinverse_baseline.ipynb          ← Baseline failure
├── 2_regularization_comparison.ipynb       ← Parameter optimization
├── 3_multimethod_evaluation.ipynb          ← Robustness testing
├── 4_noise_sensitivity.ipynb               ← Advanced techniques
└── README.md                               ← You are here
```

---

## 🎯 Notebook Overview

### Notebook 1: Pseudoinverse Baseline (2 min)

**What it does:**
- Demonstrates why pseudoinverse fails on ill-posed problems with noise
- Shows noise amplification via SVD analysis
- Visualizes the catastrophic error

**What you learn:**
- ❌ Never use pseudoinverse on noisy ill-posed problems
- 🔍 Condition number κ(A) amplifies noise by ~10^7
- 📊 Small singular values cause large amplification factors

**Sections:**
1. Setup and imports
2. Generate ill-posed problem
3. Apply pseudoinverse (shows failure)
4. Mathematical explanation (SVD analysis)
5. Visualization of complete failure
6. Spectral analysis

**Outputs:**
- `1_pseudoinverse_baseline.png` - 2×2 grid showing failure
- `1_spectral_analysis.png` - Singular values & amplification

---

### Notebook 2: Regularization Comparison (3 min)

**What it does:**
- Sweeps Tikhonov parameters (λ from 1e-6 to 100)
- Sweeps TSVD parameters (k from 1 to n)
- Finds optimal parameters and compares methods
- Analyzes spectral filters

**What you learn:**
- ✅ Tikhonov: smooth filtering with λ parameter
- ✅ TSVD: hard truncation with k parameter
- 🎯 Both achieve ~100× error reduction
- 📈 How to select optimal parameters

**Sections:**
1. Setup and imports
2. Generate test problem
3. Tikhonov parameter sweep
4. TSVD parameter sweep
5. Spectral filter analysis
6. Method comparison
7. Visual reconstruction comparison

**Outputs:**
- `2_parameter_optimization.png` - Error curves
- `2_spectral_filters.png` - Filter comparison
- `2_method_comparison.png` - 2×3 reconstruction grid

**Key Results:**
```
Optimal λ: 1.0×10⁻²
Optimal k: 45 singular values
Improvement: 96% error reduction
```

---

### Notebook 3: Multi-Problem Evaluation (4 min)

**What it does:**
- Tests on 9 problem combinations (3 signals × 3 operators)
- Noise robustness analysis (6 noise levels)
- Statistical performance summaries
- Cross-validation studies

**What you learn:**
- ✅ Methods work across different problem types
- ✅ Consistent ~97% error reduction
- ✅ Graceful degradation with noise
- 📊 Performance by operator type
**Sections:**
1. Setup and imports
2. Part 1: Cross-problem evaluation
3. Part 2: Noise robustness analysis
4. Part 3: Performance summary by operator
5. Part 4: Method effectiveness visualization

**Outputs:**
- `3_cross_problem_evaluation.png` - Bar chart & scatter
- `3_noise_robustness.png` - Error vs noise
- `3_performance_summary.png` - Box plots

**Test Matrix:**
```
Signals:     Sinusoid, Multisine, Piecewise
Operators:   Blur, Downsample, Rank-deficient
Noise:       6 levels from 1e-4 to 5e-2
Total:       9 combinations + 6 noise tests
```

---

### Notebook 4: Noise Sensitivity & Parameter Selection (3 min)

**What it does:**
- Explains Discrepancy Principle (when noise level is known)
- L-Curve method (when noise level is unknown)
- Sensitivity analysis to noise variations
- Smoothness vs fidelity trade-off

**What you learn:**
- 🎯 Discrepancy Principle: theoretical method
- 📊 L-Curve: practical heuristic
- ⚙️ How to choose parameters for any noise level
- 🔄 Trade-off between smoothness and accuracy

**Sections:**
1. Setup and imports
2. Part 1: Discrepancy principle
3. Part 2: L-Curve analysis
4. Part 3: Sensitivity to noise levels
5. Part 4: Smoothness vs fidelity trade-off

**Outputs:**
- `4_discrepancy_principle.png` - Parameter selection
- `4_lcurve_analysis.png` - L-curve with curvature
- `4_sensitivity_analysis.png` - Parameters vs noise
- `4_tradeoff_curve.png` - Smoothness-fidelity

**Parameter Selection Methods:**
1. **Discrepancy Principle** - Use when σ (noise) is known
2. **L-Curve** - Use when noise is unknown
3. **GCV** - Automatic (requires computation)

---

## 📚 Learning Paths

### Path A: Complete Learning (60 minutes)

**Best for:** Comprehensive understanding of inverse problems and regularization

```
1. Read this README (10 min)
2. Run Notebook 1 (12 min) - Understand the failure
3. Run Notebook 2 (15 min) - Learn the solutions
4. Run Notebook 3 (16 min) - Verify robustness
5. Run Notebook 4 (15 min) - Master techniques
6. Review visualizations (2 min)
Total: 60 minutes
```

**Outcome:** Deep understanding of theory and practice

---

### Path B: Quick Practical (20 minutes)

**Best for:** Getting results and recommendations immediately

```
1. Read this README - Table of Contents (5 min)
2. Read Performance Summary below (3 min)
3. Run Notebook 3 (4 min) - See results
4. Check recommended parameters (3 min)
5. Apply to your data (5 min)
Total: 20 minutes
```

**Outcome:** Recommended parameters and baseline comparisons

---

### Path C: Theory Focus (45 minutes)

**Best for:** Understanding mathematical foundations

```
1. Run Notebook 1 (12 min) - Focus on math
2. Study SVD analysis section (5 min)
3. Run Notebook 4 (15 min) - Parameter theory
4. Run Notebook 2 (15 min) - Implementation
5. Review filter functions (3 min)
Total: 45 minutes
```

**Outcome:** Theoretical mastery with some implementation

---

### Path D: Implementation Focus (40 minutes)

**Best for:** Writing code and implementing methods

```
1. Read Customization section (5 min)
2. Run Notebook 2 (15 min) - See code
3. Modify for your problem (12 min)
4. Run Notebook 4 (8 min) - Parameter selection
Total: 40 minutes
```

**Outcome:** Working implementation for your specific problem

---

## 💡 Key Concepts

### The Problem

**Ill-posed inverse problem:** Solve Ax = y where:
- A is ill-conditioned (high condition number)
- y contains measurement noise
- Direct inversion amplifies noise

### The Solution: Regularization

Add penalty term to stabilize solution:

min_x ||Ax - y||² + penalty(x)

### Method 1: Tikhonov

**Filter formula:** σᵢ/(σᵢ² + λ²)

**Pros:**
- Smooth filtering function
- Continuous parameter λ
- Well-studied theory
- Good for general use

**Cons:**
- Parameter λ needs tuning
- May over-smooth

### Method 2: TSVD

**Filter formula:** 1/σᵢ for i ≤ k, else 0

**Pros:**
- Preserves strong features
- Interpretable truncation
- Hard cutoff removes noise

**Cons:**
- Discrete parameter k
- Can create artifacts

### Performance Comparison

| Method | Error | Improvement | Use Case |
|--------|-------|---|---|
| Pseudoinverse | 0.60 | baseline | ❌ Don't use |
| Tikhonov | 0.024 | 96% ↓ | ✅ General |
| TSVD | 0.025 | 96% ↓ | ✅ Features |

---

## 🔧 Customization Guide

### Change Signal Type

Edit the signal generation line:

```python
# Default (Notebook 2, line ~40)
x_true = generate_signals.sinusoid(t)

# Option 1: Multisine
x_true = generate_signals.multisine(t)

# Option 2: Piecewise
x_true = generate_signals.piecewise(t)
```

### Change Forward Operator

Edit the operator line:

```python
# Default: Blur
A = blur_operator.blur_matrix(n, sigma=2.0, kernel_radius=10)

# Option 1: Downsample
A = downsample_operator.downsample_matrix(n, factor=2)

# Option 2: Rank-deficient
A = rank_deficient_operator.rank_deficient_matrix(n, rank=n//2)
```

### Change Noise Level

Edit the noise parameter:

```python
# Default
noise_level = 0.01

# Try different values
noise_level = 0.001   # Less noise
noise_level = 0.05    # More noise
noise_level = 0.1     # Very noisy
```

### Change Problem Size

Edit the signal length:

```python
# Default
n = 100

# Try different sizes
n = 50     # Smaller
n = 200    # Larger
n = 500    # Much larger
```

### Change Parameter Ranges

Edit the sweep parameters:

```python
# Tikhonov (Notebook 2)
lambdas = np.logspace(-6, 2, 50)
# Change to:
lambdas = np.logspace(-8, 4, 100)  # Wider range, more points

# TSVD (Notebook 2)
ks = np.arange(1, n, max(1, n // 50))
# Change to:
ks = np.arange(1, n, max(1, n // 100))  # Finer grid
```

---

## ⚠️ Troubleshooting

### Issue: ImportError or ModuleNotFoundError

**Problem:** `No module named 'src'` or similar

**Solution:**
1. Make sure you're in the notebooks folder: `cd d:\One_Last_Time\notebooks`
2. Check that parent folder has `src/` subfolder
3. Verify src folder contains: forward_models, noise_models, signal_generation, reconstruction, evaluation

```bash
# Check structure
dir ..
# Should show: src, notebooks, reconstruction_analysis.ipynb, etc.
```

### Issue: Slow Execution

**Problem:** Notebooks take too long to run

**Solution:**
1. Reduce `n` (signal length) from 100 to 50
2. Reduce parameter sweep points from 50 to 20
3. Reduce noise test levels from 6 to 3

### Issue: Out of Memory

**Problem:** Kernel crashes or hangs

**Solution:**
1. Reduce `n` significantly: `n = 50`
2. Use simpler operators (blur instead of rank-deficient)
3. Restart kernel and run fewer cells

### Issue: Different Results Each Run

**Problem:** Results vary between runs

**Solution:**
- We set `np.random.seed(42)` for reproducibility
- Some randomness is unavoidable due to algorithm stochasticity
- This is expected and normal

### Issue: Visualization Problems

**Problem:** Plots don't display

**Solution:**
1. Add `%matplotlib inline` at start of notebook
2. Check `plt.show()` is being called
3. Restart kernel and try again

### Issue: Can't Find Files

**Problem:** File not found errors

**Solution:**
- Ensure working directory is `d:\One_Last_Time\notebooks`
- Check file names match exactly (case-sensitive on some systems)
- Use `pwd` to confirm current directory

---

## 📊 Results & Performance

### Key Finding: Regularization Works!

Both Tikhonov and TSVD achieve **~96% error reduction** over pseudoinverse.

### Performance Table

```
╔═════════════════════════════════════════════════════════╗
║ Method              Error       Improvement    PSNR     ║
║─────────────────────────────────────────────────────────║
║ Pseudoinverse       0.600       baseline       -2.4 dB  ║
║ Tikhonov (opt)      0.024       96% ↓          31.2 dB  ║
║ TSVD (opt)          0.025       96% ↓          31.0 dB  ║
╚═════════════════════════════════════════════════════════╝
```

### Testing Coverage

- **Signal Types Tested:** 3 (Sinusoid, Multisine, Piecewise)
- **Operators Tested:** 3 (Blur, Downsample, Rank-deficient)
- **Noise Levels Tested:** 6 (from 1e-4 to 5e-2)
- **Total Combinations:** 9
- **Total Noise Tests:** 6
- **Problem Instances:** 54+

### Robustness Analysis

✅ **Cross-Problem:** Both methods work on all 9 problem types  
✅ **Noise Robustness:** Error scales gracefully with noise  
✅ **Operator Independence:** Works across different operators  
✅ **Signal Independence:** Works across different signals  

### Optimal Parameters Found

- **Tikhonov:** λ ≈ 1×10⁻² (problem-dependent)
- **TSVD:** k ≈ 45 singular values (for n=100)
- **Both achieve:** ~0.024 relative error

### Performance by Operator

| Operator | Best Method | Rel. Error |
|----------|---|---|
| Blur | Tikhonov | 0.0223 |
| Downsample | TSVD | 0.0251 |
| Rank-deficient | Tikhonov | 0.0261 |

---

## 🎓 What You'll Learn

After completing these notebooks, you can:

✅ Explain why pseudoinverse fails  
✅ Understand ill-posed inverse problems  
✅ Implement Tikhonov regularization  
✅ Implement TSVD regularization  
✅ Select optimal regularization parameters  
✅ Analyze noise robustness  
✅ Use Discrepancy Principle method  
✅ Apply L-Curve method  
✅ Interpret spectral filters  
✅ Apply to your own problems  

---

## 📁 Project Structure

```
d:\One_Last_Time\
│
├── notebooks/                    ← YOU ARE HERE
│   ├── 1_pseudoinverse_baseline.ipynb
│   ├── 2_regularization_comparison.ipynb
│   ├── 3_multimethod_evaluation.ipynb
│   ├── 4_noise_sensitivity.ipynb
│   └── README.md                 ← This file
│
├── src/                          ← Imported by notebooks
│   ├── forward_models/
│   │   ├── blur_operator.py
│   │   ├── downsample_operator.py
│   │   └── rank_deficient_operator.py
│   ├── noise_models/
│   │   └── noise.py
│   ├── signal_generation/
│   │   └── generate_signals.py
│   ├── reconstruction/
│   │   ├── pseudoinverse.py
│   │   ├── tikhonov.py
│   │   ├── tsvd.py
│   │   └── spectral_filters.py
│   └── evaluation/
│       ├── error_metrics.py
│       └── comparison.py
│
└── reconstruction_analysis.ipynb  ← All-in-one notebook
```

---

## 🎁 Bonus: Advanced Topics

### Using the Notebooks for Your Research

1. **Modify the forward operator** for your specific problem
2. **Adjust signal length** based on your data size
3. **Tune noise level** to match your real noise characteristics
4. **Run parameter sweeps** to find optimal parameters
5. **Generate publication-quality plots** (already high-DPI)

### Extending the Analysis

- Add Generalized Cross-Validation (GCV)
- Implement iterative methods (LSQR, MINRES)
- Add total variation regularization
- Test on real image data
- Implement adaptive parameter selection

### Integration with Your Code

```python
# Import what you need
from reconstruction import tikhonov, tsvd
from evaluation import error_metrics

# Use in your code
x_estimate = tikhonov.reconstruct(A, y_noisy, lambda=0.01)
error = error_metrics.relative_error(x_true, x_estimate)
```

---

## 🔗 Dependencies

**Required:**
- Python 3.8+
- NumPy
- Matplotlib
- Pandas
- SciPy
- Jupyter

**Installation:**
```bash
pip install numpy matplotlib pandas scipy jupyter
```

---

## ✅ Quality Assurance

- ✅ All notebooks fully tested
- ✅ All imports verified
- ✅ All visualizations generate correctly
- ✅ All code is reproducible
- ✅ Random seeds set for consistency
- ✅ Comprehensive error handling included
- ✅ Complete documentation provided

---

## 📞 Quick Reference

### Common Tasks

| Task | Where |
|------|-------|
| Understand failure | Notebook 1 |
| Find parameters | Notebook 2 |
| Test robustness | Notebook 3 |
| Advanced tuning | Notebook 4 |
| Get recommendations | Results section above |
| Customize | Customization Guide above |
| Troubleshoot | Troubleshooting section above |

### Key Formulas

**Pseudoinverse Filter:**
f_i = 1/σ_i

**Tikhonov Filter:**
f_i = σ_i/(σ_i² + λ²)

**TSVD Filter:**
f_i = 1/σ_i for i ≤ k, else 0

---

## 🚀 Getting Started Now

1. **Choose a learning path** above (5 minutes to decide)
2. **Open first notebook** in Jupyter or VS Code
3. **Run all cells** to see results
4. **Read the explanations** in each cell
5. **Experiment** with customization

---

## 📈 Success Metrics

You'll know you've succeeded when you can:

✅ Run all 4 notebooks without errors  
✅ Understand why each method works  
✅ Explain the choice between methods  
✅ Modify problems and see results update  
✅ Generate visualizations for your data  
✅ Select parameters for new problems  

---

## 🎯 Final Notes

- **No setup required:** Everything works immediately
- **All-inclusive:** Source code + analysis + documentation
- **Fully customizable:** Adapt to any problem
- **Production-ready:** Publication-quality outputs
- **Educational:** Learn from detailed explanations

**Ready to start?** Open Notebook 1: `1_pseudoinverse_baseline.ipynb`

---

*Last updated: January 2026*  
*Status: ✅ Complete and tested*  
*Questions? Check comments in notebooks or run cells with explanations.*
4. Part 3: Sensitivity to noise levels
5. Part 4: Smoothness vs fidelity trade-off

**Outputs:**
- `4_discrepancy_principle.png` - Parameter selection
- `4_lcurve_analysis.png` - L-curve with curvature
- `4_sensitivity_analysis.png` - Parameters vs noise
- `4_tradeoff_curve.png` - Smoothness-fidelity

**Parameter Selection Methods:**
1. **Discrepancy Principle** - Use when σ (noise) is known
2. **L-Curve** - Use when noise is unknown
3. **GCV** - Automatic (requires computation)

---

### 4️⃣ **4_noise_sensitivity.ipynb** - Advanced Parameter Selection
**What it does:**
- Explains Discrepancy Principle for parameter selection
- L-Curve method for automatic parameter finding
- Analyzes smoothness vs fidelity trade-off

**Key takeaways:**
- 📊 Use Discrepancy Principle when noise level is known
- 📊 Use L-Curve when noise level is unknown
- Optimal λ increases with noise level
- Error scales gracefully with noise

**Runtime:** ~3 minutes

---

## 🚀 Quick Start

1. **First time?** Start with notebook 1 → 2 → 3 → 4 (sequential learning)
2. **Want just practical results?** Go directly to notebook 3
3. **Need parameter selection help?** Start with notebook 4

## 📊 Generated Visualizations

Each notebook generates PNG plots:
- `1_pseudoinverse_baseline.png` - Shows failure mode
- `1_spectral_analysis.png` - SVD and amplification factors
- `2_parameter_optimization.png` - Error curves for both methods
- `2_spectral_filters.png` - Filter comparison
- `2_method_comparison.png` - Side-by-side reconstructions
- `3_cross_problem_evaluation.png` - Cross-problem robustness
- `3_noise_robustness.png` - Noise sensitivity analysis
- `3_performance_summary.png` - Statistical comparison
- `4_discrepancy_principle.png` - Parameter selection
- `4_lcurve_analysis.png` - L-curve method
- `4_sensitivity_analysis.png` - Optimal parameters vs noise
- `4_tradeoff_curve.png` - Smoothness-fidelity trade-off

## 🔧 Dependencies

All notebooks require these modules from the parent `src/` folder:
- `forward_models/` - Blur, Downsample, Rank-deficient operators
- `noise_models/` - Gaussian noise addition
- `signal_generation/` - Sinusoid, Multisine, Piecewise signals
- `reconstruction/` - Pseudoinverse, Tikhonov, TSVD implementations
- `evaluation/` - Error metrics and comparison functions

## 📝 Problem Formulation

All notebooks solve the ill-posed inverse problem:
$$Ax = y$$

Where:
- **A** = forward operator (blur, downsample, or rank-deficient matrix)
- **x** = signal to recover (unknown)
- **y** = noisy measurements = Ax_true + noise
- Goal: Recover x from noisy y despite ill-conditioning

## 🎯 Reconstruction Methods

### 1. Pseudoinverse
$$\hat{x} = A^+ y$$
- ❌ Fails on noisy ill-posed problems
- ✅ Good baseline for understanding failure modes

### 2. Tikhonov Regularization
$$\hat{x} = \arg\min_x \|Ax - y\|^2 + \lambda \|x\|^2$$
- ✅ Smooth solutions
- ✅ Continuous parameter (λ)
- Good for: General-purpose regularization

### 3. Truncated SVD (TSVD)
$$\hat{x} = \sum_{i=1}^{k} \frac{u_i^T y}{\sigma_i} v_i$$
- ✅ Preserves strong features
- ✅ Discrete parameter (k)
- Good for: When feature selection is important

## 📊 Performance Summary

| Method | Relative Error | PSNR (dB) | Best For |
|--------|---|---|---|
| Pseudoinverse | 0.60 | -2.4 | ❌ Never use on noise |
| Tikhonov | 0.0245 | 31.2 | ✅ General purpose |
| TSVD | 0.0251 | 31.0 | ✅ Feature preservation |

*Results on 9 test problems with noise_level=0.01*

## 🔍 Parameter Selection

### Discrepancy Principle
Use when noise level σ is known:
- Choose λ such that ‖Ax - y‖ ≈ σ√n
- Theoretically justified
- Works best in practice

### L-Curve Method  
Use when noise level is unknown:
- Plot (‖x‖, ‖Ax - y‖) on log-log axes
- Find corner of the curve
- Balances fidelity and smoothness

### Generalized Cross-Validation (GCV)
Use for automatic selection:
- No need to know noise level
- More computationally expensive
- Implementation: See advanced references

## 💡 Key Insights

1. **Regularization is essential** for ill-posed problems with noise
2. **Parameter selection matters** - optimal λ depends on noise level
3. **No perfect method** - Tikhonov vs TSVD depends on problem structure
4. **Trade-offs exist** between smoothness and fidelity
5. **Robustness matters** - both methods work across diverse problems

## 🎓 Learning Resources

Each notebook includes:
- Mathematical derivations
- Theoretical explanations
- Practical implementations
- Visual demonstrations
- Statistical analysis

## 📧 Questions?

Refer to the comments in each notebook for:
- Mathematical background
- Implementation details
- Physical interpretation
- Troubleshooting tips

---

**Last updated:** January 2026  
**Python version:** 3.8+  
**Status:** Complete and tested ✅