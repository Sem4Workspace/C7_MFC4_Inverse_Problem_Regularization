# Inverse Problem Regularization for Signal Reconstruction With LLM assistance in Paramater Selection

## Course
**22MAT230 – Mathematics for Computing IV**  
Amrita Vishwa Vidyapeetham, Coimbatore

---

#  Team 7

| Name | Roll Number |
|-----|-------------|
| G Prajwal Priyadarshan | CB.SC.U4AIE24214 |
| Kabilan K | CB.SC.U4AIE24224 |
| Kishore B | CB.SC.U4AIE24227 |
| Rahul L S | CB.SC.U4AIE24248 |

---

# Overview

Many real-world systems measure **indirect and noisy observations** of signals. Recovering the original signal from these measurements leads to an **inverse problem**.

Inverse problems are often **ill-posed**, meaning:

- Small noise in measurements causes **large reconstruction errors**
- The system matrix is **ill-conditioned**
- Direct inversion methods **amplify noise**

This project studies how **regularization techniques** stabilize these problems and produce meaningful signal reconstructions.

The project implements and compares several reconstruction methods:

- Pseudoinverse
- Tikhonov Regularization
- Truncated SVD (TSVD)
- Non-Stationary Iterated Tikhonov (NSIT)
- Fast Non-Stationary Iterated Tikhonov (FNSIT)
- LLM-guided parameter selection for TSVD,NSIT,FNSIT

---

# Objectives

The objectives of this project are:

- Understand why **pseudoinverse fails for ill-posed systems**
- Study the role of **Singular Value Decomposition (SVD)**
- Implement **regularization techniques**
- Compare different reconstruction algorithms
- Explore **iterative regularization**
- Investigate **LLM-assisted parameter tuning**

---

#  Mathematical Background

The forward model of an inverse problem is

yδ = Ax + ε


Where

- **A** → forward operator  
- **x** → true signal  
- **yδ** → noisy measurement  
- **ε** → noise

A naive reconstruction is

x = A†y


However, if **A is ill-conditioned**, small singular values cause **severe noise amplification**.

Regularization stabilizes this inversion.

---

# Methods Implemented

## 1️⃣ Pseudoinverse Reconstruction

The simplest way to recover the unknown signal is by directly computing the **Moore–Penrose pseudoinverse** of the forward operator.

The reconstruction is given by

x = A†y

where

- **A†** is the Moore–Penrose pseudoinverse of the forward matrix **A**
- **y** is the observed noisy measurement

Using Singular Value Decomposition (SVD):

A = UΣVᵀ

the pseudoinverse becomes

A† = VΣ⁻¹Uᵀ

Thus the reconstruction can be written as

x = Σ ( (uᵢᵀ y) / σᵢ ) vᵢ

where

- σᵢ are the singular values of **A**
- uᵢ and vᵢ are the singular vectors

### Problem

If **σᵢ are very small**, the term  

(uᵢᵀ y) / σᵢ  

becomes very large, causing **severe noise amplification**.

Therefore:

- small noise in **y** leads to **huge errors in x**
- the solution becomes **unstable**

This is why pseudoinverse **fails for ill-posed inverse problems**.

---

## 2️⃣ Tikhonov Regularization

To stabilize the inversion, **Tikhonov regularization** modifies the problem by adding a penalty term.

Instead of solving

Ax = y

we solve

min ||Ax − y||² + λ||x||²

The closed-form solution is

xλ = (AᵀA + λI)⁻¹ Aᵀy

where

- **λ** is the regularization parameter
- **I** is the identity matrix

### Interpretation

The parameter **λ** controls the trade-off between:

- **data fidelity** → fitting the measurements
- **solution stability** → preventing large oscillations

If

λ → 0 → solution approaches pseudoinverse  
λ → large → solution becomes overly smooth

Choosing a good **λ** is therefore a key challenge.

---

## 3️⃣ Truncated SVD (TSVD)

Another approach to stabilize the solution is **Truncated Singular Value Decomposition**.

Using SVD

A = UΣVᵀ

the pseudoinverse solution is

x = Σ ( (uᵢᵀ y) / σᵢ ) vᵢ

Instead of using all singular values, TSVD **truncates small singular values**:

x_k = Σ_{i=1}^k ( (uᵢᵀ y) / σᵢ ) vᵢ

where

- **k** is the truncation parameter
- only the **largest k singular values** are used

### Advantage

This prevents division by very small singular values, reducing noise amplification.

### Trade-off

- small **k** → stable but less accurate
- large **k** → more accurate but noisy

---

## 4️⃣ NSIT (Non-Stationary Iterated Tikhonov)

Instead of solving the problem once, **iterative regularization methods** update the solution gradually.

The **Non-Stationary Iterated Tikhonov (NSIT)** method updates the solution as

xₖ₊₁ = xₖ + (AᵀA + λₖI)⁻¹ Aᵀ (yδ − Axₖ)

where

- **xₖ** is the estimate at iteration k
- **λₖ** is the regularization parameter at iteration k
- **yδ** is the noisy observation
- **(yδ − Axₖ)** is the residual

### Key Idea

The regularization parameter **changes during iterations**.

This allows the algorithm to:

- start with **strong regularization**
- gradually **refine the solution**

The iteration typically stops using the **Morozov discrepancy principle**, which compares the residual norm with the noise level.

---

## 5️⃣ FNSIT (Fast Non-Stationary Iterated Tikhonov)

FNSIT is an accelerated version of the NSIT algorithm.

The goal is to:

- reduce computational cost
- improve convergence speed
- maintain reconstruction accuracy

FNSIT modifies the update rule to make iterations **more efficient**, especially for large-scale inverse problems.

This makes it suitable for problems involving:

- large matrices
- image reconstruction
- repeated inverse solves

---

## 6️⃣ LLM-Guided Parameter Selection

One of the biggest challenges in regularization methods is selecting **optimal parameters**, such as:

- λ in Tikhonov regularization
- truncation level **k** in TSVD
- iteration parameters in NSIT/FNSIT

Traditionally these are chosen using:

- manual tuning
- grid search
- discrepancy principles

In this project we experiment with **LLM-assisted parameter selection**.

The LLM analyzes:

- residual norms
- noise level
- reconstruction behaviour

and suggests improved parameter values.

This approach aims to:

- reduce manual tuning
- adapt parameters dynamically
- improve reconstruction quality

# System Architecture

The complete workflow of the project is illustrated below.
![alt text](diagram-export-3-11-2026-9_39_15-AM.png)


---

#  Reconstruction Pipeline


          True Signal
               │
               ▼
      Forward Operator (A)
               │
               ▼
        Noisy Observation (yδ)
               │
               ▼
     Reconstruction Algorithms
       ├── Pseudoinverse
       ├── Tikhonov Regularization
       ├── Truncated SVD (TSVD)
       ├── NSIT
       ├── FNSIT
       └── LLM Parameter Selection
               │
               ▼
        Reconstructed Signal
               │
               ▼
         Evaluation Metrics
       ├── Relative Error
       ├── Mean Squared Error (MSE)
       └── PSNR
---

---

#  Forward Operators Used

To simulate real inverse problems, multiple forward operators are used:

- Gaussian blur operator
- Downsampling operator
- Rank-deficient matrix

These operators create **ill-posed reconstruction scenarios**.

---

#  LLM Parameter Selection (Gemini API)

This project integrates **LLM-assisted parameter selection** using the **Google Gemini API**.

The LLM analyzes reconstruction behaviour and suggests improved regularization parameters based on:

- residual errors
- noise levels
- reconstruction performance

This helps **automate the parameter tuning process** and improves the stability of inverse problem reconstruction.

---

#  How to Get a Gemini API Key

To use the LLM integration in this project, you must generate a **Gemini API key**.

---

## Step 1 - Open Google AI Studio

Go to the following website:

https://aistudio.google.com

Sign in using your **Google account**.

---

## Step 2 - Generate an API Key

1. Navigate to **Get API Key**
2. Click **Create API Key**
3. Copy the generated API key

Example format:
AIzaSyXXXXXXXXXXXXXXX


---

## Step 3 - Add the API Key to the Project

Open the MATLAB script responsible for **LLM integration**.

Replace the placeholder API key with your own key.

Example:

```matlab
API_KEY = "YOUR_GEMINI_API_KEY";
```

Example with a real key format:
```matlab

API_KEY = "AIzaSyXXXXXXXXXXXXXXXXXXXX";
```



## Step 4 - Run the Experiment

After inserting the API key into the MATLAB script, run the main experiment file.

report.mlx


# Reconstruction Results

## Signal Reconstruction Example - Sinusodial Wave
![alt text](image.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-1.png)

## Image Reconstruction Example

![alt text](image-2.png)

These results show how regularization significantly improves reconstruction quality compared to naive pseudoinverse solutions.


# 📁 Repository Structure
```
C7_MFC4_Inverse_Problem_Regularization/
├── .gitignore
├── Base_Paper.pdf
├── README.md
├── Report.mlx
├── Report.pdf
├── analyze_notebook.py
├── architecture_diagram.png
├── final/
│   ├── create_gaussian_kernel.mlx
│   ├── image_reconstruction.asv
│   └── image_reconstruction.mlx
├── final_results/
│   ├── image-1.png
│   ├── image-2.png
│   └── image.png
├── finalreview_ppt.pdf
├── image-1.png
├── image-2.png
├── image-3.png
├── image.png
├── images/
│   ├── 291000.jpg
│   ├── 296007.jpg
│   ├── 296059.jpg
│   ├── README.md
│   ├── fnsit.ipynb
│   ├── forward_operator.ipynb
│   ├── llm_image_reconstruction.ipynb
│   ├── nsit_morozov.ipynb
│   ├── tank.ipynb
│   ├── tikhonov.ipynb
│   ├── toy.ipynb
│   ├── toy100x100.png
│   ├── toy16x16.png
│   └── tsvd.ipynb
├── requirements.txt
├── review_1_Matlab/
│   ├── Matlab_Codes_Images/
│   │   ├── 296059.jpg
│   │   ├── llm_image_reg_methods.mlx
│   │   ├── sample_forward_operator.mlx
│   │   ├── sample_reg_methods_all.mlx
│   │   ├── sample_tank.mlx
│   │   ├── sample_toy.mlx
│   │   ├── toy.png
│   │   └── toy16x16.png
│   ├── Matlab_Codes_Signals/
│   │   ├── Condition_number.mlx
│   │   ├── Evaluation_Metrics.mlx
│   │   ├── Forward_Models.mlx
│   │   ├── Noise_Models.mlx
│   │   ├── Reconstruction_Methods.mlx
│   │   └── Signal_Generation.mlx
│   └── Signals/
│       ├── Papers/
│       │   ├── An_Adaptive_Regularized_Solution_to_Inverse_Ill-Posed_Models.pdf
│       │   ├── Base_Paper.pdf
│       │   ├── NeurIPS-2021-learning-the-optimal-tikhonov-regularizer-for-inverse-problems-Paper (1).pdf
│       │   └── s00366-019-00920-z (2).pdf
│       ├── diagnostics/
│       │   ├── condition_number.py
│       │   ├── l_curve.py
│       │   ├── picard_plot.py
│       │   └── svd_analysis.py
│       ├── notebooks/
│       │   ├── 1_pseudoinverse_baseline.ipynb
│       │   ├── 2_regularization_comparison.ipynb
│       │   ├── 3_multimethod_evaluation.ipynb
│       │   ├── 4_noise_sensitivity.ipynb
│       │   ├── 5_nsit_advanced_comparison.ipynb
│       │   ├── 6_fnsit_toy_example.ipynb
│       │   ├── 9_simplified_llm_comparison.ipynb
│       │   └── README.md
│       └── src/
│           ├── evaluation/
│           │   ├── comparison.py
│           │   └── error_metrics.py
│           ├── forward_models/
│           │   ├── blur_operator.py
│           │   ├── downsample_operator.py
│           │   └── rank_deficient_operator.py
│           ├── noise_models/
│           │   └── noise.py
│           ├── reconstruction/
│           │   ├── fnsit.py
│           │   ├── nsit.py
│           │   ├── pseudoinverse.py
│           │   ├── spectral_filters.py
│           │   ├── tikhonov.py
│           │   └── tsvd.py
│           └── signal_generation/
│               └── generate_signals.py
└── review_1_ppt.pdf
```

# Reference Papers

Huang et al.
A Novel Iterative Integration Regularization Method for Ill-Posed Inverse Problems

#  Future Work

Possible future extensions of this project include:

- Extending experiments to **higher-dimensional inverse problems**.
- Testing the reconstruction methods on **real-world datasets**.
- Exploring **deep learning based regularization techniques**.
- Investigating **diffusion models for solving inverse problems**.
- Improving **automatic parameter selection and tuning methods**.

These directions could further enhance the robustness and applicability of regularization techniques in practical inverse problem settings.

---

#  Conclusion

This project demonstrates that **direct inversion methods fail for ill-posed inverse problems** due to the amplification of noise caused by small singular values.

Regularization techniques such as **Tikhonov regularization**, **Truncated SVD (TSVD)**, and **iterative algorithms like NSIT and FNSIT** provide stable and accurate reconstructions even in the presence of measurement noise.

Through experiments and analysis, we observe that **Singular Value Decomposition (SVD)** plays a central role in understanding the instability of inverse problems and guiding the development of effective regularization strategies.