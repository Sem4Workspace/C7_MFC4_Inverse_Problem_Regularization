import numpy as np
import matplotlib.pyplot as plt
from bsds300_loader import load_random_patch
from forward_model import forward_blur
from baseline_pseudoinverse import build_convolution_matrix

BSD_PATH = r"C:\Desktop\Sem 4\inverse-problems-regularization\data\BSDS300\images\test"
NUM_PATCHES = 30
PATCH_SIZE = 16

cond_numbers = []
spectra = []

for _ in range(NUM_PATCHES):
    x = load_random_patch(BSD_PATH, PATCH_SIZE)
    y, kernel = forward_blur(x)

    A = build_convolution_matrix(kernel, PATCH_SIZE)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    cond_numbers.append(S.max() / S.min())
    spectra.append(S)

print("Mean condition number:", np.mean(cond_numbers))
print("Median condition number:", np.median(cond_numbers))

plt.figure()
for S in spectra:
    plt.semilogy(S, alpha=0.4)
plt.title("Singular value spectra (BSDS300 test patches)")
plt.xlabel("Index")
plt.ylabel("Singular value (log scale)")
plt.grid(True)
plt.show()

plt.figure()
plt.hist(np.log10(cond_numbers), bins=15)
plt.xlabel("log10(condition number)")
plt.ylabel("Frequency")
plt.title("Condition number distribution (BSDS300)")
plt.grid(True)
plt.show()

x = load_random_patch(BSD_PATH, PATCH_SIZE)
y, kernel = forward_blur(x)

A = build_convolution_matrix(kernel, PATCH_SIZE)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

x_pinv = Vt.T @ np.diag(1/S) @ U.T @ y.flatten()

plt.figure(figsize=(9,3))
plt.subplot(1,3,1); plt.imshow(x, cmap='gray'); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(y, cmap='gray'); plt.title("Blurred+Noise")
plt.subplot(1,3,3); plt.imshow(x_pinv.reshape(PATCH_SIZE,PATCH_SIZE), cmap='gray')
plt.title("Pseudoinverse")
plt.show()

x = load_random_patch(BSD_PATH, PATCH_SIZE)
y, kernel = forward_blur(x)

A = build_convolution_matrix(kernel, PATCH_SIZE)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

x_pinv = Vt.T @ np.diag(1/S) @ U.T @ y.flatten()

plt.figure(figsize=(9,3))
plt.subplot(1,3,1); plt.imshow(x, cmap='gray'); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(y, cmap='gray'); plt.title("Blurred+Noise")
plt.subplot(1,3,3); plt.imshow(x_pinv.reshape(PATCH_SIZE,PATCH_SIZE), cmap='gray')
plt.title("Pseudoinverse")
plt.show()

from tikhonov import tikhonov_reconstruction

x_tikh = tikhonov_reconstruction(A, y.flatten(), lam=1e-2)


import matplotlib.pyplot as plt
import numpy as np
from tikhonov_sweep import sweep_lambda

lambdas = np.logspace(-6, 1, 30)
res, sol, err = sweep_lambda(A, y.flatten(), x.flatten(), lambdas)

plt.figure()
plt.loglog(res, sol, '-o')
plt.xlabel("||Ax - y||")
plt.ylabel("||x||")
plt.title("L-curve (Tikhonov)")
plt.grid(True)
plt.show()

plt.figure()
plt.semilogx(lambdas, err, '-o')
plt.xlabel("λ")
plt.ylabel("Reconstruction error")
plt.title("Error vs λ (Tikhonov)")
plt.grid(True)
plt.show()

from tsvd import tsvd_reconstruction

ks = range(1, len(S))
errors_tsvd = []

for k in ks:
    x_k = tsvd_reconstruction(U, S, Vt, y.flatten(), k)
    errors_tsvd.append(np.linalg.norm(x_k - x.flatten()))

plt.figure()
plt.plot(ks, errors_tsvd)
plt.xlabel("Truncation rank k")
plt.ylabel("Reconstruction error")
plt.title("Error vs k (TSVD)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,3))

plt.subplot(1,4,1)
plt.imshow(x.reshape(PATCH_SIZE, PATCH_SIZE), cmap='gray')
plt.title("Original")

plt.subplot(1,4,2)
plt.imshow(x_pinv.reshape(PATCH_SIZE, PATCH_SIZE), cmap='gray')
plt.title("Pseudoinverse")

plt.subplot(1,4,3)
plt.imshow(x_tikh.reshape(PATCH_SIZE, PATCH_SIZE), cmap='gray')
plt.title("Tikhonov")

plt.subplot(1,4,4)
plt.imshow(x_k.reshape(PATCH_SIZE, PATCH_SIZE), cmap='gray')
plt.title("TSVD")

plt.show()


#PHASE 3
from diagnostic_packager import package_diagnostics

diagnostics = package_diagnostics(
    singular_values=S,
    condition_number=S.max() / S.min(),
    tikh_errors=err,        # from lambda sweep
    lambdas=lambdas,
    tsvd_errors=errors_tsvd,
    ks=list(ks)
)

print(diagnostics)


#PHASE 3B
from llm_prompt import build_llm_prompt

prompt = build_llm_prompt(diagnostics)
print(prompt)


#PHASE 3C
from llm_decision import query_llm

decision = query_llm(prompt)
print("LLM decision:", decision)

#PHASE 3D
from apply_llm_decision import apply_decision

x_llm = apply_decision(
    decision,
    A,
    y.flatten(),
    U=U,
    S=S,
    Vt=Vt
)
plt.figure()
plt.imshow(x_llm.reshape(PATCH_SIZE, PATCH_SIZE), cmap='gray')
plt.title("LLM-guided reconstruction")
plt.show()
