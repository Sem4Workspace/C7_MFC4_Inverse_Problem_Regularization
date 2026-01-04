from tikhonov import tikhonov_reconstruction
from tsvd import tsvd_reconstruction

def apply_decision(decision, A, y, U=None, S=None, Vt=None):
    method = decision["method"]
    param = decision["parameter"]

    if method == "tikhonov":
        return tikhonov_reconstruction(A, y, param)

    elif method == "tsvd":
        if U is None or S is None or Vt is None:
            raise ValueError("SVD components required for TSVD")
        return tsvd_reconstruction(U, S, Vt, y, int(param))

    else:
        raise ValueError("Unknown method selected by LLM")
