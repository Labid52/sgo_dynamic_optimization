"""Standalone prediction-tail conventions for the Phase-2C addendum (reviewer R1.2).

Production code is NOT modified. This module rebuilds the MPC prediction
matrices under two conventions and is imported only by
`tail_convention_check.py`.

Decision-vector semantics, read from the implemented code rather than assumed:

    U = [u_0; u_1; ...; u_{Nc-1}]   ABSOLUTE physical inputs (not increments),
                                    bounded elementwise by [u_min, u_max]
    X_pred = Phi x + Gamma U        Phi is (P n_x, n_x), Gamma is (P n_x, Nc n_u)

`mpc_utils.build_prediction_matrices` fills, for prediction stage i
(which predicts x_{k+i+1}) and control index j:

    Gamma[i, j] = A^{i-j} B   for j < min(i+1, Nc), zero otherwise

i.e. the sum truncates at j = Nc-1. That is exactly the statement

    ZERO-TAIL:  u_j = 0 for j >= Nc

which is what the submitted and revised implementations use.

The alternative reviewer 1 asked about is

    HOLD-LAST:  u_j = u_{Nc-1} for j >= Nc

Substituting into x_{k+i+1} = A^{i+1} x + sum_{j=0}^{i} A^{i-j} B u_j gives, for
i >= Nc-1, a last column block that accumulates the whole tail:

    Gamma[i, Nc-1] = sum_{p=0}^{i-(Nc-1)} A^p B

while all earlier column blocks are unchanged. For i < Nc-1 the two conventions
coincide.

NOTE ON N_c = 1 (the Phase-2A primary configuration): the last control index is
also the first, so hold-last means u_0 is applied over the ENTIRE prediction
horizon, and Gamma[i,0] becomes sum_{p=0}^{i} A^p B instead of A^i B. This is a
substantially different prediction model, not a marginal tail correction.

DO NOT CONFUSE this with the warm-start shift-and-hold rule used between MPC
steps. That rule constructs an initial guess for the next optimisation from the
previous solution. This module changes the PREDICTION MODEL inside a single
optimisation. They are independent choices.
"""
import numpy as np


def build_prediction_matrices_tail(Ad, Bd, P, Nc, tail="zero"):
    """Phi, Gamma under the requested tail convention.

    tail="zero"      reproduces mpc_utils.build_prediction_matrices exactly
    tail="hold_last" holds u_{Nc-1} over the remainder of the horizon
    """
    if tail not in ("zero", "hold_last"):
        raise ValueError(f"unknown tail convention {tail!r}")
    Ad = np.asarray(Ad, dtype=float); Bd = np.asarray(Bd, dtype=float)
    nx, nu = Ad.shape[0], Bd.shape[1]
    Phi = np.zeros((P * nx, nx))
    Gamma = np.zeros((P * nx, Nc * nu))

    Ad_pow = np.eye(nx)
    for i in range(P):
        Ad_pow = Ad_pow @ Ad
        Phi[i * nx:(i + 1) * nx, :] = Ad_pow
        for j in range(min(i + 1, Nc)):
            r0, r1 = i * nx, (i + 1) * nx
            c0, c1 = j * nu, (j + 1) * nu
            if tail == "hold_last" and j == Nc - 1:
                # u_{Nc-1} also supplies every stage from Nc-1 up to i.
                blk = np.zeros((nx, nu))
                for p in range(0, i - (Nc - 1) + 1):
                    blk += np.linalg.matrix_power(Ad, p) @ Bd
                Gamma[r0:r1, c0:c1] = blk
            else:
                Gamma[r0:r1, c0:c1] = np.linalg.matrix_power(Ad, i - j) @ Bd
    return Phi, Gamma


def cart_box_extrema(Phi, Gamma, P, x_c, lb, ub):
    """Exact extrema of the predicted cart position over the input box.

    Same vertex argument as the Phase-2A certificate: the prediction is affine
    in U, so the extrema over a box are attained at a vertex and are available
    in closed form. Used to check whether the pendulum-cart soft penalty can
    activate under a given tail convention.
    """
    S = np.kron(np.eye(P), np.array([[1, 0, 0, 0]], dtype=float))
    a = (S @ (Phi @ x_c)).ravel()
    C = S @ Gamma
    hi = a + np.maximum(C * lb, C * ub).sum(axis=1)
    lo = a + np.minimum(C * lb, C * ub).sum(axis=1)
    return hi, lo
