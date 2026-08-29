"""Deterministic reference solvers for the revision (reviewer R2.17, R2.18).

Reviewer 2 objected to calling L-BFGS-B a "QP baseline" and asked for a dedicated
QP solver. This module makes the roles explicit and separate:

    OSQP reference      primary deterministic reference for bound-constrained
                        quadratic subproblems
    L-BFGS-B cross-check    independent multistart verification solver

Neither is called a "QP baseline" anywhere. Both are evaluated on the objective
AS IMPLEMENTED (sim_common.make_cost), so the reported objective values are
directly comparable with the metaheuristic values, and any modelling error in
H/f shows up as a disagreement rather than being hidden.

OSQP is only valid where the implemented objective really is the quadratic
0.5 U'HU + f'U on a box. For the pendulum-cart that holds only while the soft
cart penalty is inactive, which is certified separately and exactly over the
whole admissible input box by pendcart_penalty_certificate.py. Callers must pass
`allow_osqp=False` for any instance where that certificate does not hold.
"""
import numpy as np

try:
    import osqp
    from scipy import sparse
    HAVE_OSQP = True
except Exception:                                                     # pragma: no cover
    HAVE_OSQP = False

from sim_common import make_cost, reference_solve
from mpc_utils import compute_f


def problem_linear_term(problem, sysdef, x, k):
    x_c = sysdef.cost_state(x)
    ref = sysdef.reference_horizon(k, problem.P)
    return compute_f(problem.Phi, problem.Gamma, problem.Q_block, x_c, ref)


def osqp_reference(problem, f, eps=1e-12, max_iter=400000):
    """Primary deterministic reference: bound-constrained QP via OSQP.

    Returns (solution, status, info dict). The objective value is deliberately
    NOT computed here; the caller evaluates it with the implemented cost so that
    both solvers are scored on exactly the same function.
    """
    if not HAVE_OSQP:
        return None, "unavailable", {}
    d = len(problem.lb_seq)
    P = sparse.csc_matrix(0.5 * (problem.H + problem.H.T))
    A = sparse.identity(d, format="csc")
    m = osqp.OSQP()
    m.setup(P=P, q=np.asarray(f, dtype=float), A=A,
            l=np.asarray(problem.lb_seq, dtype=float),
            u=np.asarray(problem.ub_seq, dtype=float),
            eps_abs=eps, eps_rel=eps, max_iter=max_iter,
            polish=True, verbose=False)
    r = m.solve()
    status = str(r.info.status)
    if r.x is None or not np.all(np.isfinite(r.x)):
        return None, status, {"iterations": int(getattr(r.info, "iter", -1))}
    info = {
        "iterations": int(getattr(r.info, "iter", -1)),
        "primal_residual": float(getattr(r.info, "pri_res", np.nan)),
        "dual_residual": float(getattr(r.info, "dua_res", np.nan)),
        "status_polish": int(getattr(r.info, "status_polish", 0)),
        "clip_needed": float(np.max(np.abs(np.clip(r.x, problem.lb_seq, problem.ub_seq) - r.x))),
    }
    return np.clip(r.x, problem.lb_seq, problem.ub_seq), status, info


def lbfgsb_crosscheck(problem, x, k, n_restarts=20, seed=0):
    """Independent verification solver: multistart L-BFGS-B."""
    sol, val, n_starts, spread = reference_solve(problem, x, k, last_seq=None,
                                                 n_restarts=n_restarts, seed=seed)
    return sol, float(val), {"n_starts": int(n_starts), "multistart_spread": float(spread)}


def kkt_certificate(problem, g, u, mu, bound_tol=1e-9):
    """Strong-convexity bound on the reference's own suboptimality.

    With mu = lambda_min(H) > 0 on a box, J(u) - J* <= ||proj grad||^2 / (2 mu).
    Same certificate used in Phase 1, kept here so both solvers are certified by
    an identical, solver-independent criterion.
    """
    lb, ub = problem.lb_seq, problem.ub_seq
    at_lo = u <= lb + bound_tol
    at_hi = u >= ub - bound_tol
    gp = np.asarray(g, dtype=float).copy()
    gp[at_lo] = np.minimum(gp[at_lo], 0.0)
    gp[at_hi] = np.maximum(gp[at_hi], 0.0)
    gpn = float(np.linalg.norm(gp))
    return {"proj_grad_norm": gpn,
            "subopt_bound_abs": (gpn * gpn / (2.0 * mu)) if mu > 0 else np.inf,
            "n_free": int((~(at_lo | at_hi)).sum()),
            "n_active": int((at_lo | at_hi).sum())}


def deterministic_reference(problem, sysdef, x, k, allow_osqp=True,
                            n_restarts=20, seed=0):
    """Run both solvers on the same implemented objective and compare.

    Returns a record with both objective values, their discrepancy, the
    solution-vector discrepancy, solver diagnostics, the strong-convexity
    suboptimality certificate for the accepted solution, and which solver was
    accepted as the reference.
    """
    cost = make_cost(problem, x, k)
    f = problem_linear_term(problem, sysdef, x, k)
    eig = np.linalg.eigvalsh(0.5 * (problem.H + problem.H.T))
    mu, lmax = float(eig.min()), float(eig.max())

    u_l, J_l, info_l = lbfgsb_crosscheck(problem, x, k, n_restarts=n_restarts, seed=seed)

    rec = {"lambda_min": mu, "lambda_max": lmax,
           "cond_H": (lmax / mu) if mu > 0 else np.nan,
           "J_lbfgsb": J_l, "lbfgsb_multistart_spread": info_l["multistart_spread"],
           "lbfgsb_success": bool(np.isfinite(J_l))}

    if allow_osqp and HAVE_OSQP:
        u_o, status, info_o = osqp_reference(problem, f)
        rec["osqp_status"] = status
        if u_o is not None:
            J_o = float(cost(u_o))
            rec.update({"J_osqp": J_o,
                        "osqp_iterations": info_o.get("iterations", -1),
                        "osqp_primal_residual": info_o.get("primal_residual", np.nan),
                        "osqp_dual_residual": info_o.get("dual_residual", np.nan),
                        "obj_abs_diff": abs(J_o - J_l),
                        "obj_rel_diff": abs(J_o - J_l) / max(1.0, abs(J_l)),
                        "sol_inf_diff": float(np.max(np.abs(u_o - u_l))),
                        "osqp_minus_lbfgsb": J_o - J_l})
            # Accept whichever solver attains the lower value on the implemented
            # objective; both are then certified by the same KKT bound.
            if J_o <= J_l:
                u_ref, J_ref, which = u_o, J_o, "OSQP"
            else:
                u_ref, J_ref, which = u_l, J_l, "L-BFGS-B"
        else:
            u_ref, J_ref, which = u_l, J_l, "L-BFGS-B"
            rec.update({"J_osqp": np.nan, "obj_abs_diff": np.nan, "obj_rel_diff": np.nan,
                        "sol_inf_diff": np.nan, "osqp_minus_lbfgsb": np.nan})
    else:
        u_ref, J_ref, which = u_l, J_l, "L-BFGS-B"
        rec.update({"osqp_status": "not attempted (non-quadratic instance)"
                    if not allow_osqp else "unavailable",
                    "J_osqp": np.nan, "obj_abs_diff": np.nan, "obj_rel_diff": np.nan,
                    "sol_inf_diff": np.nan, "osqp_minus_lbfgsb": np.nan})

    g = problem.H @ u_ref + f
    if sysdef.cart_constraint is not None:
        S = np.kron(np.eye(problem.P), np.array([[1, 0, 0, 0]], dtype=float))
        cart = (S @ (problem.Phi @ sysdef.cost_state(x) + problem.Gamma @ u_ref)).ravel()
        viol = np.maximum(np.abs(cart) - sysdef.cart_constraint, 0.0)
        g = g + sysdef.constraint_penalty * (problem.Gamma.T @ (S.T @ (2.0 * viol * np.sign(cart))))
    cert = kkt_certificate(problem, g, u_ref, mu)
    rec.update(cert)
    rec["subopt_bound_norm"] = cert["subopt_bound_abs"] / max(1.0, abs(J_ref))
    rec["reference_solver"] = which
    rec["J_reference"] = J_ref
    return u_ref, J_ref, rec
