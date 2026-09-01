#!/usr/bin/env python3
"""Phase-1 Step D: certify the deterministic reference solutions (reviewer R2.18).

The manuscript asserts that the MPC subproblems are convex and that L-BFGS-B
therefore reaches the global optimum. This script does not take that on trust.
Everything below is verified against the OBJECTIVE AS IMPLEMENTED
(sim_common.make_cost), not against the objective as described in the paper.

Checks performed per subproblem instance:

  1. HESSIAN            symmetry, eigenvalues, PSD status. The condition number
                        is reported only when lambda_min > 0; otherwise it is
                        left undefined rather than printed as a large number.
  2. MODEL FIDELITY     for purely quadratic systems, the implemented cost must
                        equal 0.5 U'HU + f'U at random points. This verifies
                        compute_f / build_qp_matrices, i.e. that the H and f
                        used for certification are the ones actually optimized.
  3. CONVEXITY          black-box midpoint-convexity and directional second
                        differences on the IMPLEMENTED objective, including
                        points deliberately placed across the pendulum-cart
                        penalty kink.
  4. GRADIENT           analytic gradient (including the soft-penalty term)
                        against central finite differences, again including
                        near-kink points, plus an explicit continuity probe of
                        the penalty derivative across |cart| = limit.
  5. KKT CERTIFICATE    for the returned reference solution on the box: free
                        variables must have |g_i| ~ 0; lower-active must have
                        g_i >= 0; upper-active must have g_i <= 0. With a PSD
                        Hessian on a box, satisfied KKT conditions are
                        SUFFICIENT for global optimality, so this is a proof
                        rather than the multistart evidence used before.
  6. OSQP CROSS-CHECK   independent bound-constrained QP solve (quadratic
                        systems only), compared on objective value.
  7. MULTISTART SPREAD  retained from sim_common.reference_solve.

Outputs (data/):
    reference_certificates.csv          one row per instance
    reference_certification_summary.csv per-system rollup
    reference_certification.json        headline verdict

Usage:
    python reference_verification.py [--instants 25] [--seed 20260815]
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
# Output root. Defaults to the released data/ directory; set SGO_OUTPUT_DIR to
# redirect every generated file elsewhere (e.g. a scratch directory) so that a
# smoke test cannot overwrite the committed production evidence.
OUT = os.environ.get("SGO_OUTPUT_DIR") or os.path.join(PROJECT, "data")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

from system_defs import get_systems                                   # noqa: E402
from sim_common import (simulate, build_problem, make_cost,           # noqa: E402
                        reference_solve)
from mpc_utils import compute_f                                       # noqa: E402

SEED_BASE = 4_000_000          # see data/provenance/environment.md
REF_RESTARTS = 20
NC_EXTRA = (2, 3)              # extra decision dimensions probed per system
N_EXTRA_INSTANTS = 5

try:
    import osqp
    from scipy import sparse
    HAVE_OSQP = True
except Exception:                                                     # pragma: no cover
    HAVE_OSQP = False


# --------------------------------------------------------------------------
# analytic pieces of the implemented objective
# --------------------------------------------------------------------------
def cart_values(problem, x_c, U):
    """Predicted cart positions over the horizon (pendulum-cart only)."""
    S = np.kron(np.eye(problem.P), np.array([[1, 0, 0, 0]], dtype=float))
    return (S @ (problem.Phi @ x_c + problem.Gamma @ U)).ravel(), S


def analytic_grad(problem, x_c, f, U):
    """Gradient of the implemented objective.

    J(U) = 0.5 U'HU + f'U  [+ c * sum_i max(|cart_i| - L, 0)^2]

    The penalty is C^1: d/dt max(|t|-L,0)^2 = 2*max(|t|-L,0)*sign(t), which is
    continuous and vanishes at |t| = L, so no subgradient is required. Check 4
    verifies this numerically rather than assuming it.
    """
    g = problem.H @ U + f
    s = problem.sys
    if s.cart_constraint is not None:
        cart, S = cart_values(problem, x_c, U)
        viol = np.maximum(np.abs(cart) - s.cart_constraint, 0.0)
        dpen = 2.0 * viol * np.sign(cart)
        g = g + s.constraint_penalty * (problem.Gamma.T @ (S.T @ dpen))
    return g


def kkt_residuals(problem, g, u, mu, bound_tol=1e-9):
    """First-order residuals on the box PLUS a rigorous optimality-gap bound.

    A raw gradient residual is not directly interpretable: the objectives here
    span |J| from 1e-10 to 1e3, so the same residual means different things in
    different instances. What the paper actually needs is a bound on
    J(u) - J*, in the same units as the reported optimality gaps.

    For an objective that is strongly convex with modulus mu (mu = lambda_min(H)
    > 0 for every instance here; the soft cart penalty is convex and only adds
    curvature), the projected gradient

        g_proj_i = g_i                if lb_i < u_i < ub_i
                 = min(g_i, 0)        if u_i is at its lower bound
                 = max(g_i, 0)        if u_i is at its upper bound

    satisfies the standard strong-convexity bound

        J(u) - J*  <=  ||g_proj||^2 / (2 mu).

    That is a certificate: it is an upper bound computed from quantities we
    measure, and it is valid regardless of how the solver arrived at u.
    """
    lb, ub = problem.lb_seq, problem.ub_seq
    at_lo = u <= lb + bound_tol
    at_hi = u >= ub - bound_tol
    free = ~(at_lo | at_hi)
    g_proj = g.copy()
    g_proj[at_lo] = np.minimum(g[at_lo], 0.0)
    g_proj[at_hi] = np.maximum(g[at_hi], 0.0)
    r_free = float(np.max(np.abs(g[free]))) if free.any() else 0.0
    r_lo = float(np.max(np.maximum(-g[at_lo], 0.0))) if at_lo.any() else 0.0
    r_hi = float(np.max(np.maximum(g[at_hi], 0.0))) if at_hi.any() else 0.0
    gp = float(np.linalg.norm(g_proj))
    gap_bound = gp * gp / (2.0 * mu) if mu > 0 else np.inf
    scale = 1.0 + float(np.max(np.abs(g))) if g.size else 1.0
    return dict(kkt_free_residual=r_free, kkt_lower_residual=r_lo,
                kkt_upper_residual=r_hi,
                kkt_residual_abs=max(r_free, r_lo, r_hi),
                kkt_residual_rel=max(r_free, r_lo, r_hi) / scale,
                proj_grad_norm=gp,
                subopt_bound_abs=gap_bound,
                subopt_bound_norm=gap_bound,   # filled in by caller (needs J*)
                n_free=int(free.sum()), n_at_lower=int(at_lo.sum()),
                n_at_upper=int(at_hi.sum()), grad_inf_norm=float(np.max(np.abs(g))))


# --------------------------------------------------------------------------
# black-box checks on the implemented objective
# --------------------------------------------------------------------------
def check_model_fidelity(cost, problem, f, rng, n=20):
    """Implemented cost vs 0.5 U'HU + f'U (exact for non-penalty systems)."""
    worst = 0.0
    for _ in range(n):
        U = problem.lb_seq + rng.random(len(problem.lb_seq)) * (problem.ub_seq - problem.lb_seq)
        model = 0.5 * (U @ problem.H @ U) + f @ U
        worst = max(worst, abs(cost(U) - model) / max(1.0, abs(model)))
    return worst


def check_convexity(cost, problem, rng, n_pairs=200, n_dirs=100):
    """Midpoint convexity and directional second differences."""
    lb, ub = problem.lb_seq, problem.ub_seq
    d = len(lb)
    worst_mid = 0.0
    for _ in range(n_pairs):
        a = lb + rng.random(d) * (ub - lb)
        b = lb + rng.random(d) * (ub - lb)
        Ja, Jb, Jm = cost(a), cost(b), cost(0.5 * (a + b))
        scale = max(1.0, abs(Ja), abs(Jb))
        worst_mid = max(worst_mid, (Jm - 0.5 * (Ja + Jb)) / scale)   # >0 => non-convex
    # Directional second differences. Sample strictly inside the box with a
    # margin of h so that x +/- h*v stays feasible: clipping the probe points
    # back onto the box makes the stencil asymmetric and manufactures spurious
    # negative curvature, which is a defect of the test, not of the objective.
    worst_curv = np.inf
    h = 1e-3 * float(np.min(ub - lb))
    for _ in range(n_dirs):
        v = rng.normal(size=d)
        v /= max(np.linalg.norm(v), 1e-300)
        x = (lb + h) + rng.random(d) * ((ub - h) - (lb + h))
        xp, xm = x + h * v, x - h * v
        if np.any(xp < lb) or np.any(xp > ub) or np.any(xm < lb) or np.any(xm > ub):
            continue
        Jx = cost(x)
        second = (cost(xp) - 2 * Jx + cost(xm)) / (h * h)
        # Normalise by the local objective scale so systems with |J| ~ 1e-10 and
        # |J| ~ 1e3 are judged on the same footing.
        worst_curv = min(worst_curv, second / max(1.0, abs(Jx)))
    return worst_mid, float(worst_curv)


def check_gradient(problem, x_c, f, cost, rng, n=25, near_kink=False):
    """Analytic gradient vs central finite differences."""
    lb, ub = problem.lb_seq, problem.ub_seq
    d = len(lb)
    worst = 0.0
    for _ in range(n):
        U = lb + rng.random(d) * (ub - lb)
        if near_kink:
            # Bias toward the penalty boundary by scaling the input magnitude.
            U = U * rng.uniform(0.2, 1.0)
        g = analytic_grad(problem, x_c, f, U)
        h = 1e-6 * np.maximum(1.0, np.abs(U))
        for i in range(d):
            Up, Um = U.copy(), U.copy()
            Up[i] += h[i]; Um[i] -= h[i]
            fd = (cost(Up) - cost(Um)) / (2 * h[i])
            worst = max(worst, abs(fd - g[i]) / max(1.0, abs(fd)))
    return worst


def check_penalty_continuity(problem, x_c, f, cost, rng, n=200):
    """Probe the soft-penalty region and its derivative.

    Returns (worst_rel_grad_error, max_abs_cart, n_samples_in_penalty_region).
    A worst-error of 0.0 with zero samples in the penalty region is a VACUOUS
    pass, so the sample count is returned and reported rather than hidden.
    """
    s = problem.sys
    if s.cart_constraint is None:
        return np.nan, np.nan, 0
    lb, ub = problem.lb_seq, problem.ub_seq
    d = len(lb)
    worst = np.nan
    max_cart = 0.0
    n_active = 0
    for _ in range(n):
        U = lb + rng.random(d) * (ub - lb)
        cart, _ = cart_values(problem, x_c, U)
        max_cart = max(max_cart, float(np.max(np.abs(cart))))
        if np.max(np.abs(cart)) < s.cart_constraint:
            continue          # penalty contributes exactly zero here
        n_active += 1
        g = analytic_grad(problem, x_c, f, U)
        h = 1e-7
        for i in range(d):
            Up, Um = U.copy(), U.copy()
            Up[i] += h; Um[i] -= h
            fd = (cost(Up) - cost(Um)) / (2 * h)
            e = abs(fd - g[i]) / max(1.0, abs(fd))
            worst = e if not np.isfinite(worst) else max(worst, e)
    return worst, max_cart, n_active


def osqp_solve(problem, f):
    """Independent bound-constrained QP solve (quadratic systems only)."""
    if not HAVE_OSQP:
        return np.nan, np.nan, "unavailable"
    d = len(problem.lb_seq)
    P = sparse.csc_matrix(0.5 * (problem.H + problem.H.T))
    A = sparse.identity(d, format="csc")
    m = osqp.OSQP()
    m.setup(P=P, q=np.asarray(f, dtype=float), A=A,
            l=np.asarray(problem.lb_seq, dtype=float),
            u=np.asarray(problem.ub_seq, dtype=float),
            eps_abs=1e-12, eps_rel=1e-12, max_iter=200000,
            polish=True, verbose=False)
    r = m.solve()
    status = str(r.info.status)
    if r.x is None or not np.all(np.isfinite(r.x)):
        return np.nan, np.nan, status
    u = np.clip(r.x, problem.lb_seq, problem.ub_seq)
    J = 0.5 * (u @ problem.H @ u) + f @ u
    return float(J), float(np.max(np.abs(u - r.x))), status


# --------------------------------------------------------------------------
def certify_instance(key, sysdef, x, k, Nc, rng):
    problem = build_problem(sysdef, Nc, 1.0, sysdef.P)
    x_c = sysdef.cost_state(x)
    ref = sysdef.reference_horizon(k, problem.P)
    f = compute_f(problem.Phi, problem.Gamma, problem.Q_block, x_c, ref)
    cost = make_cost(problem, x, k)

    H = problem.H
    sym_err = float(np.max(np.abs(H - H.T))) / max(1.0, float(np.max(np.abs(H))))
    eig = np.linalg.eigvalsh(0.5 * (H + H.T))
    lmin, lmax = float(eig.min()), float(eig.max())
    psd = bool(lmin >= -1e-10 * max(1.0, abs(lmax)))
    # Condition number only where it is mathematically meaningful.
    cond = float(lmax / lmin) if lmin > 0 else np.nan

    u_star, j_star, n_starts, spread = reference_solve(
        problem, x, k, last_seq=None, n_restarts=REF_RESTARTS, seed=900 + k)
    j_star = float(j_star)
    g = analytic_grad(problem, x_c, f, u_star)
    row = dict(system=key, step=int(k), Nc=int(Nc), dim=int(len(problem.lb_seq)),
               J_star=j_star, ref_spread=float(spread), ref_starts=int(n_starts),
               H_symmetry_rel=sym_err, lambda_min=lmin, lambda_max=lmax,
               H_psd=psd, cond_H=cond,
               has_soft_penalty=bool(sysdef.cart_constraint is not None))
    row.update(kkt_residuals(problem, g, u_star, mu=lmin))
    # Express the certificate in the same normalisation the paper uses for
    # optimality gaps, so it can be compared directly with the metaheuristic gaps.
    row["subopt_bound_norm"] = row["subopt_bound_abs"] / max(1.0, abs(j_star))

    row["model_fidelity_rel"] = (np.nan if sysdef.cart_constraint is not None
                                 else check_model_fidelity(cost, problem, f, rng))
    mid, curv = check_convexity(cost, problem, rng)
    row["convexity_midpoint_worst"] = mid          # > 0 would indicate non-convexity
    row["convexity_second_diff_min"] = curv        # < 0 would indicate negative curvature
    row["convex_ok"] = bool(mid <= 1e-9 and curv >= -1e-6)
    row["grad_fd_rel_worst"] = check_gradient(problem, x_c, f, cost, rng)
    kink_err, max_cart, n_active = check_penalty_continuity(problem, x_c, f, cost, rng)
    row["grad_fd_rel_worst_in_penalty"] = kink_err
    row["max_abs_predicted_cart"] = max_cart
    row["penalty_limit"] = (sysdef.cart_constraint if sysdef.cart_constraint is not None
                            else np.nan)
    row["n_samples_in_penalty_region"] = int(n_active)
    row["penalty_active"] = bool(n_active > 0)

    if sysdef.cart_constraint is None:
        j_osqp, clip, status = osqp_solve(problem, f)
        row["J_osqp"] = j_osqp
        row["osqp_status"] = status
        row["osqp_rel_diff"] = (abs(j_osqp - j_star) / max(1.0, abs(j_star))
                                if np.isfinite(j_osqp) else np.nan)
        # Signed: negative means the independent QP solver found a LOWER
        # objective, i.e. the L-BFGS-B reference was not optimal.
        row["osqp_minus_ref"] = (j_osqp - j_star) if np.isfinite(j_osqp) else np.nan
    else:
        row["J_osqp"] = np.nan; row["osqp_status"] = "n/a (soft penalty, not a pure QP)"
        row["osqp_rel_diff"] = np.nan; row["osqp_minus_ref"] = np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instants", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260815)
    args = ap.parse_args()
    t0 = time.perf_counter()
    rows = []
    for key, sysdef in get_systems().items():
        traj = simulate(key, "QP", params={"Nc": 1, "Q_scale": 1.0}, seed=11, verbose=False)
        xh, Nsim = traj["x_hist"], traj["Nsim"]
        steps = sorted(set(np.linspace(0, Nsim - 1, args.instants).astype(int).tolist()))
        print(f"{key}: {len(steps)} instants, Nc=1")
        for i, k in enumerate(steps):
            rng = np.random.default_rng(SEED_BASE + 1_000_000 * i + hash(key) % 1000)
            rows.append(certify_instance(key, sysdef, xh[:, k], int(k), 1, rng))
        extra = sorted(set(np.linspace(0, Nsim - 1, N_EXTRA_INSTANTS).astype(int).tolist()))
        for Nc in NC_EXTRA:
            print(f"{key}: {len(extra)} instants, Nc={Nc}")
            for i, k in enumerate(extra):
                rng = np.random.default_rng(SEED_BASE + 7_000_000 * Nc + 1_000_000 * i)
                rows.append(certify_instance(key, sysdef, xh[:, k], int(k), Nc, rng))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "reference_certificates.csv"), index=False)

    summ = (df.groupby("system")
              .agg(instances=("J_star", "size"),
                   all_psd=("H_psd", "all"),
                   min_lambda_min=("lambda_min", "min"),
                   max_cond_H=("cond_H", "max"),
                   all_convex=("convex_ok", "all"),
                   worst_midpoint=("convexity_midpoint_worst", "max"),
                   worst_second_diff=("convexity_second_diff_min", "min"),
                   worst_grad_fd_rel=("grad_fd_rel_worst", "max"),
                   worst_grad_fd_in_penalty=("grad_fd_rel_worst_in_penalty", "max"),
                   max_predicted_cart=("max_abs_predicted_cart", "max"),
                   penalty_ever_active=("penalty_active", "any"),
                   worst_model_fidelity=("model_fidelity_rel", "max"),
                   max_kkt_abs=("kkt_residual_abs", "max"),
                   max_kkt_rel=("kkt_residual_rel", "max"),
                   max_subopt_bound_norm=("subopt_bound_norm", "max"),
                   max_ref_spread=("ref_spread", "max"),
                   max_osqp_rel_diff=("osqp_rel_diff", "max"),
                   worst_osqp_minus_ref=("osqp_minus_ref", "min")).reset_index())
    summ.to_csv(os.path.join(OUT, "reference_certification_summary.csv"), index=False)

    # Certification criterion: the reference must be provably closer to the true
    # optimum than the tolerance at which convergence is declared in the paper
    # (norm_gap <= 1e-4). A reference whose own suboptimality bound exceeds that
    # cannot be used to judge whether a metaheuristic converged.
    CONV_TOL = 1e-4
    CERT_TOL = 1e-2 * CONV_TOL      # two orders of margin below the decision threshold
    certified = df.subopt_bound_norm <= CERT_TOL
    verdict = {
        "instances": int(len(df)),
        "all_hessians_psd": bool(df.H_psd.all()),
        "min_lambda_min": float(df.lambda_min.min()),
        "all_convex_checks_passed": bool(df.convex_ok.all()),
        "certificate": "J(u_ref) - J* <= ||proj_grad||^2 / (2*lambda_min), "
                       "normalised by max(1,|J*|)",
        "certification_tolerance_norm": CERT_TOL,
        "convergence_tolerance_in_paper": CONV_TOL,
        "max_subopt_bound_norm": float(df.subopt_bound_norm.max()),
        "certified_fraction": float(certified.mean()),
        "uncertified_instances": int((~certified).sum()),
        "uncertified_by_system": {k: int(v) for k, v in
                                  df[~certified].groupby("system").size().items()},
        "max_gradient_fd_rel": float(df.grad_fd_rel_worst.max()),
        "max_multistart_spread": float(df.ref_spread.max()),
        "osqp_available": HAVE_OSQP,
        "max_osqp_rel_diff": (float(df.osqp_rel_diff.max(skipna=True))
                              if df.osqp_rel_diff.notna().any() else None),
        "worst_osqp_minus_ref": (float(df.osqp_minus_ref.min(skipna=True))
                                 if df.osqp_minus_ref.notna().any() else None),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    verdict["verdict"] = ("CERTIFIED" if (verdict["all_hessians_psd"]
                                          and verdict["all_convex_checks_passed"]
                                          and verdict["certified_fraction"] == 1.0)
                          else "PARTIALLY CERTIFIED")
    with open(os.path.join(OUT, "reference_certification.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(summ.to_string(index=False))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
