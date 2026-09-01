#!/usr/bin/env python3
"""Phase-1 Step E: quantify how dynamic the MPC objective sequence actually is
(reviewer R2.2, R1.1, R1.5, and the mechanism half of R2.16).

The submitted manuscript asserts that the MPC objective sequence is "dynamic"
without measuring it. Reviewer 2's central objection is that consecutive MPC
problems may be nearly identical, in which case this is repeated convex
optimization rather than a dynamic-optimization benchmark. This script measures
the drift instead of asserting it.

Metrics, per consecutive pair (k, k+1) along the deterministic trajectory:

  LINEAR-TERM DRIFT
    df_abs          ||f_{k+1} - f_k||
    df_scaled       ||f_{k+1} - f_k|| / f_scale, where f_scale is the MEDIAN of
                    ||f_k|| over the trajectory. This is the primary measure:
                    it is well defined even where f_k ~ 0.
    df_relative     ||f_{k+1} - f_k|| / ||f_k||, reported ONLY as a secondary
                    quantity and flagged when ||f_k|| is negligible
                    (f_near_zero), because the ratio is meaningless there.
    df_angle_deg    angle between f_k and f_{k+1}; NaN when either is negligible.
    df_from_state / df_from_reference
                    decomposition of f_{k+1}-f_k = 2 Gamma' Qbar (Phi dx - dr)
                    into its state-driven and reference-driven parts (R1.1).

  SOLUTION DRIFT
    dU_shift        ||U*_{k+1} - shift(U*_k)||   <-- PRIMARY, receding-horizon
                    continuity: how far the optimum moves relative to what the
                    warm start actually supplies. At Nc=1 the shift is the
                    identity and this coincides with dU_raw (reported as such).
    dU_raw          ||U*_{k+1} - U*_k||          <-- secondary
    both also normalised by the input range.

  VALUE DRIFT AND WARM-START QUALITY
    dJstar_norm     |J*_{k+1} - J*_k| / max(1,|J*_k|)
    warm_gap_norm   (J_{k+1}(shift(U*_k)) - J*_{k+1}) / max(1,|J*_{k+1}|)
                    the honest measure of dynamic difficulty: how suboptimal is
                    the previous solution once the horizon has receded.
    cold_gap_norm   (J_{k+1}(0) - J*_{k+1}) / max(1,|J*_{k+1}|)
    warm_over_cold  warm_gap_norm / cold_gap_norm: the fraction of the
                    cold-start difficulty that survives warm starting.

  GEOMETRY
    active_set_change  fraction of decision components changing bound-active
                       status between k and k+1
    lambda_min/lambda_max/cond_H  curvature (R1.5). cond_H is reported only when
                       lambda_min > 0; it is never assumed to be finite.

Each row also carries the reference-certification bound (subopt_bound_norm) so
that instances whose reference is not trustworthy can be excluded downstream.

Outputs (data/):
    objective_drift_raw.csv, objective_drift_summary.csv, objective_drift.json

Usage:
    python objective_drift.py [--pairs 25]
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

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, build_problem, make_cost, reference_solve  # noqa: E402
from mpc_utils import compute_f                                        # noqa: E402

REF_RESTARTS = 20
NC_LIST = (1, 3)
NEAR_ZERO_REL = 1e-6      # ||f_k|| below this fraction of f_scale is "negligible"


def shift_sequence(U, nu, Nc):
    """Receding-horizon warm start: shift left by one input block, hold the last.

    This is exactly the rule sim_common.simulate applies between MPC steps, so
    the drift measured here is the drift the optimizers actually experience.
    """
    seq = np.asarray(U, dtype=float).copy()
    if Nc > 1:
        seq[:-nu] = seq[nu:]
    return seq


def active_set(u, lb, ub, tol=1e-9):
    return (u <= lb + tol).astype(int) - (u >= ub - tol).astype(int)


def analyse(key, sysdef, Nc, n_pairs):
    # Trajectory generated with the SAME Nc as the analysis, so the drift is
    # self-consistent with the closed loop it describes.
    traj = simulate(key, "QP", params={"Nc": Nc, "Q_scale": 1.0}, seed=11, verbose=False)
    xh, Nsim = traj["x_hist"], traj["Nsim"]
    problem = build_problem(sysdef, Nc, 1.0, sysdef.P)
    nu, lb, ub = sysdef.nu, problem.lb_seq, problem.ub_seq
    urange = float(np.max(ub - lb))

    eig = np.linalg.eigvalsh(0.5 * (problem.H + problem.H.T))
    lmin, lmax = float(eig.min()), float(eig.max())
    cond = float(lmax / lmin) if lmin > 0 else np.nan

    ks = sorted(set(np.linspace(0, Nsim - 2, n_pairs).astype(int).tolist()))

    # Drift scale. The median of ||f_k|| is a bad choice here: for regulation
    # problems the trajectory converges, so most of the sequence has f_k ~ 0 and
    # the median collapses, inflating every ratio by many orders of magnitude.
    # The MAXIMUM of ||f_k|| over the trajectory is the stable, interpretable
    # reference ("drift relative to the largest linear term the controller ever
    # sees"), and it keeps the measure in [0, ~2].
    fnorms = []
    cache = {}
    for k in set(ks) | {k + 1 for k in ks}:
        x_c = sysdef.cost_state(xh[:, k])
        ref = sysdef.reference_horizon(k, problem.P)
        f = compute_f(problem.Phi, problem.Gamma, problem.Q_block, x_c, ref)
        cache[k] = (x_c, ref, f)
        fnorms.append(float(np.linalg.norm(f)))
    f_scale = float(np.max(fnorms)) if fnorms else 1.0
    f_median = float(np.median(fnorms)) if fnorms else 1.0
    if f_scale <= 0:
        f_scale = 1.0

    rows = []
    for k in ks:
        x_c0, ref0, f0 = cache[k]
        x_c1, ref1, f1 = cache[k + 1]
        df = f1 - f0
        n0, n1, ndf = (float(np.linalg.norm(f0)), float(np.linalg.norm(f1)),
                       float(np.linalg.norm(df)))
        near_zero = bool(n0 < NEAR_ZERO_REL * f_scale)

        # Decomposition of the linear-term change (reviewer R1.1).
        M = 2.0 * problem.Gamma.T @ problem.Q_block
        d_state = float(np.linalg.norm(M @ (problem.Phi @ (x_c1 - x_c0))))
        d_ref = float(np.linalg.norm(M @ (ref1 - ref0)))

        u0, J0, _, spread0 = reference_solve(problem, xh[:, k], k, last_seq=None,
                                             n_restarts=REF_RESTARTS, seed=900 + k)
        u1, J1, _, spread1 = reference_solve(problem, xh[:, k + 1], k + 1, last_seq=None,
                                             n_restarts=REF_RESTARTS, seed=900 + k + 1)
        J0, J1 = float(J0), float(J1)

        u0_shift = np.clip(shift_sequence(u0, nu, Nc), lb, ub)
        cost1 = make_cost(problem, xh[:, k + 1], k + 1)
        J_warm = float(cost1(u0_shift))
        J_cold = float(cost1(np.zeros_like(lb)))
        den1 = max(1.0, abs(J1))
        warm_gap = (J_warm - J1) / den1
        cold_gap = (J_cold - J1) / den1

        # Reference-quality bound at k+1 (same certificate as Step D).
        g1 = problem.H @ u1 + f1
        at_lo = u1 <= lb + 1e-9; at_hi = u1 >= ub - 1e-9
        gp = g1.copy(); gp[at_lo] = np.minimum(g1[at_lo], 0.0); gp[at_hi] = np.maximum(g1[at_hi], 0.0)
        subopt = float(np.dot(gp, gp) / (2.0 * lmin)) / den1 if lmin > 0 else np.inf

        a0, a1 = active_set(u0, lb, ub), active_set(u1, lb, ub)
        rows.append(dict(
            system=key, Nc=Nc, dim=int(len(lb)), k=int(k),
            f_norm_k=n0, f_norm_k1=n1, f_scale=f_scale, f_norm_median=f_median,
            df_abs=ndf, df_scaled=ndf / f_scale,
            df_relative=(ndf / n0 if not near_zero else np.nan),
            f_near_zero=near_zero,
            # The direction of f is only meaningful in more than one dimension:
            # at dim = 1 the "angle" can only be 0 or 180 degrees and carries no
            # information about landscape rotation.
            df_angle_deg=(float(np.degrees(np.arccos(np.clip(
                np.dot(f0, f1) / (n0 * n1), -1.0, 1.0))))
                if (len(lb) > 1 and n0 > NEAR_ZERO_REL * f_scale
                    and n1 > NEAR_ZERO_REL * f_scale) else np.nan),
            df_from_state=d_state, df_from_reference=d_ref,
            dU_shift=float(np.linalg.norm(u1 - u0_shift)),
            dU_raw=float(np.linalg.norm(u1 - u0)),
            dU_shift_norm=float(np.linalg.norm(u1 - u0_shift)) / urange,
            dU_raw_norm=float(np.linalg.norm(u1 - u0)) / urange,
            shift_is_identity=bool(Nc == 1),
            J_star_k=J0, J_star_k1=J1,
            dJstar_norm=abs(J1 - J0) / max(1.0, abs(J0)),
            warm_gap_norm=warm_gap, cold_gap_norm=cold_gap,
            warm_over_cold=(warm_gap / cold_gap if cold_gap > 0 else np.nan),
            active_set_change=float(np.mean(a0 != a1)),
            n_active_k=int(np.sum(a0 != 0)), n_active_k1=int(np.sum(a1 != 0)),
            lambda_min=lmin, lambda_max=lmax, cond_H=cond,
            ref_spread_k1=float(spread1), subopt_bound_norm=subopt))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=25)
    args = ap.parse_args()
    t0 = time.perf_counter()
    rows = []
    for key, sysdef in get_systems().items():
        for Nc in NC_LIST:
            print(f"drift: {key} Nc={Nc}")
            rows += analyse(key, sysdef, Nc, args.pairs)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "objective_drift_raw.csv"), index=False)

    summ = (df.groupby(["system", "Nc"])
              .agg(dim=("dim", "first"),
                   pairs=("k", "size"),
                   df_scaled_median=("df_scaled", "median"),
                   df_scaled_max=("df_scaled", "max"),
                   df_relative_median=("df_relative", "median"),
                   df_angle_median=("df_angle_deg", "median"),
                   df_angle_max=("df_angle_deg", "max"),
                   state_vs_ref_median=("df_from_state", "median"),
                   ref_contrib_median=("df_from_reference", "median"),
                   dU_shift_norm_median=("dU_shift_norm", "median"),
                   dU_shift_norm_max=("dU_shift_norm", "max"),
                   dU_raw_norm_median=("dU_raw_norm", "median"),
                   warm_gap_median=("warm_gap_norm", "median"),
                   warm_gap_max=("warm_gap_norm", "max"),
                   cold_gap_median=("cold_gap_norm", "median"),
                   warm_over_cold_median=("warm_over_cold", "median"),
                   active_change_mean=("active_set_change", "mean"),
                   cond_H=("cond_H", "first"),
                   max_subopt_bound=("subopt_bound_norm", "max")).reset_index())
    summ.to_csv(os.path.join(OUT, "objective_drift_summary.csv"), index=False)

    CONV_TOL = 1e-4      # the tolerance at which the paper declares convergence
    head = {}
    for (key, Nc), g in df.groupby(["system", "Nc"]):
        head[f"{key}_Nc{Nc}"] = {
            "median_warm_gap_norm": float(g.warm_gap_norm.median()),
            "median_cold_gap_norm": float(g.cold_gap_norm.median()),
            "frac_pairs_warm_gap_below_conv_tol": float((g.warm_gap_norm <= CONV_TOL).mean()),
            "median_dU_shift_norm": float(g.dU_shift_norm.median()),
            "median_df_scaled": float(g.df_scaled.median()),
            "median_angle_deg": (float(g.df_angle_deg.median())
                                 if g.df_angle_deg.notna().any() else None),
        }
    verdict = {"pairs_total": int(len(df)), "convergence_tolerance": CONV_TOL,
               "per_system": head, "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "objective_drift.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
