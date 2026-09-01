#!/usr/bin/env python3
"""Phase-2B step 3: objective drift for EVERY consecutive Nc=1 transition.

Phase 1 sampled 25 transitions per system, which was diagnostic only. This
computes the drift metrics for every consecutive transition k -> k+1 along the
deterministic reference trajectory of all five systems, under the fixed-MPC
Phase-2A formulation.

The deterministic reference is OSQP (certified at Nc=1 for all five systems, with
>= 10 orders of margin below the 1e-4 convergence criterion; see
REFERENCE_NC_CERTIFICATION.md). Multistart L-BFGS-B is run as an independent
cross-check on a subsample rather than at every step, since 21 extra solves per
step for HVAC's 1500 steps buys nothing once the case is certified.

Metrics per transition, with explicit guards:

  df_abs              ||f_{k+1} - f_k||
  df_over_max         ||df|| / max_k ||f_k||        (primary scaled measure)
  df_relative         ||df|| / ||f_k||              (secondary; NaN when the
                                                     denominator is negligible)
  f_near_zero         flag: ||f_k|| < 1e-6 * max_k ||f_k||
  df_angle_deg        angle(f_k, f_{k+1}); NaN if dim == 1 or either norm negligible
  df_from_state       || 2 G' Qbar Phi (x_c,k+1 - x_c,k) ||
  df_from_reference   || 2 G' Qbar (r_{k+1} - r_k) ||   -- computed, never assumed
  dU_raw              ||U*_{k+1} - U*_k||
  dU_shift            ||U*_{k+1} - shift(U*_k)||   PRIMARY continuity measure.
                      At Nc=1 the receding-horizon shift-and-hold map is the
                      IDENTITY (there is no second block to shift into first
                      place), so dU_shift == dU_raw by construction and is
                      labelled as such rather than pretending a shift occurred.
  dJstar_abs/_norm    |J*_{k+1} - J*_k| and its denominator-protected version
  warm_gap_norm       (J_{k+1}(U_warm) - J*_{k+1}) / max(1,|J*_{k+1}|)
                      PRIMARY optimization-relevance measure
  cold_gap_norm       (J_{k+1}(0) - J*_{k+1}) / max(1,|J*_{k+1}|)
  warm_over_cold      fraction of the cold-start difficulty surviving the warm start
  active_set_change   fraction of decision components changing bound-active status
  lambda_min/max, cond_H   curvature; constant per configuration by construction

Outputs:
    data/objective_drift_full_raw.csv
    data/objective_drift_full_summary.csv
    data/tables/objective_drift_full.tex
    data/figures/objective_drift_full_diagnostic.eps
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
# Output root. Defaults to the released data/ directory; set SGO_OUTPUT_DIR to
# redirect every generated file elsewhere (e.g. a scratch directory) so that a
# smoke test cannot overwrite the committed production evidence.
OUT = os.environ.get("SGO_OUTPUT_DIR") or os.path.join(PROJECT, "data")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, build_problem, make_cost              # noqa: E402
from mpc_utils import compute_f                                        # noqa: E402
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import osqp_reference, lbfgsb_crosscheck        # noqa: E402

NEAR_ZERO_REL = 1e-6
BENIGN_CRITERION = 1e-4        # the 1e-4 optimization-relevance criterion
CROSSCHECK_EVERY = 50          # L-BFGS-B cross-check subsample stride


def shift_sequence(U, nu, Nc):
    """Receding-horizon shift-and-hold, matching sim_common.simulate.

    At Nc == 1 this is the identity map: there is no later block to move into
    first position, so the warm start is simply the previous solution.
    """
    seq = np.asarray(U, dtype=float).copy()
    if Nc > 1:
        seq[:-nu] = seq[nu:]
    return seq


def analyse_system(key, sysdef, x0_eval):
    mpc = P2.fixed_mpc_params(sysdef)
    Nc, P = mpc["Nc"], mpc["P"]
    problem = build_problem(sysdef, Nc, 1.0, P)
    lb, ub = problem.lb_seq, problem.ub_seq
    nu = sysdef.nu

    traj = simulate(key, "QP", params=mpc, seed=11, verbose=False, x0_override=x0_eval)
    xh, Nsim = traj["x_hist"], traj["Nsim"]

    eig = np.linalg.eigvalsh(0.5 * (problem.H + problem.H.T))
    lmin, lmax = float(eig.min()), float(eig.max())
    cond = float(lmax / lmin) if lmin > 0 else np.nan

    # Pass 1: linear terms, reference vectors, and the OSQP optimum at every step.
    fs, refs, xcs, Us, Js = {}, {}, {}, {}, {}
    for k in range(Nsim + 1):
        x_c = sysdef.cost_state(xh[:, k])
        ref = sysdef.reference_horizon(k, P)
        f = compute_f(problem.Phi, problem.Gamma, problem.Q_block, x_c, ref)
        xcs[k], refs[k], fs[k] = x_c, ref, f
        u, status, _ = osqp_reference(problem, f)
        if u is None:
            raise RuntimeError(f"OSQP failed at {key} k={k}: {status}")
        Us[k] = np.clip(u, lb, ub)
        Js[k] = float(make_cost(problem, xh[:, k], k)(Us[k]))

    f_scale = max(float(np.max([np.linalg.norm(fs[k]) for k in range(Nsim + 1)])), 1e-300)
    M = 2.0 * problem.Gamma.T @ problem.Q_block

    rows = []
    for k in range(Nsim):
        f0, f1 = fs[k], fs[k + 1]
        n0, n1 = float(np.linalg.norm(f0)), float(np.linalg.norm(f1))
        df = f1 - f0
        ndf = float(np.linalg.norm(df))
        near_zero = bool(n0 < NEAR_ZERO_REL * f_scale)

        d_state = float(np.linalg.norm(M @ (problem.Phi @ (xcs[k + 1] - xcs[k]))))
        d_ref = float(np.linalg.norm(M @ (refs[k + 1] - refs[k])))

        u0, u1 = Us[k], Us[k + 1]
        u0_shift = np.clip(shift_sequence(u0, nu, Nc), lb, ub)
        cost1 = make_cost(problem, xh[:, k + 1], k + 1)
        J1 = Js[k + 1]
        den1 = max(1.0, abs(J1))
        warm_gap = (float(cost1(u0_shift)) - J1) / den1
        cold_gap = (float(cost1(np.zeros_like(lb))) - J1) / den1

        a0 = (u0 <= lb + 1e-9).astype(int) - (u0 >= ub - 1e-9).astype(int)
        a1 = (u1 <= lb + 1e-9).astype(int) - (u1 >= ub - 1e-9).astype(int)

        cross = {}
        if k % CROSSCHECK_EVERY == 0:
            _, J_l, info = lbfgsb_crosscheck(problem, xh[:, k], k, n_restarts=20, seed=900 + k)
            cross = {"J_lbfgsb_crosscheck": J_l,
                     "crosscheck_rel_diff": abs(J_l - Js[k]) / max(1.0, abs(Js[k])),
                     "lbfgsb_multistart_spread": info["multistart_spread"]}

        rows.append(dict(
            system=key, Nc=Nc, dim=int(len(lb)), k=k, Nsim=Nsim,
            t_seconds=k * float(sysdef.dt),
            f_norm_k=n0, f_norm_k1=n1, f_scale_max=f_scale,
            df_abs=ndf, df_over_max=ndf / f_scale,
            df_relative=(ndf / n0 if not near_zero else np.nan),
            f_near_zero=near_zero,
            df_angle_deg=(float(np.degrees(np.arccos(np.clip(
                np.dot(f0, f1) / (n0 * n1), -1.0, 1.0))))
                if (len(lb) > 1 and n0 > NEAR_ZERO_REL * f_scale
                    and n1 > NEAR_ZERO_REL * f_scale) else np.nan),
            df_from_state=d_state, df_from_reference=d_ref,
            reference_is_constant=bool(d_ref == 0.0),
            dU_raw=float(np.linalg.norm(u1 - u0)),
            dU_shift=float(np.linalg.norm(u1 - u0_shift)),
            shift_is_identity=bool(Nc == 1),
            J_star_k=Js[k], J_star_k1=J1,
            dJstar_abs=abs(J1 - Js[k]),
            dJstar_norm=abs(J1 - Js[k]) / max(1.0, abs(Js[k])),
            warm_gap_norm=warm_gap, cold_gap_norm=cold_gap,
            warm_over_cold=(warm_gap / cold_gap if cold_gap > 0 else np.nan),
            benign=bool(warm_gap <= BENIGN_CRITERION),
            active_set_change=float(np.mean(a0 != a1)),
            n_active_k=int(np.sum(a0 != 0)),
            lambda_min=lmin, lambda_max=lmax, cond_H=cond, **cross))
    return rows


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    t0 = time.perf_counter()
    ics = P2.load_initial_conditions()
    rows = []
    for key, sysdef in get_systems().items():
        print(f"full drift: {key}")
        rows += analyse_system(key, sysdef, ics[key]["eval"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "objective_drift_full_raw.csv"), index=False)

    summ = (df.groupby("system")
              .agg(transitions=("k", "size"), Nsim=("Nsim", "first"), dim=("dim", "first"),
                   benign_fraction=("benign", "mean"),
                   warm_gap_median=("warm_gap_norm", "median"),
                   warm_gap_p90=("warm_gap_norm", lambda s: float(np.percentile(s, 90))),
                   warm_gap_p99=("warm_gap_norm", lambda s: float(np.percentile(s, 99))),
                   warm_gap_max=("warm_gap_norm", "max"),
                   cold_gap_median=("cold_gap_norm", "median"),
                   warm_over_cold_median=("warm_over_cold", "median"),
                   df_over_max_median=("df_over_max", "median"),
                   df_over_max_max=("df_over_max", "max"),
                   angle_median=("df_angle_deg", "median"),
                   state_contrib_median=("df_from_state", "median"),
                   ref_contrib_median=("df_from_reference", "median"),
                   ref_contrib_max=("df_from_reference", "max"),
                   reference_constant=("reference_is_constant", "all"),
                   dU_shift_median=("dU_shift", "median"),
                   dU_shift_max=("dU_shift", "max"),
                   active_change_mean=("active_set_change", "mean"),
                   cond_H=("cond_H", "first"),
                   crosscheck_max_rel=("crosscheck_rel_diff", "max")).reset_index())
    summ.to_csv(os.path.join(OUT, "objective_drift_full_summary.csv"), index=False)

    # Where in time are the severe transitions?
    conc = []
    for key, g in df.groupby("system"):
        thr = float(np.percentile(g.warm_gap_norm, 90))
        sev = g[g.warm_gap_norm >= max(thr, BENIGN_CRITERION)]
        frac_first10 = float((sev.k < 0.10 * g.Nsim.iloc[0]).mean()) if len(sev) else np.nan
        conc.append(dict(system=key, n_nonbenign=int((~g.benign).sum()),
                         nonbenign_fraction=float((~g.benign).mean()),
                         p90_threshold=thr,
                         severe_in_first_10pct_of_trajectory=frac_first10,
                         median_k_of_nonbenign=(float(g[~g.benign].k.median())
                                                if (~g.benign).any() else np.nan),
                         Nsim=int(g.Nsim.iloc[0])))
    conc = pd.DataFrame(conc)
    conc.to_csv(os.path.join(OUT, "objective_drift_full_concentration.csv"), index=False)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Objective drift over EVERY consecutive transition of the deterministic "
             r"trajectory ($N_c=1$, fixed MPC). The warm-start suboptimality is the "
             r"optimization-relevant measure: the cost of reusing the previous optimum for the "
             r"next problem, normalized as the optimality gaps are. A transition is called "
             r"benign when it falls below the $10^{-4}$ criterion at which the study declares "
             r"convergence.}",
             r"\label{tab:objective_drift_full}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             r"System & Transitions & Benign & Warm-gap median & p90 & p99 & max & "
             r"Ref.-driven drift \\", r"\midrule"]
    for _, r in summ.iterrows():
        lines.append(f"{r.system} & {int(r.transitions)} & {100*r.benign_fraction:.1f}\\% & "
                     f"{r.warm_gap_median:.2e} & {r.warm_gap_p90:.2e} & {r.warm_gap_p99:.2e} & "
                     f"{r.warm_gap_max:.2e} & "
                     f"{'none' if r.reference_constant else f'{r.ref_contrib_median:.2e}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "objective_drift_full.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Diagnostic: warm-gap vs time, log scale, with the benign criterion marked.
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
    for ax, key in zip(axes, P2.SYSTEMS):
        g = df[df.system == key]
        ax.semilogy(g.k, np.maximum(g.warm_gap_norm, 1e-18), lw=.8, color="#1f77b4")
        ax.axhline(BENIGN_CRITERION, color="crimson", ls="--", lw=1)
        ax.set_title(f"{key} ({100*float(g.benign.mean()):.0f}% benign)")
        ax.set_xlabel("MPC step k")
        if key == P2.SYSTEMS[0]:
            ax.set_ylabel("warm-start suboptimality (norm.)")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "objective_drift_full_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    meta = {"total_transitions": int(len(df)),
            "benign_criterion": BENIGN_CRITERION,
            "overall_benign_fraction": float(df.benign.mean()),
            "per_system_benign_fraction": {k: float(g.benign.mean())
                                           for k, g in df.groupby("system")},
            "max_crosscheck_rel_diff": float(df.crosscheck_rel_diff.max(skipna=True)),
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "objective_drift_full_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(conc.to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
