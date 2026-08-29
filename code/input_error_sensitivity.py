#!/usr/bin/env python3
"""Phase-2C Task 2: first-input-error / closed-loop sensitivity (reviewer R2.16).

Reviewer 2 asked why poor frozen-subproblem accuracy does not always imply poor
closed-loop tracking. Phase 2B kept the two metrics strictly separate:

    finite-horizon optimization accuracy   (normalized gap vs the OSQP reference)
    closed-loop tracking                   (nRMSE over the trajectory)

This quantifies the link between them through the only channel that connects
them in a receding-horizon loop: the FIRST APPLIED CONTROL INPUT. Only that
input reaches the plant; the rest of the horizon is discarded.

Per system x optimizer x seed, replaying the Phase-2A Regime-A protocol with the
matched seed, at every MPC step k:

    e_u,k     = || u_alg,k - u_ref,k ||        first applied input only
    gap_k     = normalized finite-horizon objective gap at step k
    drift_k   = objective-drift severity of transition k (Phase-2B)
    track_k   = normalized state tracking error at step k
    dtrack_k  = change in tracking error into step k+1

Associations reported (rank-based, with bootstrap CIs). No causal claim is made:
this is an observational link between quantities measured on the same runs.

Outputs:
    data/input_error_sensitivity_raw.csv
    data/input_error_sensitivity_summary.csv
    data/tables/input_error_sensitivity.tex
    data/figures/input_error_sensitivity_diagnostic.eps
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(PROJECT, "data")
CACHE = os.path.join(OUT, "cache")
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, build_problem, make_cost, normalized_rmse  # noqa: E402
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import osqp_reference, problem_linear_term      # noqa: E402

SEEDS = 5           # matched Phase-2A Regime-A seeds per optimizer
BOOT = 4000
RNG = np.random.default_rng(20260817)


def boot_ci_spearman(x, y, n=BOOT, alpha=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, p = spearmanr(x, y)
    idx = RNG.integers(0, len(x), (n, len(x)))
    vals = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float(r), float(p), np.nan, np.nan
    return (float(r), float(p),
            float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def reference_trajectory_quantities(key, sysdef, x0_eval):
    """Deterministic reference inputs and per-step optimum, under fixed MPC."""
    mpc = P2.fixed_mpc_params(sysdef)
    problem = build_problem(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"])
    det = simulate(key, "QP", params=mpc, seed=11, verbose=False, x0_override=x0_eval)
    xh, Nsim = det["x_hist"], det["Nsim"]
    u_ref = np.zeros((sysdef.nu, Nsim))
    J_star = np.zeros(Nsim)
    for k in range(Nsim):
        f = problem_linear_term(problem, sysdef, xh[:, k], k)
        u, status, _ = osqp_reference(problem, f)
        if u is None:
            raise RuntimeError(f"OSQP failed {key} k={k}: {status}")
        u = np.clip(u, problem.lb_seq, problem.ub_seq)
        u_ref[:, k] = u[:sysdef.nu]
        J_star[k] = float(make_cost(problem, xh[:, k], k)(u))
    return problem, det, u_ref, J_star, Nsim


def tracking_error(sysdef, x, k):
    """Normalized tracking error of the evaluated states at one step."""
    idx = sysdef.metric_indices
    if sysdef.cost_mode == "trajectory":
        ref = sysdef.ref_traj[:, min(k, sysdef.ref_traj.shape[1] - 1)]
        scales = [max(np.ptp(sysdef.ref_traj[i, :]), 1.0) for i in idx]
    else:
        ref = sysdef.reference_at(0)
        scales = [max(abs(sysdef.x0[i] - ref[i]), abs(ref[i]), 1.0) for i in idx]
    return float(np.mean([abs(x[i] - ref[i]) / s for i, s in zip(idx, scales)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    t0 = time.perf_counter()

    ics = P2.load_initial_conditions()
    tuned = P2.load_tuned_np()["selected"]
    systems = get_systems()
    drift_all = pd.read_csv(os.path.join(OUT, "objective_drift_full_raw.csv"))

    rows = []
    for key in P2.SYSTEMS:
        sysdef = systems[key]
        mpc = P2.fixed_mpc_params(sysdef)
        x0 = ics[key]["eval"]
        problem, det, u_ref, J_star, Nsim = reference_trajectory_quantities(key, sysdef, x0)
        det_nrmse = normalized_rmse(det["x_hist"], sysdef, x0=x0)
        dmap = (drift_all[drift_all.system == key]
                .set_index("k").warm_gap_norm.to_dict())

        for alg in P2.ALGS:
            NP = int(tuned[key][alg]["NP"])
            params = dict(mpc)
            params.update(P2.strict_fe_params(NP, alg=alg))
            for t in range(args.seeds):
                seed = P2.seed_for("primary", P2.SYSTEM_INDEX[key], 0,
                                   P2.ALG_INDEX[alg], t)
                cache = os.path.join(CACHE, f"optwarm_{key}_{alg}_{seed}.npy")
                out = simulate(key, alg, params=params, seed=seed, verbose=False,
                               x0_override=x0)
                xh, uh = out["x_hist"], out["u_hist"]
                run_nrmse = normalized_rmse(xh, sysdef, x0=x0)
                for k in range(Nsim):
                    # Finite-horizon gap of THIS optimizer's step-k subproblem,
                    # evaluated on the state the optimizer itself reached.
                    f = problem_linear_term(problem, sysdef, xh[:, k], k)
                    u_o, status, _ = osqp_reference(problem, f)
                    if u_o is None:
                        continue
                    u_o = np.clip(u_o, problem.lb_seq, problem.ub_seq)
                    cost_k = make_cost(problem, xh[:, k], k)
                    Jk_star = float(cost_k(u_o))
                    seq = np.tile(uh[:, k], mpc["Nc"])
                    gap = abs(float(cost_k(seq)) - Jk_star) / max(1.0, abs(Jk_star))
                    tr_k = tracking_error(sysdef, xh[:, k], k)
                    tr_k1 = tracking_error(sysdef, xh[:, k + 1], k + 1)
                    rows.append(dict(
                        system=key, alg=alg, seed=seed, trial=t, k=k,
                        e_u=float(np.linalg.norm(uh[:, k] - u_ref[:, k])),
                        e_u_rel=float(np.linalg.norm(uh[:, k] - u_ref[:, k]) /
                                      max(float(np.max(problem.ub_seq - problem.lb_seq)), 1e-30)),
                        gap_k=gap, drift_k=float(dmap.get(k, np.nan)),
                        track_k=tr_k, track_k1=tr_k1, dtrack=tr_k1 - tr_k,
                        run_nrmse=run_nrmse, reference_nrmse=det_nrmse))
        print(f"  {key}: done ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "input_error_sensitivity_raw.csv"), index=False)

    # ---- associations -------------------------------------------------------
    recs = []
    for (key, alg), g in df.groupby(["system", "alg"]):
        r1, p1, lo1, hi1 = boot_ci_spearman(g.gap_k, g.e_u)
        r2, p2, lo2, hi2 = boot_ci_spearman(g.e_u, g.dtrack)
        r3, p3, lo3, hi3 = boot_ci_spearman(g.drift_k, g.e_u)
        recs.append(dict(system=key, alg=alg, n_steps=int(len(g)),
                         rho_gap_vs_eu=r1, p_gap_vs_eu=p1, lo_gap_vs_eu=lo1, hi_gap_vs_eu=hi1,
                         rho_eu_vs_dtrack=r2, p_eu_vs_dtrack=p2,
                         lo_eu_vs_dtrack=lo2, hi_eu_vs_dtrack=hi2,
                         rho_drift_vs_eu=r3, p_drift_vs_eu=p3,
                         mean_eu=float(g.e_u.mean()), median_eu=float(g.e_u.median()),
                         mean_eu_rel=float(g.e_u_rel.mean()),
                         run_nrmse=float(g.run_nrmse.mean()),
                         reference_nrmse=float(g.reference_nrmse.iloc[0])))
    assoc = pd.DataFrame(recs)

    # System-level sensitivity: run-level mean input error vs run-level nRMSE.
    sysrec = []
    for key, g in df.groupby("system"):
        runs = g.groupby(["alg", "seed"]).agg(mean_eu=("e_u", "mean"),
                                              nrmse=("run_nrmse", "first"),
                                              ref=("reference_nrmse", "first")).reset_index()
        r, p, lo, hi = boot_ci_spearman(runs.mean_eu, runs.nrmse)
        # sensitivity slope: relative nRMSE excess per unit mean input error
        excess = (runs.nrmse - runs.ref) / runs.ref
        s, ps, slo, shi = boot_ci_spearman(runs.mean_eu, excess)
        sysrec.append(dict(system=key, n_runs=int(len(runs)),
                           rho_meaneu_vs_nrmse=r, p=p, lo=lo, hi=hi,
                           rho_meaneu_vs_relative_excess=s, p_excess=ps,
                           median_relative_excess=float(np.median(excess)),
                           max_relative_excess=float(np.max(excess)),
                           mean_eu_range=float(runs.mean_eu.max() - runs.mean_eu.min())))
    syslevel = pd.DataFrame(sysrec)

    assoc.to_csv(os.path.join(OUT, "input_error_sensitivity_summary.csv"), index=False)
    syslevel.to_csv(os.path.join(OUT, "input_error_system_sensitivity.csv"), index=False)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Association between finite-horizon optimization error, first applied "
             r"input error, and downstream tracking, under the fixed-MPC exact-FE protocol. "
             r"Spearman $\rho$ with bootstrap 95\% CIs. These are observational associations "
             r"on the same runs; no causal claim is made.}",
             r"\label{tab:input_error_sensitivity}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"System & Opt. & $\rho$(gap, $e_u$) [CI] & $\rho$($e_u$, $\Delta$track) [CI] & "
             r"mean $e_u$ \\", r"\midrule"]
    for _, r in assoc.iterrows():
        lines.append(f"{r.system} & {r.alg} & "
                     f"${r.rho_gap_vs_eu:+.2f}$ [{r.lo_gap_vs_eu:+.2f}, {r.hi_gap_vs_eu:+.2f}] & "
                     f"${r.rho_eu_vs_dtrack:+.2f}$ [{r.lo_eu_vs_dtrack:+.2f}, {r.hi_eu_vs_dtrack:+.2f}] & "
                     f"{r.mean_eu:.3e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "input_error_sensitivity.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 5, figsize=(19, 7))
    for j, key in enumerate(P2.SYSTEMS):
        g = df[df.system == key]
        ax = axes[0, j]
        for alg in P2.ALGS:
            gg = g[g.alg == alg]
            ax.loglog(np.maximum(gg.gap_k, 1e-18), np.maximum(gg.e_u, 1e-18),
                      ".", ms=1.5, alpha=.35, label=alg)
        ax.set_title(key); ax.set_xlabel("finite-horizon gap")
        if j == 0:
            ax.set_ylabel("first-input error $e_u$"); ax.legend(fontsize=6, markerscale=4)
        ax.grid(alpha=.3)
        ax = axes[1, j]
        runs = g.groupby(["alg", "seed"]).agg(m=("e_u", "mean"), n=("run_nrmse", "first"),
                                              r=("reference_nrmse", "first")).reset_index()
        for alg in P2.ALGS:
            rr = runs[runs.alg == alg]
            ax.loglog(np.maximum(rr.m, 1e-18), rr.n, "o", ms=4, label=alg)
        ax.axhline(runs.r.iloc[0], color="k", ls="--", lw=1)
        ax.set_xlabel("run mean $e_u$")
        if j == 0:
            ax.set_ylabel("run nRMSE")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "input_error_sensitivity_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    meta = {"rows": int(len(df)), "seeds_per_optimizer": args.seeds,
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "input_error_sensitivity_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.set_option("display.width", 250)
    print(assoc[["system", "alg", "rho_gap_vs_eu", "rho_eu_vs_dtrack",
                 "mean_eu", "run_nrmse", "reference_nrmse"]].to_string(index=False))
    print()
    print(syslevel.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
