#!/usr/bin/env python3
"""Phase-2C Task 3: certified decision-space dimension sweep (reviewer R2.14).

Reviewer 2 questioned whether the optimization problems were too low-dimensional
(the submitted equal-FE study used N_c = 1, so d = n_u).

This sweeps the control horizon on the CERTIFIED reference set only:

    pendulum-cart, CSTR, flight, HVAC : N_c in {1,2,3,5,8}
    UAV                               : N_c in {1,2}

UAV N_c in {3,5,8} is EXCLUDED because independent reference-solver agreement was
not achieved there (OSQP reports a 1e-20 certificate but multistart L-BFGS-B
disagrees by up to 7.9e-4, above the 1e-4 criterion; see
REFERENCE_NC_CERTIFICATION.md). The exclusion is stated, not hidden.

Frozen subproblems are used, because the question is optimizer search difficulty
rather than closed-loop controller redesign. Decision dimension d = N_c * n_u.

Changing N_c changes more than the dimension: it also changes the conditioning
and the inter-variable coupling of H. Both are therefore recorded and the
analysis reports performance against dimension AND against condition number, so
the two are not silently confounded.

Outputs:
    data/dimension_sweep_raw.csv
    data/dimension_sweep_summary.csv
    data/dimension_sweep_protocol.json
    data/tables/dimension_sweep.tex
    data/figures/gap_vs_dimension_diagnostic.eps
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
# Output root. Defaults to the released data/ directory; set SGO_OUTPUT_DIR to
# redirect every generated file elsewhere (e.g. a scratch directory) so that a
# smoke test cannot overwrite the committed production evidence.
OUT = os.environ.get("SGO_OUTPUT_DIR") or os.path.join(PROJECT, "data")
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, build_problem, solve_one_step, make_cost  # noqa: E402
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import osqp_reference, problem_linear_term      # noqa: E402

CERTIFIED = {"pendcart": [1, 2, 3, 5, 8], "cstr": [1, 2, 3, 5, 8],
             "flight": [1, 2, 3, 5, 8], "hvac": [1, 2, 3, 5, 8],
             "uav": [1, 2]}
EXCLUDED = {"uav": [3, 5, 8]}
TRIALS = 20
BUDGET = P2.FE_BUDGET
CONV_TOL = 1e-4
SEED_BASE = 15_000_000


def instants(key, Nsim, drift):
    """Four predeclared roles per system, chosen from drift data only."""
    d = drift[drift.system == key]
    high = int(d.loc[d.warm_gap_norm.idxmax(), "k"]) if len(d) else Nsim // 3
    cand = {"initial_transient": max(1, int(0.03 * Nsim)),
            "mid_trajectory": int(0.50 * Nsim),
            "near_settled": int(0.90 * Nsim),
            "high_drift": min(max(high, 0), Nsim - 1)}
    seen, out = set(), {}
    for role, k in cand.items():
        if k not in seen:
            out[role] = k; seen.add(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()
    t0 = time.perf_counter()

    ics = P2.load_initial_conditions()
    tuned = P2.load_tuned_np()["selected"]
    systems = get_systems()
    drift = pd.read_csv(os.path.join(OUT, "objective_drift_full_raw.csv"))

    proto = {"certified_cases": CERTIFIED, "excluded_cases": EXCLUDED,
             "exclusion_reason":
                 "UAV N_c >= 3 lacks independent reference-solver agreement: OSQP reports a "
                 "1e-20 KKT certificate but multistart L-BFGS-B disagrees by up to 7.9e-4, "
                 "above the 1e-4 convergence criterion (REFERENCE_NC_CERTIFICATION.md). No "
                 "optimality gap may be computed against that reference.",
             "budget_fe": BUDGET, "trials": args.trials,
             "analysis": "frozen subproblems; performance reported against BOTH decision "
                         "dimension and condition number, since changing N_c changes both.",
             "instants": {}}
    for key in P2.SYSTEMS:
        sysdef = systems[key]
        det = simulate(key, "QP", params=P2.fixed_mpc_params(sysdef), seed=11,
                       verbose=False, x0_override=ics[key]["eval"])
        proto["instants"][key] = instants(key, det["Nsim"], drift)
    with open(os.path.join(OUT, "dimension_sweep_protocol.json"), "w") as f:
        json.dump(proto, f, indent=2)

    rows = []
    for key in P2.SYSTEMS:
        sysdef = systems[key]
        P_h = int(sysdef.P)
        x0 = ics[key]["eval"]
        det = simulate(key, "QP", params=P2.fixed_mpc_params(sysdef), seed=11,
                       verbose=False, x0_override=x0)
        xh = det["x_hist"]
        for Nc in CERTIFIED[key]:
            if Nc > P_h:
                print(f"  {key}: skip Nc={Nc} (> P={P_h})")
                continue
            problem = build_problem(sysdef, Nc, 1.0, P_h)
            lb, ub = problem.lb_seq, problem.ub_seq
            d_dim = len(lb)
            eig = np.linalg.eigvalsh(0.5 * (problem.H + problem.H.T))
            lmin, lmax = float(eig.min()), float(eig.max())
            cond = float(lmax / lmin) if lmin > 0 else np.nan
            for role, k in proto["instants"][key].items():
                k = int(k)
                x = xh[:, k]
                f = problem_linear_term(problem, sysdef, x, k)
                u_star, status, _ = osqp_reference(problem, f)
                if u_star is None:
                    raise RuntimeError(f"OSQP failed {key} Nc={Nc} k={k}: {status}")
                u_star = np.clip(u_star, lb, ub)
                cost = make_cost(problem, x, k)
                J_star = float(cost(u_star))
                n_active = int(np.sum((u_star <= lb + 1e-9) | (u_star >= ub - 1e-9)))
                for alg in P2.ALGS:
                    NP = int(tuned[key][alg]["NP"])
                    params = {"P": P_h, "Nc": Nc, "Q_scale": 1.0}
                    params.update(P2.strict_fe_params(NP, budget=BUDGET, alg=alg))
                    for t in range(args.trials):
                        seed = (SEED_BASE + 1_000_000 * P2.SYSTEM_INDEX[key]
                                + 100_000 * Nc + 1_000 * P2.ALG_INDEX[alg] + t)
                        np.random.seed(seed)
                        sol, J, _, _, wall, nfe = solve_one_step(problem, x, k, alg,
                                                                 params, None)
                        gap = abs(float(J) - J_star) / max(1.0, abs(J_star))
                        rows.append(dict(
                            system=key, Nc=Nc, dim=d_dim, role=role, k=k, alg=alg,
                            NP=NP, trial=t, seed=seed, J_star=J_star, J=float(J),
                            norm_gap=gap, converged_1e4=bool(gap <= CONV_TOL),
                            first_control_error=float(np.linalg.norm(
                                sol[:sysdef.nu] - u_star[:sysdef.nu])),
                            lambda_min=lmin, lambda_max=lmax, cond_H=cond,
                            n_active_reference=n_active,
                            realized_fe=int(nfe), fe_exact=bool(int(nfe) == BUDGET),
                            time_ms=float(wall * 1e3)))
            print(f"  {key:9s} Nc={Nc} d={d_dim:2d} cond={cond:.2e} done "
                  f"({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "dimension_sweep_raw.csv"), index=False)

    summ = (df.groupby(["system", "Nc", "dim", "alg"])
              .agg(n=("norm_gap", "size"), median_gap=("norm_gap", "median"),
                   q1=("norm_gap", lambda s: float(np.percentile(s, 25))),
                   q3=("norm_gap", lambda s: float(np.percentile(s, 75))),
                   conv_rate=("converged_1e4", "mean"),
                   first_ctrl_err=("first_control_error", "median"),
                   cond_H=("cond_H", "first"), lambda_min=("lambda_min", "first"),
                   n_active_reference=("n_active_reference", "mean"),
                   time_ms=("time_ms", "median"),
                   fe_exact=("fe_exact", "all")).reset_index())
    summ.to_csv(os.path.join(OUT, "dimension_sweep_summary.csv"), index=False)

    # Associations: performance vs dimension, and vs conditioning.
    assoc = []
    for alg, g in summ.groupby("alg"):
        rd, pd_ = spearmanr(g.dim, g.median_gap)
        rc, pc = spearmanr(g.cond_H, g.median_gap)
        rdc, pdc = spearmanr(g.dim, g.conv_rate)
        # within-system, to remove between-system confounding
        within = []
        for key, gg in g.groupby("system"):
            if gg.dim.nunique() >= 3:
                r, _ = spearmanr(gg.dim, gg.median_gap)
                within.append(r)
        assoc.append(dict(alg=alg, n_cells=int(len(g)),
                          rho_dim_vs_gap=float(rd), p_dim_vs_gap=float(pd_),
                          rho_cond_vs_gap=float(rc), p_cond_vs_gap=float(pc),
                          rho_dim_vs_conv=float(rdc), p_dim_vs_conv=float(pdc),
                          mean_within_system_rho_dim_vs_gap=float(np.mean(within))
                          if within else np.nan,
                          n_systems_within=len(within)))
    assoc = pd.DataFrame(assoc)
    assoc.to_csv(os.path.join(OUT, "dimension_sweep_association.csv"), index=False)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Decision-space dimension sweep on the certified reference set. "
             r"Frozen subproblems, exact 1000-evaluation budget, 20 trials, median normalized "
             r"gap with strict convergence rate in parentheses. UAV $N_c\ge3$ is excluded "
             r"because independent reference-solver agreement was not achieved there.}",
             r"\label{tab:dimension_sweep}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             r"System & $N_c$ & $d$ & $\kappa(H)$ & SGO & GWO & PSO & WOA \\", r"\midrule"]
    for key in P2.SYSTEMS:
        for Nc in CERTIFIED[key]:
            g = summ[(summ.system == key) & (summ.Nc == Nc)]
            if g.empty:
                continue
            cells = []
            for alg in P2.ALGS:
                r = g[g.alg == alg]
                cells.append(f"{float(r.median_gap.iloc[0]):.1e} ({100*float(r.conv_rate.iloc[0]):.0f}\\%)"
                             if len(r) else "--")
            lines.append(f"{key} & {Nc} & {int(g.dim.iloc[0])} & "
                         f"{float(g.cond_H.iloc[0]):.1e} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "dimension_sweep.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for alg in P2.ALGS:
        g = summ[summ.alg == alg].sort_values("dim")
        axes[0].semilogy(g.dim, np.maximum(g.median_gap, 1e-18), "o", ms=5, label=alg)
        axes[1].loglog(g.cond_H, np.maximum(g.median_gap, 1e-18), "o", ms=5, label=alg)
    for ax, xl in zip(axes, ["decision dimension $d = N_c n_u$", r"condition number $\kappa(H)$"]):
        ax.axhline(CONV_TOL, color="crimson", ls="--", lw=.8)
        ax.set_xlabel(xl); ax.set_ylabel("median normalized gap")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "gap_vs_dimension_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    meta = {"cells": int(len(summ)), "dims": sorted(summ.dim.unique().tolist()),
            "excluded": EXCLUDED, "fe_exact_everywhere": bool(df.fe_exact.all()),
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "dimension_sweep_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.set_option("display.width", 250)
    print(assoc.to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
