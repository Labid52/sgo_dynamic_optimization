#!/usr/bin/env python3
"""Phase-2B step 8: budget sensitivity across multiple difficult instances (R2.10).

The submitted study drew its budget conclusion from a single instance (UAV k=0)
and parameterised the x-axis by (NP, MaxIter), which is not a budget: the four
optimizers consumed between 362 and 1220 evaluations at the nominal "(20, 60)"
setting. Both problems are fixed here.

Instance selection is made from quantities INDEPENDENT of this experiment:
  * drift severity from the frozen strata (High or Severe), and/or
  * top-decile cold-start difficulty measured in the dense frozen sweep.
UAV k=0 is retained if it still qualifies on those criteria.

The x-axis is the exact number of objective-function evaluations. Budgets are
enforced per objective call with early stopping and stagnation disabled, and
trial seeds are PAIRED across budget levels so the budget effect is a
within-trial comparison rather than a reshuffle of random starts.

Outputs:
    data/budget_sensitivity_v2_raw.csv
    data/budget_sensitivity_v2_summary.csv
    data/budget_instance_selection.json
    data/tables/budget_sensitivity_v2.tex
    data/figures/budget_vs_gap_diagnostic.eps
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
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, build_problem, solve_one_step, make_cost  # noqa: E402
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import osqp_reference, problem_linear_term      # noqa: E402

# Budgets on the only meaningful axis: exact objective-function evaluations.
# The smallest must exceed the largest population size in use (30).
BUDGETS = [250, 500, 1000, 2000, 5000]
TRIALS = 30
CONV_TOL = 1e-4
N_INSTANCES_TARGET = 8


def select_instances():
    """Difficult instances, chosen from drift strata and cold-start difficulty."""
    sweep = pd.read_csv(os.path.join(OUT, "frozen_sweep_v2_raw.csv"))
    cold = sweep[sweep.start == "cold"]
    # Per-instance difficulty: median cold-start final gap across all optimizers.
    diff = (cold.groupby(["system", "k"])
            .agg(median_cold_gap=("final_gap_norm", "median"),
                 stratum=("stratum", "first"),
                 drift=("warm_gap_norm", "first"),
                 conv=("converged_1e4", "mean")).reset_index())
    thr = float(np.percentile(diff.median_cold_gap, 90))
    hard_by_gap = diff[diff.median_cold_gap >= thr]
    hard_by_drift = diff[diff.stratum.isin(["High", "Severe"])]
    pool = pd.concat([hard_by_gap, hard_by_drift]).drop_duplicates(["system", "k"])

    # Spread across systems: take the hardest few from each contributing system.
    chosen = []
    per_sys = max(1, N_INSTANCES_TARGET // max(pool.system.nunique(), 1))
    for key, g in pool.groupby("system"):
        chosen.append(g.nlargest(per_sys + 1, "median_cold_gap"))
    sel = pd.concat(chosen).nlargest(max(N_INSTANCES_TARGET, 6), "median_cold_gap")

    # Retain UAV k=0 if it legitimately qualifies.
    uav0 = diff[(diff.system == "uav") & (diff.k == 0)]
    uav0_qualifies = bool(len(uav0) and (
        float(uav0.median_cold_gap.iloc[0]) >= thr or
        uav0.stratum.iloc[0] in ("High", "Severe")))
    if uav0_qualifies and not ((sel.system == "uav") & (sel.k == 0)).any():
        sel = pd.concat([sel, uav0])

    meta = {"criteria": {
                "top_decile_cold_start_difficulty": f"median cold-start gap >= {thr:.3e}",
                "drift_strata": "High or Severe"},
            "selection_independent_of_budget_experiment": True,
            "uav_k0_qualifies_on_criteria": uav0_qualifies,
            "n_selected": int(len(sel)),
            "systems_covered": sorted(sel.system.unique().tolist()),
            "instances": [{"system": r.system, "k": int(r.k), "stratum": r.stratum,
                           "drift_warm_gap": float(r.drift),
                           "median_cold_gap": float(r.median_cold_gap),
                           "cold_conv_rate": float(r.conv)}
                          for _, r in sel.iterrows()]}
    with open(os.path.join(OUT, "budget_instance_selection.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return sel, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()
    t0 = time.perf_counter()

    sel, meta = select_instances()
    print(json.dumps({k: v for k, v in meta.items() if k != "instances"}, indent=2))
    for i in meta["instances"]:
        print(f"  {i['system']:9s} k={i['k']:4d} {i['stratum']:9s} "
              f"drift={i['drift_warm_gap']:.2e} cold gap={i['median_cold_gap']:.2e}")

    ics = P2.load_initial_conditions()
    tuned = P2.load_tuned_np()["selected"]
    systems = get_systems()
    rows = []
    for inst_i, inst in enumerate(meta["instances"]):
        key, k = inst["system"], int(inst["k"])
        sysdef = systems[key]
        mpc = P2.fixed_mpc_params(sysdef)
        problem = build_problem(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"])
        lb, ub = problem.lb_seq, problem.ub_seq
        xh = simulate(key, "QP", params=mpc, seed=11, verbose=False,
                      x0_override=ics[key]["eval"])["x_hist"]
        x = xh[:, k]
        f = problem_linear_term(problem, sysdef, x, k)
        u_star, status, _ = osqp_reference(problem, f)
        if u_star is None:
            raise RuntimeError(f"OSQP failed at {key} k={k}: {status}")
        u_star = np.clip(u_star, lb, ub)
        cost = make_cost(problem, x, k)
        J_star = float(cost(u_star))

        for alg in P2.ALGS:
            NP = int(tuned[key][alg]["NP"])
            for B in BUDGETS:
                params = dict(mpc)
                params.update(P2.strict_fe_params(NP, budget=B, alg=alg))
                for t in range(args.trials):
                    # Seed depends on trial only (not on B), so budget levels are paired.
                    seed = P2.seed_for("budget_v2", inst_i, 0, P2.ALG_INDEX[alg], t)
                    np.random.seed(seed)
                    sol, J, _, _, wall, nfe = solve_one_step(problem, x, k, alg,
                                                             params, None)
                    gap = abs(float(J) - J_star) / max(1.0, abs(J_star))
                    rows.append(dict(
                        system=key, k=k, stratum=inst["stratum"],
                        drift_warm_gap=inst["drift_warm_gap"],
                        alg=alg, NP=NP, budget_fe=B, trial=t, seed=seed,
                        J_star=J_star, J=float(J), norm_gap=gap,
                        converged_1e4=bool(gap <= CONV_TOL),
                        first_control_error=float(np.linalg.norm(
                            sol[:sysdef.nu] - u_star[:sysdef.nu])),
                        realized_fe=int(nfe), fe_exact=bool(int(nfe) == B),
                        time_ms=float(wall * 1e3)))
        print(f"  {key} k={k} done ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "budget_sensitivity_v2_raw.csv"), index=False)
    summ = (df.groupby(["system", "k", "stratum", "alg", "budget_fe"])
              .agg(n=("norm_gap", "size"),
                   median_gap=("norm_gap", "median"),
                   q1=("norm_gap", lambda s: float(np.percentile(s, 25))),
                   q3=("norm_gap", lambda s: float(np.percentile(s, 75))),
                   conv_rate=("converged_1e4", "mean"),
                   first_ctrl_err_median=("first_control_error", "median"),
                   time_ms_median=("time_ms", "median"),
                   fe_exact=("fe_exact", "all")).reset_index())
    summ.to_csv(os.path.join(OUT, "budget_sensitivity_v2_summary.csv"), index=False)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Budget sensitivity across difficult frozen instances. The budget axis is "
             r"the exact number of objective-function evaluations, enforced per objective call "
             r"with early stopping disabled; trial seeds are paired across budgets. Instances "
             r"were selected from drift strata and cold-start difficulty measured independently "
             r"of this experiment.}",
             r"\label{tab:budget_sensitivity_v2}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r"Instance & Opt. & " + " & ".join(f"{b} FE" for b in BUDGETS) + r" \\",
             r"\midrule"]
    for (key, k), g in summ.groupby(["system", "k"]):
        for alg in P2.ALGS:
            gg = g[g.alg == alg].set_index("budget_fe")
            cells = []
            for B in BUDGETS:
                if B in gg.index:
                    cells.append(f"{gg.loc[B,'median_gap']:.2e} ({100*gg.loc[B,'conv_rate']:.0f}\\%)")
                else:
                    cells.append("--")
            lines.append(f"{key} $k{{=}}{k}$ & {alg} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "budget_sensitivity_v2.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    inst_list = summ[["system", "k"]].drop_duplicates().values.tolist()
    n = len(inst_list)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    for ax, (key, k) in zip(axes.ravel(), inst_list):
        g = summ[(summ.system == key) & (summ.k == k)]
        for alg in P2.ALGS:
            gg = g[g.alg == alg].sort_values("budget_fe")
            ax.loglog(gg.budget_fe, np.maximum(gg.median_gap, 1e-18), "o-", ms=4, label=alg)
        ax.axhline(CONV_TOL, color="crimson", ls="--", lw=.8)
        ax.set_title(f"{key} k={k} ({g.stratum.iloc[0]})")
        ax.set_xlabel("objective-function evaluations"); ax.grid(alpha=.3)
        ax.set_ylabel("median normalized gap")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "budget_vs_gap_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    m = {"instances": n, "budgets": BUDGETS, "trials": args.trials,
         "fe_exact_everywhere": bool(df.fe_exact.all()),
         "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "budget_sensitivity_v2_meta.json"), "w") as f:
        json.dump(m, f, indent=2)
    pd.set_option("display.width", 250)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
