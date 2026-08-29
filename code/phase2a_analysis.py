#!/usr/bin/env python3
"""Phase-2A step 11-12: LaTeX tables, diagnostic figures, and the descriptive
comparison against the submitted jointly tuned results.

No statistical testing or effect sizes here -- those belong to a later phase.
This produces the artifacts the Phase-2A report needs and answers, per system:

  * did the optimizer ranking change when the MPC-design confound was removed?
  * is SGO relatively stronger, weaker or similar?
  * did exact FE accounting change the picture?

Outputs:
    data/tables/primary_nrmse.tex
    data/tables/primary_budget_accounting.tex
    data/tables/primary_timing.tex
    data/tables/reference_crosscheck.tex
    data/figures/primary_nrmse_box.eps
    data/figures/primary_nfe_check.eps
    data/submitted_vs_revised_comparison.csv
    data/submitted_vs_revised_summary.json
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(PROJECT, "data")
SUBMITTED = os.path.join(OUT, "mpc_submitted_protocol")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
sys.path.insert(0, ROOT)
import phase2_protocol as P2                                           # noqa: E402

LABEL = {"pendcart": "Pendulum-cart", "cstr": "CSTR", "flight": "Flight control",
         "hvac": "HVAC", "uav": "UAV"}
REGIME_LABEL = {"A_tuned_NP": "A (tuned $NP$, exact FE)",
                "B_common_NP": "B (common $NP{=}20$, exact FE)"}


def _fmt(v, sig=4):
    if not np.isfinite(v):
        return "--"
    return f"${v:.{sig}g}$"


def _sci(v, sig=3):
    if not np.isfinite(v):
        return "--"
    s = f"{v:.{sig}e}"
    m, e = s.split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def table_nrmse(summ):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Closed-loop normalized RMSE under the fixed-MPC primary experiment. "
             r"The MPC formulation is identical for all optimizers and for the deterministic "
             r"reference. Every run consumes exactly the same number of objective evaluations "
             r"per MPC step. Mean $\pm$ standard deviation over 20 runs; the deterministic "
             r"reference is a single value.}",
             r"\label{tab:primary_nrmse}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Regime & System & SGO & GWO & PSO & WOA & Deterministic ref. \\", r"\midrule"]
    for regime in ["A_tuned_NP", "B_common_NP"]:
        for i, key in enumerate(P2.SYSTEMS):
            g = summ[(summ.system == key) & (summ.regime == regime)]
            if g.empty:
                continue
            cells = []
            for alg in P2.ALGS:
                r = g[g.alg == alg]
                if r.empty:
                    cells.append("--"); continue
                m, s = float(r.nrmse_mean.iloc[0]), float(r.nrmse_std.iloc[0])
                best = int(r.rank_in_system_regime.iloc[0]) == 1
                cell = f"{m:.5e}".replace("e", r"{\times}10^{") + "}"
                cell = f"${m:.5g}\\!\\pm\\!{s:.2g}$"
                cells.append(r"\textbf{" + cell + "}" if best else cell)
            ref = float(g.reference_nrmse.iloc[0])
            head = REGIME_LABEL[regime] if i == 0 else ""
            lines.append(f"{head} & {LABEL[key]} & " + " & ".join(cells) +
                         f" & ${ref:.5g}$ \\\\")
        lines.append(r"\midrule" if regime == "A_tuned_NP" else "")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(x for x in lines if x != "") + "\n"


def table_budget(summ):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Evaluation-budget accounting. The nominal budget is enforced at the "
             r"objective-call level, with early stopping and stagnation stopping disabled, so "
             r"the realized number of objective evaluations per MPC step is identical for every "
             r"optimizer.}",
             r"\label{tab:primary_budget}",
             r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Regime & System & Optimizer & $NP$ & Nominal FE & Realized FE/step & Exact \\",
             r"\midrule"]
    for regime in ["A_tuned_NP", "B_common_NP"]:
        for key in P2.SYSTEMS:
            g = summ[(summ.system == key) & (summ.regime == regime)]
            for _, r in g.iterrows():
                lines.append(f"{REGIME_LABEL[regime]} & {LABEL[key]} & {r.alg} & "
                             f"{int(r.NP)} & {P2.FE_BUDGET} & {r.nfe_per_step_mean:.1f} & "
                             f"{'yes' if bool(r.nfe_exact) else 'NO'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_timing(summ):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Computation time per MPC step (unoptimized single-process Python, "
             r"single-threaded BLAS). $dt$ is the sampling interval. These are relative "
             r"measurements and are not a deployability claim.}",
             r"\label{tab:primary_timing}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
             r"Regime & System & Opt. & Mean (ms) & Median (ms) & p95 (ms) & Max (ms) & "
             r"$dt$ (ms) & p95/$dt$ & Max/$dt$ \\", r"\midrule"]
    for regime in ["A_tuned_NP", "B_common_NP"]:
        for key in P2.SYSTEMS:
            g = summ[(summ.system == key) & (summ.regime == regime)]
            for _, r in g.iterrows():
                lines.append(
                    f"{REGIME_LABEL[regime]} & {LABEL[key]} & {r.alg} & "
                    f"{r.time_ms_mean:.2f} & {r.time_ms_median:.2f} & {r.time_ms_p95:.2f} & "
                    f"{r.time_ms_max:.2f} & {r.dt_s*1e3:.0f} & {r.p95_over_dt:.3f} & "
                    f"{r.max_over_dt:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_crosscheck(cross):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Deterministic reference verification. OSQP is the primary reference "
             r"solver and multistart L-BFGS-B the independent cross-check; both are evaluated "
             r"on the implemented objective. The suboptimality bound is the strong-convexity "
             r"certificate $\|\mathrm{proj}\,\nabla J\|^2/(2\lambda_{\min})$, normalized as the "
             r"optimality gaps are.}",
             r"\label{tab:reference_crosscheck}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"System & Instances & Max $|J_{\mathrm{OSQP}}-J_{\mathrm{L\text{-}BFGS\text{-}B}}|$ "
             r"& Max rel. & Max $\|u\|_\infty$ diff & Max subopt. bound & OSQP status \\",
             r"\midrule"]
    for key in P2.SYSTEMS:
        g = cross[cross.system == key]
        if g.empty:
            continue
        statuses = sorted(set(g.osqp_status.astype(str)))
        lines.append(f"{LABEL[key]} & {len(g)} & {_sci(g.obj_abs_diff.max())} & "
                     f"{_sci(g.obj_rel_diff.max())} & {_sci(g.sol_inf_diff.max())} & "
                     f"{_sci(g.subopt_bound_norm.max())} & "
                     f"{statuses[0] if len(statuses)==1 else '/'.join(statuses)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def figures(raw, summ):
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    # nRMSE distributions, one panel per system, both regimes side by side.
    fig, axes = plt.subplots(1, len(P2.SYSTEMS), figsize=(18, 4.2))
    for ax, key in zip(np.atleast_1d(axes), P2.SYSTEMS):
        data, labels = [], []
        for regime in ["A_tuned_NP", "B_common_NP"]:
            for alg in P2.ALGS:
                v = raw[(raw.system == key) & (raw.regime == regime) &
                        (raw.alg == alg)].nrmse.values
                data.append(v)
                labels.append(f"{alg}\n{'A' if regime.startswith('A') else 'B'}")
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor("#cfd8dc" if i < len(P2.ALGS) else "#eceff1")
        for i, v in enumerate(data):
            ax.plot(np.full(len(v), i + 1) + np.random.uniform(-.12, .12, len(v)),
                    v, ".", ms=3, color="#37474f", alpha=.6)
        ref = summ[(summ.system == key)].reference_nrmse.iloc[0]
        ax.axhline(ref, color="k", ls="--", lw=1, label="deterministic ref.")
        ax.set_title(LABEL[key]); ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        if key == P2.SYSTEMS[0]:
            ax.set_ylabel("closed-loop nRMSE"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "primary_nrmse_box.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    # Budget check: realized evaluations per step must sit exactly on the budget.
    fig, ax = plt.subplots(figsize=(9, 4))
    x, lab = [], []
    for i, key in enumerate(P2.SYSTEMS):
        for j, alg in enumerate(P2.ALGS):
            v = raw[(raw.system == key) & (raw.alg == alg)].realized_nfe_per_step_mean.values
            x.append(v); lab.append(f"{key[:4]}\n{alg}")
    ax.boxplot(x, labels=lab, showfliers=True)
    ax.axhline(P2.FE_BUDGET, color="crimson", ls="--", lw=1.2,
               label=f"nominal budget = {P2.FE_BUDGET}")
    ax.set_ylabel("realized evaluations per MPC step")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "primary_nfe_check.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)


def compare_with_submitted(summ):
    """Descriptive comparison: submitted joint-tuned vs revised fixed-MPC."""
    sub = pd.read_csv(os.path.join(SUBMITTED, "statistical_raw_results.csv"))
    sub = sub[(sub.part == "closed_loop") & (~sub.alg.astype(str).str.startswith("QP"))]
    rows, summary = [], {}
    for key in P2.SYSTEMS:
        entry = {}
        for mode, label in [("tuned", "submitted_joint_tuned"),
                            ("equal_fe", "submitted_equal_fe")]:
            s = sub[(sub.system == key) & (sub["mode"] == mode)]
            means = {a: float(s[s.alg == a].nrmse.mean()) for a in P2.ALGS
                     if not s[s.alg == a].empty}
            if means:
                entry[label] = {"means": means,
                                "ranking": sorted(means, key=means.get)}
        for regime in ["A_tuned_NP", "B_common_NP"]:
            g = summ[(summ.system == key) & (summ.regime == regime)]
            if g.empty:
                continue
            means = {r.alg: float(r.nrmse_mean) for _, r in g.iterrows()}
            entry[f"revised_{regime}"] = {"means": means,
                                          "ranking": sorted(means, key=means.get)}
        base = entry.get("submitted_joint_tuned")
        for regime in ["A_tuned_NP", "B_common_NP"]:
            rev = entry.get(f"revised_{regime}")
            if not (base and rev):
                continue
            sgo_rank_sub = base["ranking"].index("SGO") + 1
            sgo_rank_rev = rev["ranking"].index("SGO") + 1
            rows.append(dict(
                system=key, regime=regime,
                submitted_ranking=",".join(base["ranking"]),
                revised_ranking=",".join(rev["ranking"]),
                ranking_changed=bool(base["ranking"] != rev["ranking"]),
                submitted_best=base["ranking"][0], revised_best=rev["ranking"][0],
                sgo_rank_submitted=sgo_rank_sub, sgo_rank_revised=sgo_rank_rev,
                sgo_rank_delta=sgo_rank_rev - sgo_rank_sub,
                sgo_relative=("improved" if sgo_rank_rev < sgo_rank_sub else
                              "worsened" if sgo_rank_rev > sgo_rank_sub else "unchanged"),
                sgo_nrmse_submitted=base["means"]["SGO"],
                sgo_nrmse_revised=rev["means"]["SGO"],
                spread_submitted=(max(base["means"].values()) - min(base["means"].values()))
                / max(abs(min(base["means"].values())), 1e-300),
                spread_revised=(max(rev["means"].values()) - min(rev["means"].values()))
                / max(abs(min(rev["means"].values())), 1e-300)))
        summary[key] = entry
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "submitted_vs_revised_comparison.csv"), index=False)
    with open(os.path.join(OUT, "submitted_vs_revised_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return df


def main():
    raw = pd.read_csv(os.path.join(OUT, "primary_closed_loop_raw.csv"))
    summ = pd.read_csv(os.path.join(OUT, "primary_closed_loop_summary.csv"))
    cross_path = os.path.join(OUT, "primary_reference_crosscheck.csv")
    cross = pd.read_csv(cross_path) if os.path.exists(cross_path) else pd.DataFrame()

    with open(os.path.join(OUT, "tables", "primary_nrmse.tex"), "w") as f:
        f.write(table_nrmse(summ))
    with open(os.path.join(OUT, "tables", "primary_budget_accounting.tex"), "w") as f:
        f.write(table_budget(summ))
    with open(os.path.join(OUT, "tables", "primary_timing.tex"), "w") as f:
        f.write(table_timing(summ))
    if not cross.empty:
        with open(os.path.join(OUT, "tables", "reference_crosscheck.tex"), "w") as f:
            f.write(table_crosscheck(cross))
    figures(raw, summ)
    cmp_df = compare_with_submitted(summ)
    pd.set_option("display.width", 250)
    print(cmp_df.to_string(index=False))
    print("\nwrote tables/ and figures/ under data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
