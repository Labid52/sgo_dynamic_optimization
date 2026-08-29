#!/usr/bin/env python3
"""Phase-2 closure: ANALYSIS-ONLY generation of missing manuscript artifacts.

Reads committed CSVs only. Runs no simulation, calls no optimizer, no MPC.
Fills the artifact gaps found by the Phase-2 closure audit:

    data/tables/sgo_ablation.tex
    data/tables/warm_start_mechanism.tex
    data/figures/sgo_ablation_diagnostic.eps

Nothing here changes any raw result.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ROOT), "data")
VARIANTS = ["baseline", "A_no_greedy", "B_no_bridge", "C_fitness_group", "D_no_warm_slot"]
VLAB = {"baseline": "baseline", "A_no_greedy": "A: no greedy", "B_no_bridge": "B: no bridge",
        "C_fitness_group": "C: fitness group", "D_no_warm_slot": "D: no warm slot"}
ROLES = ["plateau", "recoverable_difficult", "benign_reliable", "stratum_Moderate", "stratum_Severe"]


def table_ablation():
    d = pd.read_csv(os.path.join(OUT, "sgo_ablation_raw.csv"))
    p = d[d.budget_fe == 1000]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{SGO mechanism ablation at the primary 1000-evaluation budget: median normalized "
         r"optimality gap and strict convergence rate by instance role, plus incumbent-return rates. "
         r"Diagnostic only; no variant is proposed as an algorithm. The lower block reports the "
         r"5000-evaluation plateau instances, where fitness-based grouping breaks the flight plateau "
         r"but not the UAV one.}",
         r"\label{tab:sgo_ablation}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}l" + "c" * len(VARIANTS) + r"@{}}", r"\toprule",
         r"Role (instances) & " + " & ".join(VLAB[v] for v in VARIANTS) + r" \\", r"\midrule"]
    for role in ROLES:
        g = p[p.role == role]
        if g.empty:
            continue
        ni = g.groupby(["system", "k"]).ngroups
        cells = [f"{g[g.variant==v].norm_gap.median():.1e} / "
                 f"{100*g[g.variant==v].converged_1e4.mean():.1f}\\%" for v in VARIANTS]
        L.append(f"{role.replace('_',' ')} ({ni}) & " + " & ".join(cells) + r" \\")
    L += [r"\midrule",
          r"\multicolumn{%d}{@{}l}{\textit{incumbent-return rate, 1000 FE}} \\" % (len(VARIANTS) + 1)]
    for role in ["plateau", "recoverable_difficult"]:
        g = p[p.role == role]
        cells = [f"{100*g[g.variant==v].returned_incumbent.mean():.1f}\\%" for v in VARIANTS]
        L.append(f"{role.replace('_',' ')} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule",
          r"\multicolumn{%d}{@{}l}{\textit{plateau instances at 5000 FE: gap / convergence}} \\"
          % (len(VARIANTS) + 1)]
    pl = d[(d.plateau) & (d.budget_fe == 5000)]
    for (sy, k), g in pl.groupby(["system", "k"]):
        cells = [f"{g[g.variant==v].norm_gap.median():.1e} / "
                 f"{100*g[g.variant==v].converged_1e4.mean():.0f}\\%" for v in VARIANTS]
        L.append(f"{sy} $k{{=}}{k}$ & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "sgo_ablation.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    return d


def table_warm_start():
    w = pd.read_csv(os.path.join(OUT, "warm_start_mechanism.csv"))
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Warm-start mechanism. The start gap is the normalized suboptimality of the point "
         r"the optimizer is initialised from; the incumbent-return rate is the fraction of trials "
         r"returning that point unchanged. HVAC's reference warm start is already optimal, which is "
         r"why every optimizer trivially meets the convergence criterion there; flight's is not.}",
         r"\label{tab:warm_start_mechanism}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"System & Start condition & Median start gap & Within $10^{-4}$ & "
         r"Incumbent returned & Median final gap & Convergence \\", r"\midrule"]
    for sy in ["pendcart", "cstr", "flight", "hvac", "uav"]:
        for st in ["cold", "reference_warm", "optimizer_warm"]:
            g = w[(w.system == sy) & (w.start == st)]
            if g.empty:
                continue
            L.append(f"{sy} & {st.replace('_',' ')} & {g.start_gap_median.mean():.2e} & "
                     f"{100*g.frac_start_within_1e4.mean():.1f}\\% & "
                     f"{100*g.incumbent_return_rate.mean():.1f}\\% & "
                     f"{g.final_gap_median.mean():.2e} & {100*g.conv_rate.mean():.1f}\\% \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "warm_start_mechanism.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def figure_ablation(d):
    p = d[d.budget_fe == 1000]
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(ROLES)); wdt = 0.16
    for j, v in enumerate(VARIANTS):
        vals = [100 * p[(p.role == r) & (p.variant == v)].converged_1e4.mean() for r in ROLES]
        ax[0].bar(x + (j - 2) * wdt, vals, wdt, label=VLAB[v])
    ax[0].set_xticks(x); ax[0].set_xticklabels([r.replace("_", "\n") for r in ROLES], fontsize=7)
    ax[0].set_ylabel("convergence rate (%)"); ax[0].set_title("(a) 1000 FE, by instance role")
    ax[0].legend(fontsize=6)
    for j, v in enumerate(VARIANTS):
        vals = [100 * p[(p.role == r) & (p.variant == v)].returned_incumbent.mean()
                for r in ["plateau", "recoverable_difficult", "benign_reliable"]]
        ax[1].bar(np.arange(3) + (j - 2) * wdt, vals, wdt, label=VLAB[v])
    ax[1].set_xticks(np.arange(3))
    ax[1].set_xticklabels(["plateau", "recoverable", "benign"], fontsize=7)
    ax[1].set_ylabel("incumbent-return rate (%)"); ax[1].set_title("(b) premature incumbent return")
    pl = d[(d.plateau) & (d.budget_fe == 5000)]
    keys = sorted({(s, int(k)) for s, k in zip(pl.system, pl.k)})
    xk = np.arange(len(keys))
    for j, v in enumerate(VARIANTS):
        vals = [100 * pl[(pl.system == s) & (pl.k == k) & (pl.variant == v)].converged_1e4.mean()
                for s, k in keys]
        ax[2].bar(xk + (j - 2) * wdt, vals, wdt, label=VLAB[v])
    ax[2].set_xticks(xk); ax[2].set_xticklabels([f"{s[:4]}\nk={k}" for s, k in keys], fontsize=7)
    ax[2].set_ylabel("convergence rate (%)"); ax[2].set_title("(c) plateau instances, 5000 FE")
    for a in ax:
        a.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "sgo_ablation_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d = table_ablation()
    table_warm_start()
    figure_ablation(d)
    print("wrote tables/sgo_ablation.tex, tables/warm_start_mechanism.tex, "
          "figures/sgo_ablation_diagnostic.eps")
