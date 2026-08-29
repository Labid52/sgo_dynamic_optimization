#!/usr/bin/env python3
"""Phase 3B: ANALYSIS-ONLY regeneration of every main-paper figure and table.

Reads committed CSVs under data/ only. Runs no simulation, calls no
optimizer, solves no MPC subproblem, and writes nothing outside
data/figures_tables/. Raw data are never modified.

    python code/phase3b_manuscript_artifacts.py

Outputs
    data/figures_tables/figures/fig{1..7}_*.{pdf,eps}
    data/figures_tables/tables/tab{1..8}_*.tex
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(ROOT), "data")
FIG = os.path.join(V2, "figures_tables", "figures")
TAB = os.path.join(V2, "figures_tables", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

SYS = ["pendcart", "cstr", "flight", "hvac", "uav"]
SYSLAB = {"pendcart": "Pendulum-cart", "cstr": "CSTR", "flight": "Flight control",
          "hvac": "HVAC", "uav": "UAV"}
ALGS = ["SGO", "GWO", "PSO", "WOA"]
# One colour per optimizer, used identically in every figure of the paper.
# Palette is colour-vision-deficiency safe (Okabe-Ito); each optimizer also keeps a
# distinct marker and line style so the figures survive grayscale printing.
COL = {"SGO": "#0072B2",   # blue
       "GWO": "#D55E00",   # vermillion
       "PSO": "#009E73",   # bluish green
       "WOA": "#CC79A7"}   # reddish purple
STY = {"SGO": dict(color=COL["SGO"], marker="o", ls="-", ms=4.5),
       "GWO": dict(color=COL["GWO"], marker="s", ls="--", ms=4.2),
       "PSO": dict(color=COL["PSO"], marker="^", ls="-.", ms=4.6),
       "WOA": dict(color=COL["WOA"], marker="D", ls=":", ms=4.0)}
# One colour per system, used identically wherever systems are distinguished.
SCOL = {"pendcart": "#0072B2", "cstr": "#D55E00", "flight": "#009E73",
        "hvac": "#CC79A7", "uav": "#56585B"}
SMK = {"pendcart": "o", "cstr": "s", "flight": "^", "hvac": "D", "uav": "v"}
ACCENT = "#56585B"
STRATA = ["Benign", "Moderate", "High", "Severe"]
CRIT = 1e-4

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.labelpad": 2.5, "axes.titlepad": 3.0,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.borderpad": 0.3,
    "legend.handletextpad": 0.4, "legend.labelspacing": 0.25,
})


def rd(name):
    return pd.read_csv(os.path.join(V2, name))


def save(fig, stem):
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), format=ext, dpi=300)
    plt.close(fig)
    print(f"  figure  {stem}.pdf/.eps")


def wtab(stem, lines):
    p = os.path.join(TAB, f"{stem}.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  table   {stem}.tex")


def pct(x, d=1):
    """LaTeX-escaped percentage (tables only; matplotlib labels use a bare %)."""
    return f"{100 * x:.{d}f}\\%"


def esc(x, d=2):
    if pd.isna(x):
        return "--"
    s = f"{x:.{d}e}"
    m, e = s.split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def kap(x):
    return f"{x:.3f}" if x < 1e3 else esc(x, 2)


# --------------------------------------------------------------------------
# FIGURE 1 - the optimization environment MPC generates
# --------------------------------------------------------------------------
def fig1():
    d = rd("objective_drift_full_raw.csv")
    fig, ax = plt.subplots(1, 5, figsize=(7.3, 2.0), sharey=True, layout="constrained")
    for i, s in enumerate(SYS):
        g = d[d.system == s].sort_values("k")
        y = np.maximum(g.warm_gap_norm.values, 1e-18)
        a = ax[i]
        a.semilogy(g.k.values, y, color=SCOL[s], lw=0.6, alpha=0.9)
        nb = g[~g.benign.astype(bool)]
        a.semilogy(nb.k.values, np.maximum(nb.warm_gap_norm.values, 1e-18), ls="none",
                   marker=SMK[s], ms=2.6, mfc="none", mec=SCOL[s], mew=0.7)
        a.axhline(CRIT, color=ACCENT, ls="--", lw=0.8)
        a.set_title(f"{SYSLAB[s]}\n{100*g.benign.mean():.1f}% benign", fontsize=7.5)
        a.set_xlabel("MPC step $k$")
        a.set_ylim(1e-18, 1e3)
        a.set_yticks([1e-15, 1e-10, 1e-5, 1e0])
        a.locator_params(axis="x", nbins=4)
    ax[0].set_ylabel("warm-start suboptimality")
    ax[0].text(0.05, 0.895, r"$10^{-4}$", transform=ax[0].transAxes, fontsize=6.5,
               color=ACCENT)
    save(fig, "fig1_objective_drift")


# --------------------------------------------------------------------------
# FIGURE 2 - metric separation (rasterised dense scatter)
# --------------------------------------------------------------------------
def fig2():
    d = rd("input_error_sensitivity_raw.csv")
    ips = rd("input_error_sensitivity_summary.csv")
    ss = rd("input_error_system_sensitivity.csv").set_index("system")
    fig = plt.figure(figsize=(7.3, 4.0), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.0, 1.12], hspace=0.02)

    axt = top.subplots(1, 5, sharey=True)
    for i, s in enumerate(SYS):
        a = axt[i]
        g = d[d.system == s]
        a.scatter(np.maximum(g.gap_k, 1e-18), np.maximum(g.e_u, 1e-18), s=0.6,
                  alpha=0.28, linewidths=0, color=SCOL[s], rasterized=True)
        a.set_xscale("log"); a.set_yscale("log")
        rho = ips[ips.system == s].rho_gap_vs_eu
        a.set_title(f"{SYSLAB[s]}\n" r"$\rho=$" f"{rho.min():+.2f} to {rho.max():+.2f}",
                    fontsize=7)
        a.set_xlabel("normalized gap")
        a.set_xlim(1e-18, 1e2); a.set_ylim(1e-12, 1e2)
        a.set_xticks([1e-15, 1e-5])
        a.set_yticks([1e-10, 1e-5, 1e0])
    axt[0].set_ylabel(r"$\|u^a_k-u^\star_k\|_2$")
    top.suptitle("(a) finite-horizon error reaches the applied input "
                 "(per MPC step, all optimizers pooled)", fontsize=8, x=0.012, ha="left")

    axb = bot.subplots(1, 2)
    for j, s in enumerate(["uav", "hvac"]):
        a = axb[j]
        g = d[d.system == s].groupby(["alg", "seed"]).agg(
            eu=("e_u", "mean"), nr=("run_nrmse", "first"), ref=("reference_nrmse", "first"))
        for alg in ALGS:
            h = g.loc[alg]
            a.plot(h.eu, h.nr, ls="none", marker=STY[alg]["marker"], ms=4.4,
                   mfc="none", mec=STY[alg]["color"], mew=1.2, label=alg)
        a.axhline(g.ref.iloc[0], color=ACCENT, ls="--", lw=0.8)
        a.set_yscale("log")
        a.set_xlabel("run-mean first-input error")
        a.set_ylabel("closed-loop nRMSE")
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        r = ss.loc[s]
        a.set_title(f"{SYSLAB[s]}: " r"$\rho=$" f"{r.rho_meaneu_vs_nrmse:+.2f}"
                    f" ($p={r.p:.0e}$)", fontsize=7.5)
        a.legend(loc="best", ncol=2, fontsize=6.5)
    bot.suptitle("(b) its relation to closed-loop tracking is system-specific and can invert",
                 fontsize=8, x=0.012, ha="left")
    save(fig, "fig2_metric_separation")


# --------------------------------------------------------------------------
# FIGURE 3 - fair fixed-MPC comparison and effect sizes
# --------------------------------------------------------------------------
def fig3():
    raw = rd("primary_closed_loop_raw.csv")
    raw = raw[raw.regime == "B_common_NP"].copy()
    raw["excess"] = 100 * (raw.nrmse - raw.deterministic_reference_nrmse) \
        / raw.deterministic_reference_nrmse
    es = rd("effect_sizes_primary.csv")
    es = es[(es.alg_a == "SGO") & (es.regime == "B_common_NP")]

    ABBR = {"pendcart": "P-cart", "cstr": "CSTR", "flight": "Flight",
            "hvac": "HVAC", "uav": "UAV"}
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.8), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    pos, ticks, labs = [], [], []
    for i, s in enumerate(SYS):
        for j, alg in enumerate(ALGS):
            v = raw[(raw.system == s) & (raw.alg == alg)].excess.values
            p = i * 5 + j
            bp = a.boxplot([v], positions=[p], widths=0.68, showfliers=False,
                           patch_artist=True, medianprops=dict(color="k", lw=1.0))
            bp["boxes"][0].set(facecolor=STY[alg]["color"], alpha=0.55, lw=0.6,
                               edgecolor=STY[alg]["color"])
            pos.append(p)
        ticks.append(i * 5 + 1.5)
        labs.append(SYSLAB[s])
    a.set_xticks(ticks); a.set_xticklabels(labs, fontsize=7)
    a.set_xticks(pos, minor=True)
    a.set_xticklabels(ALGS * 5, minor=True, fontsize=5.0, rotation=90)
    a.tick_params(axis="x", which="minor", pad=1)
    a.tick_params(axis="x", which="major", pad=14)
    a.axhline(0, color=ACCENT, ls="--", lw=0.8)
    a.set_yscale("symlog", linthresh=0.01)
    a.set_ylabel("closed-loop nRMSE excess over\ndeterministic reference (%)")
    a.set_title("(a) fixed MPC, exact 1000-evaluation budget", loc="left")

    a = ax[1]
    rows = []
    for s in SYS:
        for o in ["GWO", "PSO", "WOA"]:
            r = es[(es.system == s) & (es.alg_b == o)]
            if len(r):
                rows.append((f"{ABBR[s]} vs {o}", r.cliffs_delta.iloc[0],
                             r.cliffs_delta_lo.iloc[0], r.cliffs_delta_hi.iloc[0],
                             r.evidence.iloc[0]))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, dlt, lo, hi, ev) in zip(y, rows):
        clear = ev == "clear separation"
        c = COL["GWO"] if clear else COL["SGO"]
        a.plot([lo, hi], [yy, yy], color=c, lw=1.3 if clear else 0.9,
               alpha=1.0 if clear else 0.7)
        a.plot(dlt, yy, marker="o", ms=4.4 if clear else 3.4,
               mfc=c if clear else "white", mec=c, mew=1.0)
    a.axvline(0, color=ACCENT, ls="--", lw=0.8)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    a.set_xlabel(r"Cliff's $\delta$ (SGO $-$ other), 95% CI")
    a.set_xlim(-1.12, 1.12)
    a.set_title("(b) filled marker: clear separation", loc="left")
    save(fig, "fig3_fair_comparison")


# --------------------------------------------------------------------------
# FIGURE 4 - drift severity vs finite-horizon difficulty
# --------------------------------------------------------------------------
def fig4():
    r = rd("frozen_sweep_v2_raw.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.7), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1, 1.05]))

    a = ax[0]
    w = 0.2
    for j, alg in enumerate(ALGS):
        vals = [100 * r[(r.start == "cold") & (r.alg == alg)
                        & (r.stratum == st)].converged_1e4.mean() for st in STRATA]
        a.bar(np.arange(4) + (j - 1.5) * w, vals, width=w, label=alg,
              color=STY[alg]["color"], edgecolor="white", lw=0.5)
    a.set_xticks(range(4)); a.set_xticklabels(STRATA)
    a.set_xlabel("pre-registered drift stratum")
    a.set_ylabel(r"convergence at $10^{-4}$ (%)")
    a.set_ylim(0, 108)
    a.legend(ncol=4, loc="upper right", columnspacing=0.8, handletextpad=0.3)
    a.set_title("(a) cold start, 1000 evaluations", loc="left")

    a = ax[1]
    m = (r[r.start == "cold"].groupby(["system", "k", "alg"])
         .agg(gap=("final_gap_norm", "median"), drift=("warm_gap_norm", "first"))
         .reset_index())
    ms = m[m.alg == "SGO"]
    ass_ = rd("drift_performance_association.csv")
    ass_ = ass_[(ass_.alg == "SGO") & (ass_.start == "cold")].set_index("system")
    for s in SYS:
        g = ms[ms.system == s]
        a.plot(np.maximum(g.drift, 1e-18), np.maximum(g.gap, 1e-18), ls="none",
               marker=SMK[s], ms=3.4, mfc="none", mec=SCOL[s], mew=0.9,
               label=f"{SYSLAB[s]} ($\\rho={ass_.loc[s].spearman_drift_vs_gap:+.2f}$)")
    a.axhline(CRIT, color=ACCENT, ls="--", lw=0.8)
    a.axvline(CRIT, color=ACCENT, ls=":", lw=0.8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("measured drift of the instance")
    a.set_ylabel("SGO median normalized gap")
    a.legend(loc="upper left", fontsize=6.0)
    a.set_title("(b) per-instance, SGO", loc="left")
    save(fig, "fig4_drift_difficulty")


# --------------------------------------------------------------------------
# FIGURE 5 - budget response
# --------------------------------------------------------------------------
def fig5():
    b = rd("budget_sensitivity_v2_summary.csv")
    inst = b[["system", "k"]].drop_duplicates().values.tolist()
    n = len(inst)
    ABBR = {"pendcart": "P-cart", "cstr": "CSTR", "flight": "Flight",
            "hvac": "HVAC", "uav": "UAV"}
    SAB = {"Benign": "Ben", "Moderate": "Mod", "High": "High", "Severe": "Sev"}
    nc = (n + 1) // 2
    fig, ax = plt.subplots(2, nc, figsize=(7.3, 3.5), sharey=True, sharex=True,
                           layout="constrained")
    ax = ax.ravel()
    for i, (s, k) in enumerate(inst):
        a = ax[i]
        g = b[(b.system == s) & (b.k == k)]
        for alg in ALGS:
            h = g[g.alg == alg].sort_values("budget_fe")
            a.plot(h.budget_fe, np.maximum(h.median_gap, 1e-18), label=alg, **STY[alg])
        a.axhline(CRIT, color=ACCENT, ls="--", lw=0.8)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_title(f"{ABBR[s]} $k={k}$, {SAB[g.stratum.iloc[0]]}", fontsize=6.8, pad=2.5)
        a.set_xticks([250, 1000, 5000]); a.set_xticklabels(["250", "1k", "5k"])
        a.set_xlim(200, 6500)
        a.set_ylim(1e-17, 1e1)
        a.set_yticks([1e-15, 1e-10, 1e-5, 1e0])
    for i in range(n, len(ax)):
        ax[i].axis("off")
        # a column whose lower panel is empty must keep its own x tick labels
        ax[i - nc].tick_params(labelbottom=True)
    ax[0].set_ylabel("median normalized gap")
    ax[nc].set_ylabel("median normalized gap")
    ax[0].legend(ncol=2, fontsize=6, loc="lower left", columnspacing=0.7)
    fig.supxlabel("exact objective evaluations per solve", fontsize=8)
    save(fig, "fig5_budget")


# --------------------------------------------------------------------------
# FIGURE 6 - decision dimension and SGO mechanism
# --------------------------------------------------------------------------
def fig6():
    d = rd("dimension_sweep_summary.csv")
    ab = rd("sgo_ablation_summary.csv")
    assoc = rd("dimension_sweep_association.csv").set_index("alg")
    fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.7), layout="constrained")

    a = ax[0]
    for alg in ALGS:
        g = d[d.alg == alg].sort_values("dim")
        a.plot(g.dim, 100 * g.conv_rate, ls="none", marker=STY[alg]["marker"],
               ms=4.2, mfc="none", mec=STY[alg]["color"], mew=0.9,
               label=f"{alg} ($\\rho={assoc.loc[alg].rho_dim_vs_conv:+.2f}$)")
    a.set_xlabel(r"decision dimension $d=N_c n_u$")
    a.set_ylabel(r"convergence at $10^{-4}$ (%)")
    a.set_ylim(-6, 110)
    a.legend(ncol=1, columnspacing=0.6, fontsize=6, loc="upper right")
    a.set_title("(a) 22 certified cells", loc="left", fontsize=7.5)

    a = ax[1]
    for s in SYS:
        g = d[(d.system == s) & (d.alg == "SGO")].sort_values("dim")
        if g.empty:
            continue
        a.plot(g.cond_H, 100 * g.conv_rate, ls="none", marker=SMK[s], ms=4.4,
               mfc="none", mec=SCOL[s], mew=1.0, label=SYSLAB[s])
    a.set_xscale("log")
    a.set_xlabel(r"$\kappa(H)$ of the same cell")
    a.set_ylabel(r"SGO convergence (%)")
    a.set_ylim(-6, 110)
    a.legend(fontsize=6, loc="upper right")
    a.set_title(r"(b) SGO: $\rho=-0.25$, $p=0.27$", loc="left", fontsize=7.5)

    a = ax[2]
    pl = ab[(ab.plateau) & (ab.budget_fe == 5000)]
    order = ["baseline", "C_fitness_group", "D_no_warm_slot", "B_no_bridge", "A_no_greedy"]
    lab = {"baseline": "baseline", "C_fitness_group": "fitness group",
           "D_no_warm_slot": "no warm slot", "B_no_bridge": "no bridge",
           "A_no_greedy": "no greedy"}
    keys = [("flight", 6), ("flight", 7), ("uav", 0)]
    KL = {("flight", 6): "Flight $k{=}6$", ("flight", 7): "Flight $k{=}7$",
          ("uav", 0): "UAV $k{=}0$"}
    w = 0.26
    for j, (s, k) in enumerate(keys):
        vals = [100 * pl[(pl.system == s) & (pl.k == k)
                         & (pl.variant == v)].conv_rate.iloc[0] for v in order]
        a.bar(np.arange(len(order)) + (j - 1) * w, vals, width=w,
              label=KL[(s, k)], edgecolor="white", lw=0.5,
              color=[COL["PSO"], COL["SGO"], COL["GWO"]][j])
    a.set_xticks(range(len(order)))
    a.set_xticklabels([lab[v] for v in order], fontsize=6, rotation=22, ha="right")
    a.set_ylabel(r"convergence at 5000 FE (%)")
    a.set_ylim(0, 112)
    a.legend(fontsize=6, loc="upper right")
    a.set_title("(c) SGO ablation, plateau instances", loc="left", fontsize=7.5)
    save(fig, "fig6_dimension_mechanism")


# --------------------------------------------------------------------------
# FIGURE 7 - prediction-tail sensitivity
# --------------------------------------------------------------------------
def fig7():
    det = rd("tail_deterministic_summary.csv").set_index("system")
    pe = rd("tail_paired_effects.csv")
    oc = rd("tail_ordering_comparison.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.5), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1, 1, 1.05]))

    a = ax[0]
    sgn = [100 * (det.loc[s].nrmse_hold - det.loc[s].nrmse_zero)
           / det.loc[s].nrmse_zero for s in SYS]
    a.barh(range(5), sgn, color=[SCOL[x] for x in SYS], edgecolor="white", lw=0.6)
    a.set_yticks(range(5)); a.set_yticklabels([SYSLAB[s] for s in SYS], fontsize=6.5)
    a.set_xscale("symlog", linthresh=1)
    a.axvline(0, color=ACCENT, lw=0.8)
    a.set_xlabel("change in deterministic\nclosed-loop nRMSE (%)")
    a.set_title("(a) hold-last vs zero-tail", loc="left")

    a = ax[1]
    rows = [(s, alg) for s in ["flight", "uav"] for alg in ALGS]
    y = np.arange(len(rows))[::-1]
    for yy, (s, alg) in zip(y, rows):
        r = pe[(pe.system == s) & (pe.alg == alg)].iloc[0]
        a.plot([r.ci95_lo, r.ci95_hi], [yy, yy], color=STY[alg]["color"], lw=1.1)
        a.plot(r.d_nrmse_rel_pct, yy, marker=STY[alg]["marker"], ms=4.2,
               mfc=STY[alg]["color"], mec=STY[alg]["color"])
    a.axvline(0, color=ACCENT, ls="--", lw=0.8)
    a.set_yticks(y)
    a.set_yticklabels([f"{SYSLAB[s][:6]}. {alg}" for s, alg in rows], fontsize=6)
    a.set_xlabel("paired change in\nclosed-loop nRMSE (%)")
    a.set_title("(b) same seeds, no retuning", loc="left")

    a = ax[2]
    FAMSTY = {"optimization_accuracy_mean_gap": ("--", "s", "finite-horizon accuracy"),
              "closed_loop_nrmse": ("-", "o", "closed-loop nRMSE")}
    seen = set()
    for gi, s in enumerate(["flight", "uav"]):
        x0 = gi * 2.4
        for f, (ls, mk, fl) in FAMSTY.items():
            r = oc[(oc.system == s) & (oc.metric_family == f)].iloc[0]
            lab = fl if fl not in seen else None
            seen.add(fl)
            c = COL["SGO"] if f == "closed_loop_nrmse" else COL["GWO"]
            a.plot([x0, x0 + 1], [r.sgo_rank_zero, r.sgo_rank_hold_last], ls=ls,
                   marker=mk, color=c, ms=4.6, lw=1.3,
                   mfc=c if not r.sgo_rank_unchanged else "white", label=lab)
        a.text(x0 + 0.5, 0.42, SYSLAB[s], ha="center", fontsize=7)
    a.set_xticks([0, 1, 2.4, 3.4])
    a.set_xticklabels(["zero", "hold", "zero", "hold"], fontsize=6.5)
    a.set_xlim(-0.35, 3.75)
    a.set_ylim(4.85, 0.25)
    a.legend(fontsize=6.0, loc="lower right", framealpha=1.0)
    a.set_yticks([1, 2, 3, 4])
    a.set_ylabel("SGO rank of 4")
    a.set_title("(c) filled marker: rank changed", loc="left")
    save(fig, "fig7_tail_convention")


# --------------------------------------------------------------------------
# TABLES
# --------------------------------------------------------------------------
def tab2_drift():
    s = rd("objective_drift_full_summary.csv").set_index("system")
    c = rd("objective_drift_full_concentration.csv").set_index("system")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Measured optimization-relevant drift over \emph{every} consecutive transition "
         r"of the deterministic closed-loop trajectory (fixed MPC, $N_c=1$). The drift measure is "
         r"the shifted-reference warm-start normalized suboptimality of Eq.~\eqref{eq:drift}; a "
         r"transition is \emph{benign} when it falls below the $10^{-4}$ criterion at which this "
         r"study declares convergence. \emph{In first 10\%} is the share of non-benign transitions "
         r"falling inside the first tenth of the trajectory. The last column reports the "
         r"reference-driven component of $\Delta f_k$, computed from the implemented equations. "
         r"All remaining drift is state-driven.}",
         r"\label{tab:drift}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r"System & Trans. & Benign & Median & p90 & Max & In first 10\% & "
         r"$\kappa(H)$ & Ref.-driven \\", r"\midrule"]
    for sy in SYS:
        r, k = s.loc[sy], c.loc[sy]
        conc = "--" if pd.isna(k.severe_in_first_10pct_of_trajectory) else \
            pct(k.severe_in_first_10pct_of_trajectory, 0)
        ref = "none (exactly 0)" if r.ref_contrib_max == 0 else esc(r.ref_contrib_median, 2)
        L.append(f"{SYSLAB[sy]} & {int(r.transitions)} & {pct(r.benign_fraction)} & "
                 f"{esc(r.warm_gap_median)} & {esc(r.warm_gap_p90)} & {esc(r.warm_gap_max)} & "
                 f"{conc} & {kap(r.cond_H)} & {ref} \\\\")
    tot = int(s.transitions.sum())
    ben = float((s.benign_fraction * s.transitions).sum() / s.transitions.sum())
    L += [r"\midrule",
          f"\\textbf{{All}} & {tot} & \\textbf{{{pct(ben)}}} & \\multicolumn{{6}}{{l}}"
          r"{\emph{per-system benign fraction 65.3\%--100\%}} \\",
          r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tab2_drift", L)


def tab3_primary():
    s = rd("primary_closed_loop_summary.csv")
    cmp_ = rd("submitted_vs_revised_comparison.csv").set_index(["system", "regime"])
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Fixed-MPC primary comparison. The MPC formulation, the initial condition and "
         r"the deterministic reference are identical for all four optimizers; every run consumes "
         r"exactly 1000 objective calls per MPC step (verified per run). Values are mean "
         r"$\pm$ standard deviation of the closed-loop nRMSE over 20 runs. The last two columns "
         r"report how the ranking and the between-optimizer spread compare with the jointly tuned, "
         r"nominally-equal-budget protocol of the co-design analysis (Section~\ref{subsec:codesign}).}",
         r"\label{tab:primary}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
         r"Regime & System & SGO & GWO & PSO & WOA & Ref. & Rank & "
         r"Spread \\", r"\midrule"]
    for reg, rlab in [("A_tuned_NP", r"A ($NP$ per optimizer)"),
                      ("B_common_NP", r"B ($NP{=}20$ common)")]:
        for i, sy in enumerate(SYS):
            g = s[(s.system == sy) & (s.regime == reg)].set_index("alg")
            best = g.nrmse_mean.idxmin()
            cells = []
            for alg in ALGS:
                r = g.loc[alg]
                t = f"{r.nrmse_mean:.5g}\\,$\\pm${r.nrmse_std:.0e}"
                cells.append(r"\textbf{" + t + "}" if alg == best else t)
            c = cmp_.loc[(sy, reg)]
            ratio = c.spread_submitted / c.spread_revised
            L.append(f"{rlab if i == 0 else ''} & {SYSLAB[sy]} & " + " & ".join(cells) +
                     f" & {g.reference_nrmse.iloc[0]:.6g} & "
                     f"{'yes' if c.ranking_changed else 'no'} & {ratio:.1f}$\\times$ \\\\")
        if reg == "A_tuned_NP":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tab3_primary", L)


def tab4_effects():
    e = rd("effect_sizes_primary.csv")
    e = e[e.alg_a == "SGO"]
    EV = {"clear separation": "clear separation",
          "numerically ordered but statistically uncertain": "uncertain",
          "practically negligible under descriptive thresholds": "negligible"}
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Effect sizes for SGO against each reference optimizer under the fixed-MPC "
         r"protocol. A negative $\Delta$mean means SGO attains the lower closed-loop nRMSE. "
         r"Cliff's $\delta$ is oriented SGO$-$other with a bootstrap 95\% CI (10\,000 resamples); "
         r"$p$ is Holm--Bonferroni adjusted within the six pairwise comparisons of each "
         r"(system, regime) family. \emph{Uncertain} means the $\delta$ interval contains zero; "
         r"\emph{negligible} means the difference is resolvable but below the smallest descriptive "
         r"practical threshold (0.5\%). Across all 60 pairwise comparisons only 9 show clear "
         r"separation and all 9 are UAV.}",
         r"\label{tab:effects}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}llcrrcrl@{}}", r"\toprule",
         r"System & Reg. & vs & $\Delta$mean & rel.\ \% & Cliff's $\delta$ [95\% CI] & "
         r"$p_{\mathrm{Holm}}$ & Evidence \\", r"\midrule"]
    for sy in SYS:
        for reg, rl in [("A_tuned_NP", "A"), ("B_common_NP", "B")]:
            for o in ["GWO", "PSO", "WOA"]:
                r = e[(e.system == sy) & (e.regime == reg) & (e.alg_b == o)]
                if not len(r):
                    continue
                r = r.iloc[0]
                dm = r.mean_a - r.mean_b
                L.append(f"{SYSLAB[sy]} & {rl} & {o} & ${dm:+.2e}$ & "
                         f"{r.relative_diff_pct:.3f} & "
                         f"${r.cliffs_delta:+.2f}$ [{r.cliffs_delta_lo:+.2f}, "
                         f"{r.cliffs_delta_hi:+.2f}] & {r.p_holm:.3g} & {EV[r.evidence]} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tab4_effects", L)


def tab5_strata():
    r = rd("frozen_sweep_v2_raw.csv")
    c = r[r.start == "cold"]
    ninst = c[["system", "k", "stratum"]].drop_duplicates().groupby("stratum").size()
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Finite-horizon accuracy by pre-registered drift stratum (cold start, 215 "
         r"instances, 20 trials each, exact 1000-evaluation budget). Strata were fixed from the "
         r"drift measure alone before any optimizer trial and were not redrawn. Two imbalances are "
         r"reported rather than removed: HVAC contributes no non-benign transition at all, and the "
         r"Severe stratum is UAV-dominated (16 of 19 transitions; 8 of the 11 sampled instances). "
         r"The within-system continuous associations reported in "
         r"Section~\ref{subsec:drift_difficulty} do not "
         r"depend on stratum balance.}",
         r"\label{tab:strata}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}lrcccc@{}}", r"\toprule",
         r"Stratum & Inst. & Optimizer & Median gap & IQR & Convergence \\", r"\midrule"]
    for st in STRATA:
        for i, alg in enumerate(ALGS):
            g = c[(c.stratum == st) & (c.alg == alg)]
            q1, q3 = g.final_gap_norm.quantile([0.25, 0.75])
            L.append(f"{st if i == 0 else ''} & {int(ninst[st]) if i == 0 else ''} & {alg} & "
                     f"{g.final_gap_norm.median():.3e} & [{q1:.2e}, {q3:.2e}] & "
                     f"{pct(g.converged_1e4.mean())} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tab5_strata", L)


def tab6_warm():
    w = rd("warm_start_mechanism.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Warm-start semantics on the three systems where they matter. \emph{Cold} starts "
         r"from the zero sequence. \emph{Reference warm} shifts the deterministic reference solution "
         r"of the previous step, which is not available to a deployed controller. \emph{Optimizer "
         r"warm} shifts the optimizer's own previous solution, which is the deployed condition. The "
         r"start gap is the normalized suboptimality of the point the optimizer is initialised from "
         r"and is reported for SGO. The lower block gives the fraction of SGO trials that return the "
         r"start point unchanged. Per-optimizer detail for all five systems is in the supplement.}",
         r"\label{tab:warm}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"& & \multicolumn{2}{c}{start point (SGO)} & "
         r"\multicolumn{4}{c}{convergence at $10^{-4}$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}",
         r"System & Start condition & median gap & within $10^{-4}$ & SGO & GWO & PSO & WOA \\",
         r"\midrule"]
    for sy in ["hvac", "flight", "uav"]:
        for i, (st, sl) in enumerate([("cold", "cold"),
                                      ("reference_warm", "reference warm"),
                                      ("optimizer_warm", "optimizer warm")]):
            g = w[(w.system == sy) & (w.start == st)].set_index("alg")
            r0 = g.loc["SGO"]
            L.append(f"{SYSLAB[sy] if i == 0 else ''} & {sl} & {r0.start_gap_median:.2e} & "
                     f"{pct(r0.frac_start_within_1e4)} & "
                     + " & ".join(pct(g.loc[al].conv_rate) for al in ALGS) + r" \\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\midrule",
                  r"\multicolumn{8}{@{}l}{\textit{SGO trials returning the start point unchanged}} \\"]
    for sy in ["hvac", "flight", "uav"]:
        g = w[(w.system == sy) & (w.alg == "SGO")].set_index("start")
        L.append(f"{SYSLAB[sy]} & & \\multicolumn{{6}}{{l}}{{cold "
                 f"{pct(g.loc['cold'].incumbent_return_rate)}, reference warm "
                 f"{pct(g.loc['reference_warm'].incumbent_return_rate)}, optimizer warm "
                 f"{pct(g.loc['optimizer_warm'].incumbent_return_rate)}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tab6_warm", L)


def tab7_dim_abl():
    d = rd("dimension_sweep_summary.csv")
    a = rd("dimension_sweep_association.csv").set_index("alg")
    ab = rd("sgo_ablation_summary.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Decision dimension and SGO mechanism. \emph{Upper block}: rank associations "
         r"across the 22 accepted (system, $N_c$) cells spanning $d=1$ to $24$, at the exact "
         r"1000-evaluation budget, with $p$ values for the conditioning association. The UAV at "
         r"$N_c\ge3$ is excluded because the two reference solvers do not agree there. Per-cell "
         r"convergence rates appear in Figure~\ref{fig:mechanism}(a) and, in full, in the "
         r"supplement. \emph{Lower block}: SGO ablation on the three plateau instances at 5000 "
         r"evaluations, reported as median gap and convergence rate.}",
         r"\label{tab:dim_abl}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"\multicolumn{7}{@{}l}{\textit{(a) rank association over the 22 accepted cells}} \\",
         r"Quantity & & SGO & GWO & PSO & WOA & \\", r"\midrule"]
    L.append(r"$\rho(d,\ \mathrm{convergence})$ & & "
             + " & ".join(f"${a.loc[al].rho_dim_vs_conv:+.2f}$" for al in ALGS) + r" & \\")
    L.append(r"$\rho(d,\ \mathrm{median\ gap})$ & & "
             + " & ".join(f"${a.loc[al].rho_dim_vs_gap:+.2f}$" for al in ALGS) + r" & \\")
    L.append(r"$\rho(\kappa(H),\ \mathrm{median\ gap})$ & & "
             + " & ".join(f"${a.loc[al].rho_cond_vs_gap:+.2f}$ ($p={a.loc[al].p_cond_vs_gap:.2f}$)"
                          for al in ALGS) + r" & \\")
    hv = d[(d.system == "hvac") & (d.alg == "SGO")].sort_values("dim")
    hg = d[(d.system == "hvac") & (d.alg == "GWO")].sort_values("dim")
    L += [r"\midrule",
          r"\multicolumn{7}{@{}l}{\textit{HVAC control case: }"
          r"$\kappa(H)$\textit{ stays near 1 while convergence falls}} \\",
          r"$d$ & & " + " & ".join(str(int(x)) for x in hv.dim.values) + r" \\",
          r"$\kappa(H)$ & & " + " & ".join(f"{x:.3f}" for x in hv.cond_H.values) + r" \\",
          r"SGO convergence & & " + " & ".join(pct(x, 0) for x in hv.conv_rate.values) + r" \\",
          r"GWO convergence & & " + " & ".join(pct(x, 0) for x in hg.conv_rate.values) + r" \\"]
    pl = ab[(ab.plateau) & (ab.budget_fe == 5000)]
    order = ["baseline", "C_fitness_group", "D_no_warm_slot", "B_no_bridge", "A_no_greedy"]
    OL = {"baseline": "baseline", "C_fitness_group": "fitness grouping",
          "D_no_warm_slot": "no warm slot", "B_no_bridge": "no bridge",
          "A_no_greedy": "no greedy"}
    L += [r"\midrule",
          r"\multicolumn{7}{@{}l}{\textit{(b) SGO ablation at 5000 evaluations, plateau "
          r"instances: median gap / convergence}} \\",
          r"Instance & " + " & ".join(OL[v] for v in order) + r" & \\"]
    for sy, k in [("flight", 6), ("flight", 7), ("uav", 0)]:
        cells = []
        for v in order:
            r = pl[(pl.system == sy) & (pl.k == k) & (pl.variant == v)].iloc[0]
            cells.append(f"{r.median_gap:.1e} / {pct(r.conv_rate, 0)}")
        L.append(f"{SYSLAB[sy]} $k={k}$ & " + " & ".join(cells) + r" & \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tab7_dim_abl", L)


def tab8_sens():
    p = rd("perturbation_summary.csv")
    t = rd("tail_deterministic_summary.csv").set_index("system")
    tp = rd("tail_paired_effects.csv")
    oc = rd("tail_ordering_comparison.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Sensitivity to modelling choices. \emph{(a)} Bounded perturbations of the "
         r"nominal sequences: measured non-benign fraction and each optimizer's closed-loop nRMSE "
         r"excess over the deterministic reference \emph{of the same perturbed problem} "
         r"(10 runs, exact budget). \emph{(b)} Replacing the zero-tail prediction convention with "
         r"hold-last, nothing retuned: deterministic closed-loop change and the median first "
         r"applied-control change as a fraction of the input range. \emph{(c)} Paired-seed "
         r"stochastic comparison on flight and UAV: SGO's rank of four under each convention, "
         r"separately by metric family.}",
         r"\label{tab:sens}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"\multicolumn{7}{@{}l}{\textit{(a) bounded perturbation: nRMSE excess over the "
         r"deterministic reference}} \\",
         r"System & Condition & Non-benign & SGO & GWO & PSO & WOA \\", r"\midrule"]
    for sy in ["hvac", "uav"]:
        for cond, cl in [("nominal", "nominal"), ("process_disturbance", "process disturbance"),
                         ("measurement_noise", "measurement noise"),
                         ("reference_event", "abrupt reference")]:
            g = p[(p.system == sy) & (p.condition == cond)].set_index("alg")
            L.append(f"{SYSLAB[sy]} & {cl} & {pct(g.frac_non_benign.iloc[0])} & "
                     + " & ".join(f"${100*g.loc[al].relative_excess_vs_reference:+.2f}\\%$"
                                  for al in ALGS) + r" \\")
        L.append(r"\addlinespace")
    L = L[:-1]
    L += [r"\midrule",
          r"\multicolumn{7}{@{}l}{\textit{(b) prediction tail: hold-last vs zero-tail, "
          r"deterministic, no retuning}} \\",
          r"System & & nRMSE zero-tail & nRMSE hold-last & Change & "
          r"Median $|\Delta u_0|$ / range & \\"]
    tr = rd("tail_reference_raw.csv")
    med = tr.groupby("system").first_control_diff_rel_range.median()
    for sy in SYS:
        r = t.loc[sy]
        ch = 100 * (r.nrmse_hold - r.nrmse_zero) / r.nrmse_zero
        m = 100 * med[sy]
        if m >= 1:
            mf = f"${m:.1f}\\%$"
        elif m >= 0.01:
            mf = f"${m:.2f}\\%$"
        elif m >= 0.001:
            mf = f"${m:.3f}\\%$"
        else:
            mf = r"$<0.001\%$"
        L.append(f"{SYSLAB[sy]} & & {r.nrmse_zero:.6g} & {r.nrmse_hold:.6g} & "
                 f"${ch:+.1f}\\%$ & {mf} & \\\\")
    L += [r"\midrule",
          r"\multicolumn{7}{@{}l}{\textit{(c) prediction tail: SGO rank of four under each "
          r"convention (paired seeds, exact budget)}} \\",
          r"System & Metric family & zero-tail & hold-last & Paired $\Delta$nRMSE & 95\% CI & \\"]
    FAM = {"optimization_accuracy_mean_gap": "finite-horizon accuracy",
           "closed_loop_nrmse": "closed-loop nRMSE"}
    for sy in ["flight", "uav"]:
        pr = tp[(tp.system == sy) & (tp.alg == "SGO")].iloc[0]
        for i, f in enumerate(FAM):
            r = oc[(oc.system == sy) & (oc.metric_family == f)].iloc[0]
            extra = (f"${pr.d_nrmse_rel_pct:+.1f}\\%$ & $[{pr.ci95_lo:+.1f}, {pr.ci95_hi:+.1f}]$"
                     if f == "closed_loop_nrmse" else " & ")
            L.append(f"{SYSLAB[sy] if i == 0 else ''} & {FAM[f]} & {int(r.sgo_rank_zero)} & "
                     f"{int(r.sgo_rank_hold_last)} & {extra} & \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tab8_sens", L)


def supp_tables():
    """Compact supplement-role tables the main text cites in one line."""
    m = rd("multi_ic_summary.csv")
    st = rd("multi_ic_ranking_stability.csv")
    st = st[st.ic_a == "ALL"].set_index("system")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Multi-initial-condition robustness (5 systems $\times$ 5 initial conditions "
         r"$\times$ 4 optimizers $\times$ 5 seeds $=$ 500 runs, fixed MPC, exact budget). The "
         r"initial conditions were frozen before running and exclude those used for optimizer "
         r"tuning. $\bar\tau$ is the mean pairwise Kendall correlation between the per-IC "
         r"optimizer rankings. The last column counts the (system, IC) cells in which the best "
         r"optimizer is separated from the runner-up at $\alpha=0.05$.}",
         r"\label{tab:multi_ic}", r"\begin{tabular}{@{}lccl@{}}", r"\toprule",
         r"System & $\bar\tau$ across ICs & Distinct winners & Winner by IC \\", r"\midrule"]
    for sy in SYS:
        r = st.loc[sy]
        wins = json.loads(r.best_by_ic)
        L.append(f"{SYSLAB[sy]} & ${r.mean_pairwise_tau:+.2f}$ & {int(r.n_distinct_best)} & "
                 + ", ".join(sorted(set(wins.values()))) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    wtab("tabS_multi_ic", L)

    c = rd("reference_nc_sweep_summary.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Deterministic-reference check. OSQP is the primary solver and multistart "
         r"L-BFGS-B is an independent cross-check. A cell is accepted when the OSQP solution carries "
         r"a strong-convexity suboptimality bound far below the $10^{-4}$ convergence criterion and "
         r"the two solvers agree to better than $10^{-4}$ relative. Twenty-two of 25 cells are "
         r"accepted. The three UAV cells at $N_c\ge3$ are not, and are excluded from every "
         r"optimality-gap computation in this study.}",
         r"\label{tab:refcert}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}lrrcccc@{}}", r"\toprule",
         r"System & $N_c$ & $d$ & Suboptimality bound & Solver disagreement & "
         r"$\kappa(H)$ & Accepted \\", r"\midrule"]
    NO = r"\textbf{no}"
    for sy in SYS:
        for _, r in c[c.system == sy].sort_values("Nc").iterrows():
            ok = "yes" if r.certified else NO
            L.append(f"{SYSLAB[sy]} & {int(r.Nc)} & {int(r.dim)} & "
                     f"{r.max_subopt_bound_norm:.1e} & {r.max_solver_rel_disagreement:.1e} & "
                     f"{r.cond_H:.3g} & {ok} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tabS_refcert", L)

    d = rd("dimension_sweep_summary.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Strict convergence rate at the $10^{-4}$ criterion by decision dimension "
         r"$d=N_c n_u$ on accepted frozen instances at the exact 1000-evaluation budget, 20 trials "
         r"per cell. The UAV at $N_c\ge3$ is excluded because the two reference solvers do not agree "
         r"there.}",
         r"\label{tab:dimgrid}", r"\begin{tabular}{@{}lrrcccc@{}}", r"\toprule",
         r"System & $N_c$ & $d$ & SGO & GWO & PSO & WOA \\", r"\midrule"]
    for sy in SYS:
        g = d[d.system == sy]
        for j, dd in enumerate(sorted(g.dim.unique())):
            h = g[g.dim == dd].set_index("alg")
            L.append(f"{SYSLAB[sy] if j == 0 else ''} & {int(h.Nc.iloc[0])} & {int(dd)} & "
                     + " & ".join(pct(h.loc[al].conv_rate, 0) for al in ALGS) + r" \\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    wtab("tabS_dimgrid", L)

    w = rd("warm_start_mechanism.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Warm-start mechanism for all five systems and all four optimizers. The start gap "
         r"is the normalized suboptimality of the point the optimizer is initialised from, and the "
         r"incumbent-return rate is the fraction of trials that return that point unchanged.}",
         r"\label{tab:warmfull}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"System & Start condition & Optimizer & Median start gap & Within $10^{-4}$ & "
         r"Incumbent returned & Convergence \\", r"\midrule"]
    for sy in SYS:
        first_s = True
        for st, sl in [("cold", "cold"), ("reference_warm", "reference warm"),
                       ("optimizer_warm", "optimizer warm")]:
            for i, alg in enumerate(ALGS):
                r = w[(w.system == sy) & (w.start == st) & (w.alg == alg)].iloc[0]
                L.append(f"{SYSLAB[sy] if first_s else ''} & {sl if i == 0 else ''} & {alg} & "
                         f"{r.start_gap_median:.2e} & {pct(r.frac_start_within_1e4)} & "
                         f"{pct(r.incumbent_return_rate)} & {pct(r.conv_rate)} \\\\")
                first_s = False
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    wtab("tabS_warmfull", L)

    s = rd("primary_closed_loop_summary.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Computation time per MPC step in the fixed-MPC primary experiment "
         r"(regime B), as a fraction of the sampling interval $dt$. Timings come from an "
         r"unoptimized single-process Python implementation on one machine and support relative "
         r"comparison only; they are not a real-time deployability claim.}",
         r"\label{tab:timing}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"System & Optimizer & mean (ms) & p95/$dt$ & max/$dt$ & det.\ ref. \\", r"\midrule"]
    for sy in SYS:
        g = s[(s.system == sy) & (s.regime == "B_common_NP")].set_index("alg")
        for i, alg in enumerate(ALGS):
            r = g.loc[alg]
            L.append(f"{SYSLAB[sy] if i == 0 else ''} & {alg} & {r.time_ms_mean:.1f} & "
                     f"{r.p95_over_dt:.3f} & {r.max_over_dt:.3f} & "
                     f"{'' if i else ''} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    wtab("tabS_timing", L)


def main():
    print("Phase 3B manuscript artifacts (analysis only, no simulation)")
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
    tab2_drift(); tab3_primary(); tab4_effects(); tab5_strata()
    tab6_warm(); tab7_dim_abl(); tab8_sens(); supp_tables()
    print("done")


if __name__ == "__main__":
    main()
