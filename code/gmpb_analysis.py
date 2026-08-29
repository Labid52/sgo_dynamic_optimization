#!/usr/bin/env python3
"""Analysis of the GMPB dynamic study: summaries, statistics, tables, figures.

Reads only the committed run outputs; runs no optimizer and no benchmark.

    python code/gmpb_analysis.py <outdir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import CASES, CASE_LIST   # noqa: E402

ALGS = ["SGO", "GWO", "PSO", "WOA"]
MODES = ["cold_restart", "previous_best_seeded", "population_persistence"]
MODE_LABEL = {"cold_restart": "cold restart",
              "previous_best_seeded": "previous-best seeded",
              "population_persistence": "population persistence"}
# same optimizer colours as the manuscript figures
COL = {"SGO": "#0072B2", "GWO": "#D55E00", "PSO": "#009E73", "WOA": "#CC79A7"}
MK = {"SGO": "o", "GWO": "s", "PSO": "^", "WOA": "D"}
LS = {"SGO": "-", "GWO": "--", "PSO": "-.", "WOA": ":"}
ACCENT = "#56585B"
BOOT = 10000
RNG = np.random.default_rng(20260901)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------- helpers
def cliffs_delta(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def boot_ci_delta(a, b, n=BOOT):
    out = np.empty(n)
    ia = RNG.integers(0, len(a), (n, len(a)))
    ib = RNG.integers(0, len(b), (n, len(b)))
    for i in range(n):
        out[i] = cliffs_delta(a[ia[i]], b[ib[i]])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_ci_median_diff(d, n=BOOT):
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = np.median(d[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        v = (len(p) - rank) * p[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return adj


def save(fig, outdir, stem):
    d = os.path.join(outdir, "figures")
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(d, f"{stem}.{ext}"), format=ext, dpi=300)
    plt.close(fig)
    print(f"  figure  {stem}.pdf/.eps")


def esc(x, sig=3):
    if not np.isfinite(x):
        return "--"
    return f"{x:.{sig}g}"


# ---------------------------------------------------------------- summaries
def primary_summary(raw, outdir):
    off = raw.pivot_table(index=["case", "seed"], columns="alg",
                          values="offline_error").reset_index()
    off["case_order"] = off.case.map({c: i for i, c in enumerate(CASE_LIST)})
    off = off.sort_values(["case_order", "seed"]).drop(columns="case_order")
    off.to_csv(os.path.join(outdir, "gmpb_primary_offline_error.csv"), index=False)

    rows = []
    for c in CASE_LIST:
        g = raw[raw.case == c]
        med = {a: g[g.alg == a].offline_error.median() for a in ALGS}
        rank = {a: r + 1 for r, a in enumerate(sorted(ALGS, key=lambda x: med[x]))}
        for a in ALGS:
            v = g[g.alg == a].offline_error.values
            rows.append(dict(case=c, alg=a, n=len(v), **CASES[c],
                             best=v.min(), worst=v.max(), mean=v.mean(),
                             median=np.median(v), std=v.std(ddof=1),
                             q1=np.percentile(v, 25), q3=np.percentile(v, 75),
                             iqr=np.percentile(v, 75) - np.percentile(v, 25),
                             rank=rank[a],
                             ebbc_mean=g[g.alg == a].best_error_before_change.mean()))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(outdir, "gmpb_primary_summary.csv"), index=False)
    return off, s


def primary_statistics(off, outdir):
    rows = []
    for c in CASE_LIST:
        g = off[off.case == c]
        ps, recs = [], []
        for other in ["GWO", "PSO", "WOA"]:
            a = g["SGO"].values
            b = g[other].values
            d = a - b
            try:
                st, p = wilcoxon(a, b)
            except ValueError:
                st, p = np.nan, 1.0
            dl = cliffs_delta(a, b)
            lo, hi = boot_ci_delta(a, b)
            mlo, mhi = boot_ci_median_diff(d)
            ps.append(p)
            recs.append(dict(family="primary", case=c, comparison=f"SGO vs {other}",
                             n_pairs=len(d),
                             median_SGO=float(np.median(a)), median_other=float(np.median(b)),
                             median_diff=float(np.median(d)),
                             median_diff_ci_lo=mlo, median_diff_ci_hi=mhi,
                             wilcoxon_stat=float(st), p_raw=float(p),
                             cliffs_delta=float(dl), delta_ci_lo=lo, delta_ci_hi=hi,
                             sgo_lower_offline_error=bool(np.median(d) < 0)))
        adj = holm(ps)
        for r, pa in zip(recs, adj):
            r["p_holm"] = float(pa)
            r["significant_holm"] = bool(pa < 0.05)
            rows.append(r)
    return rows


def temporal_summary(tr, outdir):
    piv = tr.pivot_table(index=["case", "seed"], columns="mode",
                         values="offline_error").reset_index()
    rows = []
    for c in sorted(tr.case.unique(), key=lambda x: CASE_LIST.index(x)):
        g = tr[tr.case == c]
        for m in MODES:
            v = g[g["mode"] == m].offline_error.values
            rows.append(dict(case=c, mode=m, n=len(v), best=v.min(), worst=v.max(),
                             mean=v.mean(), median=np.median(v), std=v.std(ddof=1),
                             q1=np.percentile(v, 25), q3=np.percentile(v, 75)))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(outdir, "gmpb_sgo_temporal_summary.csv"), index=False)

    stat = []
    for c in piv.case.unique():
        g = piv[piv.case == c]
        pairs = [("cold_restart", "previous_best_seeded"),
                 ("cold_restart", "population_persistence"),
                 ("previous_best_seeded", "population_persistence")]
        ps, recs = [], []
        for m1, m2 in pairs:
            a, b = g[m1].values, g[m2].values
            d = a - b
            try:
                st, p = wilcoxon(a, b)
            except ValueError:
                st, p = np.nan, 1.0
            lo, hi = boot_ci_delta(a, b)
            mlo, mhi = boot_ci_median_diff(d)
            ps.append(p)
            recs.append(dict(family="temporal", case=c,
                             comparison=f"{MODE_LABEL[m1]} vs {MODE_LABEL[m2]}",
                             n_pairs=len(d),
                             median_SGO=float(np.median(a)), median_other=float(np.median(b)),
                             median_diff=float(np.median(d)),
                             median_diff_ci_lo=mlo, median_diff_ci_hi=mhi,
                             wilcoxon_stat=float(st), p_raw=float(p),
                             cliffs_delta=float(cliffs_delta(a, b)),
                             delta_ci_lo=lo, delta_ci_hi=hi,
                             sgo_lower_offline_error=bool(np.median(d) < 0)))
        adj = holm(ps)
        for r, pa in zip(recs, adj):
            r["p_holm"] = float(pa)
            r["significant_holm"] = bool(pa < 0.05)
            stat.append(r)
    return s, stat


def factor_analysis(summ, outdir):
    axes = {"peak_number": (["F1", "F2", "F3", "F4", "F5"], "PeakNumber"),
            "change_frequency": (["F2", "F6", "F7", "F8"], "ChangeFrequency"),
            "dimension": (["F2", "F9", "F10"], "Dimension"),
            "shift_severity": (["F2", "F11", "F12"], "ShiftSeverity")}
    rows = []
    for ax, (cases, param) in axes.items():
        base = cases[0]
        for a in ALGS:
            g = summ[(summ.alg == a) & (summ.case.isin(cases))].set_index("case")
            xs = np.array([CASES[c][param] for c in cases], float)
            ys = np.array([g.loc[c, "median"] for c in cases], float)
            rho, p = spearmanr(xs, ys)
            for c in cases:
                rows.append(dict(axis=ax, parameter=param, case=c,
                                 parameter_value=CASES[c][param], alg=a,
                                 median_offline_error=float(g.loc[c, "median"]),
                                 mean_offline_error=float(g.loc[c, "mean"]),
                                 rank_in_case=int(g.loc[c, "rank"]),
                                 relative_to_baseline=float(g.loc[c, "median"] / g.loc[base, "median"]),
                                 spearman_param_vs_median=float(rho),
                                 spearman_p=float(p)))
    f = pd.DataFrame(rows)
    f.to_csv(os.path.join(outdir, "gmpb_factor_analysis.csv"), index=False)
    return f, axes


# ---------------------------------------------------------------- figures
def fig_across_cases(summ, off, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.8), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = ax[0]
    x = np.arange(len(CASE_LIST))
    w = 0.2
    for j, alg in enumerate(ALGS):
        g = summ[summ.alg == alg].set_index("case").loc[CASE_LIST]
        a.bar(x + (j - 1.5) * w, g["median"], width=w, color=COL[alg],
              edgecolor="white", lw=0.5, label=alg,
              yerr=[g["median"] - g["q1"], g["q3"] - g["median"]],
              error_kw=dict(lw=0.6, ecolor=ACCENT, capsize=1.2))
    a.set_xticks(x); a.set_xticklabels(CASE_LIST)
    a.set_yscale("log")
    a.set_ylabel("offline error $E_o$ (median, IQR bars)")
    a.set_xlabel("GMPB competition instance")
    a.legend(ncol=4, columnspacing=0.8)
    a.set_title("(a) 31 runs per case, exact official budgets", loc="left")

    a = ax[1]
    for j, alg in enumerate(ALGS):
        g = summ[summ.alg == alg].set_index("case").loc[CASE_LIST]
        a.plot(x, g["rank"], ls=LS[alg], marker=MK[alg], ms=4.2, color=COL[alg],
               mfc="none", mew=1.1, label=alg)
    a.set_xticks(x); a.set_xticklabels(CASE_LIST, fontsize=6)
    a.set_yticks([1, 2, 3, 4]); a.invert_yaxis()
    a.set_ylabel("rank by median $E_o$ (1 = best)")
    a.set_title("(b) rank by case", loc="left")
    save(fig, outdir, "gmpb_sgo_across_cases")


def fig_factor_effects(fac, axes, outdir):
    fig, ax = plt.subplots(1, 4, figsize=(7.3, 2.4), layout="constrained")
    titles = {"peak_number": "(a) number of peaks",
              "change_frequency": "(b) change frequency",
              "dimension": "(c) dimension",
              "shift_severity": "(d) shift severity"}
    for i, (axname, (cases, param)) in enumerate(axes.items()):
        a = ax[i]
        for alg in ALGS:
            g = fac[(fac.axis == axname) & (fac.alg == alg)].sort_values("parameter_value")
            lw = 1.6 if alg == "SGO" else 0.9
            al = 1.0 if alg == "SGO" else 0.65
            a.plot(g.parameter_value, g.median_offline_error, ls=LS[alg],
                   marker=MK[alg], ms=4.2 if alg == "SGO" else 3.4, color=COL[alg],
                   mfc="none", mew=1.1, lw=lw, alpha=al, label=alg)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(param)
        if i == 0:
            a.set_ylabel("median offline error $E_o$")
            a.legend(fontsize=6, ncol=2)
        a.set_title(titles[axname], loc="left", fontsize=7.5)
        vals = sorted(fac[fac.axis == axname].parameter_value.unique())
        a.set_xticks(vals)
        a.set_xticklabels([f"{int(v)}" for v in vals])
        a.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
        a.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    save(fig, outdir, "gmpb_factor_effects")


def fig_temporal(tsum, tr, outdir):
    cases = [c for c in CASE_LIST if c in set(tsum.case)]
    fig, ax = plt.subplots(1, len(cases), figsize=(7.3, 2.5), layout="constrained",
                           sharey=False)
    shade = {"cold_restart": "#BBBBBB", "previous_best_seeded": "#7FA8CC",
             "population_persistence": COL["SGO"]}
    for i, c in enumerate(cases):
        a = ax[i]
        data = [tr[(tr.case == c) & (tr["mode"] == m)].offline_error.values for m in MODES]
        bp = a.boxplot(data, positions=range(len(MODES)), widths=0.6,
                       showfliers=False, patch_artist=True,
                       medianprops=dict(color="k", lw=1.0))
        for b, m in zip(bp["boxes"], MODES):
            b.set(facecolor=shade[m], alpha=0.75, lw=0.6, edgecolor=ACCENT)
        for j, m in enumerate(MODES):
            v = tr[(tr.case == c) & (tr["mode"] == m)].offline_error.values
            a.plot(np.full(len(v), j) + RNG.normal(0, 0.05, len(v)), v, ".",
                   ms=2.0, color=ACCENT, alpha=0.55)
        a.set_xticks(range(len(MODES)))
        a.set_xticklabels(["cold", "prev-best", "persist"], fontsize=6.5, rotation=20,
                          ha="right")
        a.set_title(f"{c}", fontsize=8)
        if i == 0:
            a.set_ylabel("SGO offline error $E_o$")
    save(fig, outdir, "gmpb_sgo_temporal_modes")


def fig_recovery(envp, outdir):
    d = pd.read_csv(envp)
    d = d[d.environment > 1]              # environment 1 has no preceding change
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.4), layout="constrained")
    a = ax[0]
    for alg in ALGS:
        g = d[d.alg == alg].groupby("case").post_change_error.median()
        g = g.reindex(CASE_LIST)
        a.plot(range(len(CASE_LIST)), g.values, ls=LS[alg], marker=MK[alg], ms=3.6,
               color=COL[alg], mfc="none", mew=1.0, label=alg)
    a.set_xticks(range(len(CASE_LIST)))
    a.set_xticklabels(CASE_LIST, fontsize=5.6, rotation=90)
    a.set_yscale("log"); a.set_ylabel("median post-change $E$")
    a.set_title("(a) post-change error", loc="left", fontsize=7.5)
    a.legend(fontsize=6, ncol=2)

    a = ax[1]
    for alg in ALGS:
        g = d[d.alg == alg].groupby("case").end_of_environment_error.median()
        g = g.reindex(CASE_LIST)
        a.plot(range(len(CASE_LIST)), g.values, ls=LS[alg], marker=MK[alg], ms=3.6,
               color=COL[alg], mfc="none", mew=1.0)
    a.set_xticks(range(len(CASE_LIST)))
    a.set_xticklabels(CASE_LIST, fontsize=5.6, rotation=90)
    a.set_yscale("log"); a.set_ylabel("median end-of-env $E$")
    a.set_title("(b) end-of-environment error", loc="left", fontsize=7.5)

    a = ax[2]
    for alg in ALGS:
        g = d[(d.alg == alg) & (d.recovery_evals_90 > 0)]
        gg = g.groupby("case").apply(
            lambda x: np.median(x.recovery_evals_90 / x.change_frequency)
            if "change_frequency" in x else np.nan)
        vals = []
        for c in CASE_LIST:
            sub = g[g.case == c]
            cf = CASES[c]["ChangeFrequency"]
            vals.append(np.median(sub.recovery_evals_90 / cf) if len(sub) else np.nan)
        a.plot(range(len(CASE_LIST)), vals, ls=LS[alg], marker=MK[alg], ms=3.6,
               color=COL[alg], mfc="none", mew=1.0)
    a.set_xticks(range(len(CASE_LIST)))
    a.set_xticklabels(CASE_LIST, fontsize=5.6, rotation=90)
    a.set_ylabel("fraction of the environment used")
    a.set_title("(c) evaluations to 90% recovery", loc="left", fontsize=7.5)
    save(fig, outdir, "gmpb_recovery_after_change")


# ---------------------------------------------------------------- tables
def table_primary(summ, stat, outdir):
    st = pd.DataFrame(stat)
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{GMPB offline error $E_o$ over 31 independent runs per case at the "
         r"official budgets. Lower is better. The three right-hand columns give the "
         r"comparator medians, and the last column SGO's rank of four by median $E_o$. "
         r"Evidence tables: \texttt{gmpb\_primary\_summary.csv}, "
         r"\texttt{gmpb\_statistics.csv}.}",
         r"\label{tab:gmpb_primary}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
         r"Case & peaks / $cf$ / $d$ / $\tilde{s}$ & \multicolumn{3}{c}{SGO} & "
         r"GWO & PSO & WOA & SGO rank \\",
         r"\cmidrule(lr){3-5}",
         r" & & mean & median & SD & median & median & median & \\", r"\midrule"]
    for c in CASE_LIST:
        g = summ[summ.case == c].set_index("alg")
        s = g.loc["SGO"]
        L.append(f"{c} & {CASES[c]['PeakNumber']} / {CASES[c]['ChangeFrequency']} / "
                 f"{CASES[c]['Dimension']} / {CASES[c]['ShiftSeverity']} & "
                 f"{esc(s['mean'],4)} & {esc(s['median'],4)} & {esc(s['std'],3)} & "
                 f"{esc(g.loc['GWO','median'],4)} & {esc(g.loc['PSO','median'],4)} & "
                 f"{esc(g.loc['WOA','median'],4)} & {int(s['rank'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    d = os.path.join(outdir, "tables")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "gmpb_primary_summary.tex"), "w").write("\n".join(L) + "\n")
    print("  table   gmpb_primary_summary.tex")


def table_temporal(tsum, stat, outdir):
    st = pd.DataFrame([s for s in stat if s["family"] == "temporal"])
    cases = [c for c in CASE_LIST if c in set(tsum.case)]
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{SGO temporal-transfer diagnostic: offline error $E_o$ over 31 paired "
         r"runs for three ways of carrying information across an environmental change. "
         r"The modes share every SGO operator and differ only in what survives a change. "
         r"Paired Wilcoxon with Holm correction inside each case; Cliff's $\delta$ with "
         r"bootstrap 95\% interval. Evidence: \texttt{gmpb\_sgo\_temporal\_summary.csv}, "
         r"\texttt{gmpb\_statistics.csv}.}",
         r"\label{tab:gmpb_temporal}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"Case & Mode & mean & median & SD & \multicolumn{3}{c}{vs population persistence} \\",
         r"\cmidrule(lr){6-8}",
         r" & & & & & median diff & $p_{\mathrm{Holm}}$ & Cliff's $\delta$ \\", r"\midrule"]
    for c in cases:
        for i, m in enumerate(MODES):
            r = tsum[(tsum.case == c) & (tsum["mode"] == m)].iloc[0]
            cmp_ = ""
            if m != "population_persistence":
                q = st[(st.case == c) &
                       (st.comparison == f"{MODE_LABEL[m]} vs population persistence")]
                if len(q):
                    q = q.iloc[0]
                    cmp_ = (f"{esc(q.median_diff,3)} & {esc(q.p_holm,2)} & "
                            f"{q.cliffs_delta:+.2f} [{q.delta_ci_lo:+.2f}, {q.delta_ci_hi:+.2f}]")
            if not cmp_:
                cmp_ = "-- & -- & --"
            L.append(f"{c if i == 0 else ''} & {MODE_LABEL[m]} & {esc(r['mean'],4)} & "
                     f"{esc(r['median'],4)} & {esc(r['std'],3)} & {cmp_} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    d = os.path.join(outdir, "tables")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "gmpb_sgo_temporal_modes.tex"), "w").write("\n".join(L) + "\n")
    print("  table   gmpb_sgo_temporal_modes.tex")


def main():
    # default to the committed GMPB data directory so the script is runnable
    # with no arguments from inside the released package
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gmpb")
    raw = pd.read_csv(os.path.join(outdir, "gmpb_primary_raw.csv"))
    off, summ = primary_summary(raw, outdir)
    stat = primary_statistics(off, outdir)

    tp = os.path.join(outdir, "gmpb_sgo_temporal_raw.csv")
    tsum = None
    if os.path.exists(tp):
        tr = pd.read_csv(tp)
        tsum, tstat = temporal_summary(tr, outdir)
        stat += tstat
    pd.DataFrame(stat).to_csv(os.path.join(outdir, "gmpb_statistics.csv"), index=False)

    fac, axes = factor_analysis(summ, outdir)
    fig_across_cases(summ, off, outdir)
    fig_factor_effects(fac, axes, outdir)
    envp = os.path.join(outdir, "gmpb_primary_environments.csv")
    if os.path.exists(envp):
        fig_recovery(envp, outdir)
    if tsum is not None:
        fig_temporal(tsum, tr, outdir)
        table_temporal(tsum, stat, outdir)
    table_primary(summ, stat, outdir)
    print("analysis done")


if __name__ == "__main__":
    main()
