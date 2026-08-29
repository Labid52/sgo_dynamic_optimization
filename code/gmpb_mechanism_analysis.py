#!/usr/bin/env python3
"""Analysis of the GMPB grouping-mechanism ablation.

Reads committed run outputs only; runs no optimizer and no benchmark.
The independent unit is the run (benchmark seed).  Environment-level data are
aggregated to one value per run before any inferential test.

    python code/gmpb_mechanism_analysis.py <gmpb_dir> <mech_dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import CASES, CASE_LIST      # noqa: E402

INTERACTION_CASES = ["F2", "F8", "F10", "F12"]
# manuscript colours; the intervention is a variant of SGO, so it shares SGO's hue
C_RANDOM = "#0072B2"      # SGO blue, baseline random grouping
C_FITNESS = "#E69F00"     # orange, fitness-grouping ablation
C_REF = {"GWO": "#D55E00", "PSO": "#009E73", "WOA": "#CC79A7"}
ACCENT = "#56585B"
BOOT = 10000
RNG = np.random.default_rng(20260902)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return ((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b))


def boot_ci_delta(a, b, n=BOOT):
    ia = RNG.integers(0, len(a), (n, len(a)))
    ib = RNG.integers(0, len(b), (n, len(b)))
    out = np.array([cliffs_delta(a[ia[i]], b[ib[i]]) for i in range(n)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_ci_median(d, n=BOOT):
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = np.median(d[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ci_mean(d, n=BOOT):
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = np.mean(d[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def save(fig, outdir, stem):
    d = os.path.join(outdir, "figures")
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(d, f"{stem}.{ext}"), format=ext, dpi=300)
    plt.close(fig)
    print(f"  figure  {stem}.pdf/.eps")


def paired(a, b, label, family, case, unit, lower_is_better=True):
    """One paired comparison, a = fitness grouping, b = baseline random."""
    d = a - b
    try:
        st, p = wilcoxon(a, b)
    except ValueError:
        st, p = np.nan, 1.0
    mlo, mhi = boot_ci_median(d)
    alo, ahi = boot_ci_mean(d)
    dl = cliffs_delta(a, b)
    lo, hi = boot_ci_delta(a, b)
    base_med = float(np.median(b))
    return dict(family=family, case=case, quantity=unit, comparison=label,
                n_pairs=len(d),
                median_fitness=float(np.median(a)), median_random=base_med,
                mean_fitness=float(np.mean(a)), mean_random=float(np.mean(b)),
                sd_fitness=float(np.std(a, ddof=1)), sd_random=float(np.std(b, ddof=1)),
                iqr_fitness=float(np.percentile(a, 75) - np.percentile(a, 25)),
                iqr_random=float(np.percentile(b, 75) - np.percentile(b, 25)),
                median_diff=float(np.median(d)), median_diff_ci_lo=mlo, median_diff_ci_hi=mhi,
                mean_diff=float(np.mean(d)), mean_diff_ci_lo=alo, mean_diff_ci_hi=ahi,
                relative_median_diff=float(np.median(d) / base_med) if base_med else np.nan,
                wilcoxon_stat=float(st), p_raw=float(p),
                cliffs_delta=float(dl), delta_ci_lo=lo, delta_ci_hi=hi,
                favourable=bool(np.median(d) < 0) if lower_is_better
                else bool(np.median(d) > 0))


def per_run(envdf, cols):
    """Aggregate environment-level data to one value per run. Environments after
    the first are used, since environment 1 has no preceding change."""
    e = envdf[envdf.environment > 1]
    return e.groupby(["case", "seed"])[cols].mean().reset_index()


def main():
    _root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    gdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_root, "gmpb")
    mdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_root, "gmpb_mechanism")
    os.makedirs(mdir, exist_ok=True)

    # ---------------- data ----------------------------------------------
    base = pd.read_csv(os.path.join(gdir, "gmpb_primary_raw.csv"))
    base = base[base.alg == "SGO"].copy()
    base["grouping"] = "random"
    fit = pd.read_csv(os.path.join(mdir, "gmpb_fitness_group_raw.csv"))
    base_env = pd.read_csv(os.path.join(gdir, "gmpb_primary_environments.csv"))
    base_env = base_env[base_env.alg == "SGO"].copy()
    fit_env = pd.read_csv(os.path.join(mdir, "gmpb_fitness_group_environments.csv"))
    fit_div = pd.read_csv(os.path.join(mdir, "gmpb_fitness_group_diversity_trace.csv"))
    inter = pd.read_csv(os.path.join(mdir, "gmpb_temporal_interaction_raw.csv"))
    base_tmp = pd.read_csv(os.path.join(gdir, "gmpb_sgo_temporal_raw.csv"))

    # ---------------- summary -------------------------------------------
    rows = []
    for c in CASE_LIST:
        for g, d in (("random", base), ("fitness", fit)):
            v = d[d.case == c].sort_values("seed").offline_error.values
            rows.append(dict(case=c, grouping=g, n=len(v), **CASES[c],
                             best=v.min(), worst=v.max(), mean=v.mean(),
                             median=np.median(v), std=v.std(ddof=1),
                             q1=np.percentile(v, 25), q3=np.percentile(v, 75)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(mdir, "gmpb_mechanism_summary.csv"), index=False)

    # ---------------- primary statistics --------------------------------
    stats, ps = [], []
    for c in CASE_LIST:
        a = fit[fit.case == c].sort_values("seed").offline_error.values
        b = base[base.case == c].sort_values("seed").offline_error.values
        stats.append(paired(a, b, "fitness vs random grouping", "primary_offline_error",
                            c, "offline error"))
        ps.append(stats[-1]["p_raw"])
    for r, pa in zip(stats, holm(ps)):
        r["p_holm"] = float(pa)
        r["significant_holm"] = bool(pa < 0.05)

    # ---------------- mechanism diagnostics -----------------------------
    # The committed baseline environments file predates the fractional-error and
    # normalised-recovery columns.  gmpb_baseline_repro_* re-runs the baseline
    # through the same code path with the full diagnostics; its offline errors are
    # bit identical to the committed ones (checked in the report), so it is used
    # for the diagnostics while the committed file remains the performance record.
    repro = os.path.join(mdir, "gmpb_baseline_repro_environments.csv")
    be = pd.read_csv(repro)
    be = be[be.environment > 1].copy()
    be["recovery_frac_90"] = np.where(be.recovery_evals_90 > 0,
                                      be.recovery_evals_90 / be.change_frequency, np.nan)
    base_div_trace = pd.read_csv(os.path.join(mdir, "gmpb_baseline_repro_diversity_trace.csv"))
    fe = fit_env[fit_env.environment > 1]
    b_run = be.groupby(["case", "seed"]).agg(
        end_err=("end_of_environment_error", "mean"),
        rec90=("recovery_frac_90", "mean"),
        late=("late_half_fraction", "mean")).reset_index()
    f_run = fe.groupby(["case", "seed"]).agg(
        end_err=("end_of_environment_error", "mean"),
        rec90=("recovery_frac_90", "mean"),
        late=("late_half_fraction", "mean")).reset_index()
    # diversity per run
    b_div = base[["case", "seed", "mean_population_diversity"]].rename(
        columns={"mean_population_diversity": "div"})
    f_div = fit[["case", "seed", "mean_population_diversity"]].rename(
        columns={"mean_population_diversity": "div"})

    for name, bcol, fcol, bsrc, fsrc, lower in [
            ("mean population diversity", "div", "div", b_div, f_div, True),
            ("end-of-environment error", "end_err", "end_err", b_run, f_run, True),
            ("evaluation fraction to 90% recovery", "rec90", "rec90", b_run, f_run, False),
            ("late-half share of improvement", "late", "late", b_run, f_run, False)]:
        fam = "diagnostic_" + name.split()[0]
        sub, sp = [], []
        for c in CASE_LIST:
            a = fsrc[fsrc.case == c].sort_values("seed")[fcol].values
            b = bsrc[bsrc.case == c].sort_values("seed")[bcol].values
            sub.append(paired(a, b, "fitness vs random grouping", fam, c, name,
                              lower_is_better=lower))
            sp.append(sub[-1]["p_raw"])
        for r, pa in zip(sub, holm(sp)):
            r["p_holm"] = float(pa)
            r["significant_holm"] = bool(pa < 0.05)
        stats += sub

    # ---------------- 2x2 interaction -----------------------------------
    irows, istats = [], []
    for c in INTERACTION_CASES:
        cells = {}
        cells[("random", "population_persistence")] = base[base.case == c].sort_values("seed").offline_error.values
        cells[("fitness", "population_persistence")] = fit[fit.case == c].sort_values("seed").offline_error.values
        bt = base_tmp[(base_tmp.case == c) & (base_tmp["mode"] == "previous_best_seeded")]
        cells[("random", "previous_best_seeded")] = bt.sort_values("seed").offline_error.values
        it = inter[inter.case == c]
        cells[("fitness", "previous_best_seeded")] = it.sort_values("seed").offline_error.values
        for (g, m), v in cells.items():
            irows.append(dict(case=c, grouping=g, mode=m, n=len(v), best=v.min(),
                              worst=v.max(), mean=v.mean(), median=np.median(v),
                              std=v.std(ddof=1), q1=np.percentile(v, 25),
                              q3=np.percentile(v, 75)))
        eff = {}
        for m in ("population_persistence", "previous_best_seeded"):
            a = cells[("fitness", m)]
            b = cells[("random", m)]
            r = paired(a, b, f"fitness vs random within {m}", "interaction", c,
                       "offline error")
            istats.append(r)
            eff[m] = a - b
        dd = eff["population_persistence"] - eff["previous_best_seeded"]
        lo, hi = boot_ci_median(dd)
        istats.append(dict(family="interaction", case=c, quantity="offline error",
                           comparison="difference of differences "
                                      "(effect under persistence minus effect under previous-best)",
                           n_pairs=len(dd), median_diff=float(np.median(dd)),
                           median_diff_ci_lo=lo, median_diff_ci_hi=hi,
                           mean_diff=float(np.mean(dd)),
                           p_raw=np.nan, p_holm=np.nan, significant_holm=False,
                           favourable=bool(np.median(dd) < 0)))
    pd.DataFrame(irows).to_csv(
        os.path.join(mdir, "gmpb_temporal_interaction_summary.csv"), index=False)
    ip = [r["p_raw"] for r in istats if not np.isnan(r.get("p_raw", np.nan))]
    if ip:
        adj = holm(ip)
        k = 0
        for r in istats:
            if not np.isnan(r.get("p_raw", np.nan)):
                r["p_holm"] = float(adj[k]); r["significant_holm"] = bool(adj[k] < 0.05); k += 1
    stats += istats
    pd.DataFrame(stats).to_csv(os.path.join(mdir, "gmpb_mechanism_statistics.csv"), index=False)

    # ---------------- per-case classification ---------------------------
    st = pd.DataFrame(stats)
    cls = []
    for c in CASE_LIST:
        perf = st[(st.family == "primary_offline_error") & (st.case == c)].iloc[0]
        dv = st[(st.family == "diagnostic_mean") & (st.case == c)].iloc[0]
        rc = st[(st.family == "diagnostic_evaluation") & (st.case == c)].iloc[0]
        perf_ok = bool(perf.significant_holm and perf.favourable)
        div_ok = bool(dv.significant_holm and dv.favourable)
        rec_ok = bool(rc.significant_holm and rc.favourable)
        if perf_ok and div_ok and rec_ok:
            lab = "full chain"
        elif perf_ok and (div_ok or rec_ok):
            lab = "partial chain"
        elif perf_ok:
            lab = "performance only"
        elif div_ok or rec_ok:
            lab = "diagnostic only"
        else:
            lab = "no effect"
        cls.append(dict(case=c, **CASES[c],
                        offline_error_random=perf.median_random,
                        offline_error_fitness=perf.median_fitness,
                        offline_error_rel_change=perf.relative_median_diff,
                        offline_error_significant=perf_ok,
                        diversity_random=dv.median_random, diversity_fitness=dv.median_fitness,
                        diversity_reduced=div_ok,
                        recovery_frac90_random=rc.median_random,
                        recovery_frac90_fitness=rc.median_fitness,
                        recovery_later=rec_ok, classification=lab))
    pd.DataFrame(cls).to_csv(
        os.path.join(mdir, "gmpb_mechanism_case_classification.csv"), index=False)

    # ---------------- figures -------------------------------------------
    fig_offline(summ, st, gdir, mdir)
    fig_diversity(fit_div, base_div_trace, base, fit, mdir)
    fig_recovery(be, fe, mdir)
    fig_interaction(pd.DataFrame(irows), mdir)
    table_primary(summ, st, mdir)
    print("analysis done")


def fig_offline(summ, st, gdir, mdir):
    ref = pd.read_csv(os.path.join(gdir, "gmpb_primary_summary.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.8), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = ax[0]
    x = np.arange(len(CASE_LIST))
    w = 0.38
    for j, (g, col) in enumerate([("random", C_RANDOM), ("fitness", C_FITNESS)]):
        s = summ[summ.grouping == g].set_index("case").loc[CASE_LIST]
        a.bar(x + (j - 0.5) * w, s["median"], width=w, color=col, edgecolor="white",
              lw=0.5, label=f"{g} grouping",
              yerr=[s["median"] - s["q1"], s["q3"] - s["median"]],
              error_kw=dict(lw=0.6, ecolor=ACCENT, capsize=1.2))
    for alg in ("GWO", "PSO", "WOA"):
        r = ref[ref.alg == alg].set_index("case").loc[CASE_LIST]["median"]
        a.plot(x, r.values, ls="none", marker="_", ms=11, mew=1.3, color=C_REF[alg],
               label=f"{alg} (existing)")
    a.set_xticks(x); a.set_xticklabels(CASE_LIST)
    a.set_yscale("log")
    a.set_ylabel("offline error $E_o$ (median, IQR)")
    a.set_xlabel("GMPB competition instance")
    a.legend(ncol=3, fontsize=6, columnspacing=0.8)
    a.set_title("(a) SGO under two grouping rules; controls shown for context only",
                loc="left", fontsize=7.5)

    a = ax[1]
    p = st[st.family == "primary_offline_error"].set_index("case").loc[CASE_LIST]
    y = np.arange(len(CASE_LIST))[::-1]
    for yy, c in zip(y, CASE_LIST):
        r = p.loc[c]
        sig = r.significant_holm
        col = C_FITNESS if r.favourable else C_REF["GWO"]
        a.plot([r.median_diff_ci_lo, r.median_diff_ci_hi], [yy, yy], color=col,
               lw=1.3 if sig else 0.9, alpha=1.0 if sig else 0.6)
        a.plot(r.median_diff, yy, "o", ms=4.4 if sig else 3.4,
               mfc=col if sig else "white", mec=col, mew=1.0)
    a.axvline(0, color=ACCENT, ls="--", lw=0.8)
    a.set_yticks(y); a.set_yticklabels(CASE_LIST, fontsize=6.5)
    a.set_xlabel("paired median change in $E_o$\n(fitness $-$ random), 95% CI")
    a.set_title("(b) filled: Holm-significant", loc="left", fontsize=7.5)
    save(fig, mdir, "gmpb_grouping_offline_error")


def fig_diversity(fit_div, base_div, base, fit, mdir):
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.6), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    for df, g, col in [(base_div, "random", C_RANDOM), (fit_div, "fitness", C_FITNESS)]:
        d = df[df.environment > 1]
        prof = d.groupby("target_fraction").diversity.median()
        a.plot(prof.index, prof.values, "-o", color=col, ms=4, label=f"{g} grouping")
        q1 = d.groupby("target_fraction").diversity.quantile(0.25)
        q3 = d.groupby("target_fraction").diversity.quantile(0.75)
        a.fill_between(prof.index, q1.values, q3.values, color=col, alpha=0.18, lw=0)
    a.set_xlabel("fraction of the environment budget")
    a.set_ylabel("normalised population diversity")
    a.set_title("(a) within-environment diversity, 12 cases pooled (median, IQR)",
                loc="left", fontsize=7.5)
    a.legend(fontsize=6.5)

    a = ax[1]
    x = np.arange(len(CASE_LIST))
    w = 0.38
    for j, (df, g, col) in enumerate([(base, "random", C_RANDOM), (fit, "fitness", C_FITNESS)]):
        v = df.groupby("case").mean_population_diversity.median().reindex(CASE_LIST)
        a.bar(x + (j - 0.5) * w, v.values, width=w, color=col, edgecolor="white",
              lw=0.5, label=f"{g} grouping")
    a.set_xticks(x); a.set_xticklabels(CASE_LIST, fontsize=6.5, rotation=90)
    a.set_ylabel("run-mean diversity")
    a.set_yscale("log")
    a.legend(fontsize=6.5)
    a.set_title("(b) per case", loc="left", fontsize=7.5)
    save(fig, mdir, "gmpb_grouping_diversity")


def fig_recovery(be, fe, mdir):
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.4), layout="constrained")
    a = ax[0]
    for df, g, col in [(be, "random", C_RANDOM), (fe, "fitness", C_FITNESS)]:
        v = df.groupby("case").post_change_error.median().reindex(CASE_LIST)
        a.plot(range(len(CASE_LIST)), v.values, "-o", ms=3.4, color=col, mfc="none",
               mew=1.0, label=f"{g} grouping")
    a.set_xticks(range(len(CASE_LIST)))
    a.set_xticklabels(CASE_LIST, fontsize=5.6, rotation=90)
    a.set_yscale("log"); a.set_ylabel("median post-change $E$")
    a.set_title("(a) error just after a change", loc="left", fontsize=7.5)
    a.legend(fontsize=6.5)

    a = ax[1]
    for df, g, col in [(be, "random", C_RANDOM), (fe, "fitness", C_FITNESS)]:
        v = df.groupby("case").end_of_environment_error.median().reindex(CASE_LIST)
        a.plot(range(len(CASE_LIST)), v.values, "-o", ms=3.4, color=col, mfc="none",
               mew=1.0)
    a.set_xticks(range(len(CASE_LIST)))
    a.set_xticklabels(CASE_LIST, fontsize=5.6, rotation=90)
    a.set_yscale("log"); a.set_ylabel("median end-of-env $E$")
    a.set_title("(b) error before the next change", loc="left", fontsize=7.5)

    a = ax[2]
    for df, g, col in [(be, "random", C_RANDOM), (fe, "fitness", C_FITNESS)]:
        v = df.groupby("case").recovery_frac_90.median().reindex(CASE_LIST)
        a.plot(range(len(CASE_LIST)), v.values, "-o", ms=3.4, color=col, mfc="none",
               mew=1.0)
    a.set_xticks(range(len(CASE_LIST)))
    a.set_xticklabels(CASE_LIST, fontsize=5.6, rotation=90)
    a.set_ylabel("fraction of environment used")
    a.set_title("(c) evaluations to 90% recovery", loc="left", fontsize=7.5)
    save(fig, mdir, "gmpb_grouping_recovery")


def fig_interaction(irows, mdir):
    fig, ax = plt.subplots(1, 4, figsize=(7.3, 2.5), layout="constrained")
    modes = ["population_persistence", "previous_best_seeded"]
    lab = {"population_persistence": "persist", "previous_best_seeded": "prev-best"}
    for i, c in enumerate(INTERACTION_CASES):
        a = ax[i]
        for j, g in enumerate(["random", "fitness"]):
            v = [irows[(irows.case == c) & (irows.grouping == g)
                       & (irows["mode"] == m)]["median"].iloc[0] for m in modes]
            a.bar(np.arange(2) + (j - 0.5) * 0.38, v, width=0.38,
                  color=C_RANDOM if g == "random" else C_FITNESS,
                  edgecolor="white", lw=0.5, label=f"{g}")
        a.set_xticks(range(2))
        a.set_xticklabels([lab[m] for m in modes], fontsize=6.5)
        a.set_title(c, fontsize=8)
        if i == 0:
            a.set_ylabel("median offline error $E_o$")
            a.legend(fontsize=6)
    save(fig, mdir, "gmpb_grouping_temporal_interaction")


def table_primary(summ, st, mdir):
    p = st[st.family == "primary_offline_error"].set_index("case")
    dv = st[st.family == "diagnostic_mean"].set_index("case")
    rc = st[st.family == "diagnostic_evaluation"].set_index("case")
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Fitness-based grouping ablation of SGO on GMPB, 31 paired runs per case. "
         r"Baseline is vanilla SGO under the same official protocol. Paired Wilcoxon with Holm "
         r"correction across the 12 cases. Diagnostic only; the ablation is not proposed as an "
         r"algorithm and is not compared with the controls.}",
         r"\label{tab:gmpb_grouping}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"Case & $E_o$ random & $E_o$ fitness & median diff [95\% CI] & rel. & "
         r"$p_{\mathrm{Holm}}$ & Cliff's $\delta$ & diversity r$\to$f & 90\% rec. r$\to$f \\",
         r"\midrule"]
    for c in CASE_LIST:
        r = p.loc[c]; d = dv.loc[c]; q = rc.loc[c]
        L.append(f"{c} & {r.median_random:.4g} & {r.median_fitness:.4g} & "
                 f"{r.median_diff:+.3g} [{r.median_diff_ci_lo:+.3g}, {r.median_diff_ci_hi:+.3g}] & "
                 f"{100*r.relative_median_diff:+.1f}\\% & {r.p_holm:.2g} & "
                 f"{r.cliffs_delta:+.2f} & "
                 f"{d.median_random:.3g}$\\to${d.median_fitness:.3g} & "
                 f"{q.median_random:.3g}$\\to${q.median_fitness:.3g} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    dd = os.path.join(mdir, "tables")
    os.makedirs(dd, exist_ok=True)
    open(os.path.join(dd, "gmpb_grouping_summary.tex"), "w").write("\n".join(L) + "\n")
    print("  table   gmpb_grouping_summary.tex")


if __name__ == "__main__":
    main()
