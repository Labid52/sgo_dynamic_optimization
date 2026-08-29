#!/usr/bin/env python3
"""Analysis of the comparator temporal-transfer study.

Reads committed run outputs only.  The independent unit is the run; environment
level data are aggregated within run before any inferential test.

    python code/gmpb_temporal_analysis.py <outdir>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

CASES = ["F2", "F8", "F10", "F12"]
ALGS = ["SGO", "GWO", "PSO", "WOA"]
MODES = ["cold_restart", "previous_best_seeded", "population_persistence"]
MLAB = {"cold_restart": "cold", "previous_best_seeded": "prev-best",
        "population_persistence": "persist"}
COL = {"SGO": "#0072B2", "GWO": "#D55E00", "PSO": "#009E73", "WOA": "#CC79A7"}
MK = {"SGO": "o", "GWO": "s", "PSO": "^", "WOA": "D"}
ACCENT = "#56585B"
BOOT = 10000
RNG = np.random.default_rng(20260903)

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


def boot_ci(v, stat=np.median, n=BOOT):
    idx = RNG.integers(0, len(v), (n, len(v)))
    s = stat(np.asarray(v)[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


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


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "gmpb_temporal_comparators")
    raw = pd.read_csv(os.path.join(outdir, "gmpb_temporal_comparator_raw.csv"),
                      float_precision="round_trip")

    # ---------------- summary -------------------------------------------
    rows = []
    for c in CASES:
        for a in ALGS:
            for m in MODES:
                v = raw[(raw.case == c) & (raw.alg == a) & (raw["mode"] == m)] \
                    .sort_values("seed").offline_error.values
                lo, hi = boot_ci(v)
                rows.append(dict(case=c, alg=a, mode=m, n=len(v), best=v.min(),
                                 worst=v.max(), mean=v.mean(), median=np.median(v),
                                 std=v.std(ddof=1), q1=np.percentile(v, 25),
                                 q3=np.percentile(v, 75),
                                 median_ci_lo=lo, median_ci_hi=hi))
    summ = pd.DataFrame(rows)
    # descriptive rank within each case and mode
    summ["rank_in_case_mode"] = summ.groupby(["case", "mode"])["median"].rank().astype(int)
    summ.to_csv(os.path.join(outdir, "gmpb_temporal_comparator_summary.csv"), index=False)

    # ---------------- within-optimizer temporal effects ------------------
    def vec(c, a, m):
        return raw[(raw.case == c) & (raw.alg == a) & (raw["mode"] == m)] \
            .sort_values("seed").offline_error.values

    EFF = {"E1_prevbest_minus_cold": ("previous_best_seeded", "cold_restart"),
           "E2_persist_minus_cold": ("population_persistence", "cold_restart"),
           "E3_persist_minus_prevbest": ("population_persistence", "previous_best_seeded")}
    eff_rows = []
    for c in CASES:
        for a in ALGS:
            ps = []
            block = []
            for name, (mA, mB) in EFF.items():
                A, B = vec(c, a, mA), vec(c, a, mB)
                d = A - B
                lr = np.log(A / B)
                try:
                    st, p = wilcoxon(A, B)
                except ValueError:
                    st, p = np.nan, 1.0
                dlo, dhi = boot_ci(d)
                llo, lhi = boot_ci(lr)
                ps.append(p)
                block.append(dict(case=c, alg=a, effect=name,
                                  mode_A=mA, mode_B=mB, n=len(d),
                                  median_A=float(np.median(A)), median_B=float(np.median(B)),
                                  median_diff=float(np.median(d)),
                                  diff_ci_lo=dlo, diff_ci_hi=dhi,
                                  median_log_ratio=float(np.median(lr)),
                                  log_ratio_ci_lo=llo, log_ratio_ci_hi=lhi,
                                  percent_change=float(100 * (np.exp(np.median(lr)) - 1)),
                                  wilcoxon_stat=float(st), p_raw=float(p),
                                  cliffs_delta=float(cliffs_delta(A, B)),
                                  A_better=bool(np.median(d) < 0)))
            for r, pa in zip(block, holm(ps)):
                r["p_holm"] = float(pa)
                r["significant_holm"] = bool(pa < 0.05)
            eff_rows += block
    eff = pd.DataFrame(eff_rows)
    eff.to_csv(os.path.join(outdir, "gmpb_temporal_effects.csv"), index=False)

    # ---------------- cross-optimizer comparison -------------------------
    cross = []
    for c in CASES:
        for name, (mA, mB) in EFF.items():
            sgo_lr = np.log(vec(c, "SGO", mA) / vec(c, "SGO", mB))
            ps, block = [], []
            for other in ["GWO", "PSO", "WOA"]:
                oth_lr = np.log(vec(c, other, mA) / vec(c, other, mB))
                d = sgo_lr - oth_lr          # paired on the shared benchmark seed
                try:
                    st, p = wilcoxon(sgo_lr, oth_lr)
                except ValueError:
                    st, p = np.nan, 1.0
                lo, hi = boot_ci(d)
                ps.append(p)
                block.append(dict(case=c, effect=name, comparison=f"SGO vs {other}",
                                  n=len(d),
                                  sgo_median_log_ratio=float(np.median(sgo_lr)),
                                  other_median_log_ratio=float(np.median(oth_lr)),
                                  median_difference=float(np.median(d)),
                                  diff_ci_lo=lo, diff_ci_hi=hi,
                                  wilcoxon_stat=float(st), p_raw=float(p),
                                  cliffs_delta=float(cliffs_delta(sgo_lr, oth_lr)),
                                  sgo_effect_larger=bool(np.median(d) > 0),
                                  same_sign=bool(np.sign(np.median(sgo_lr))
                                                 == np.sign(np.median(oth_lr)))))
            for r, pa in zip(block, holm(ps)):
                r["p_holm"] = float(pa)
                r["significant_holm"] = bool(pa < 0.05)
            cross += block
    cr = pd.DataFrame(cross)
    cr.to_csv(os.path.join(outdir, "gmpb_temporal_cross_optimizer_statistics.csv"),
              index=False)

    # ---------------- behavioural diagnostics ----------------------------
    env = pd.read_csv(os.path.join(outdir, "gmpb_temporal_comparator_environments.csv"),
                      float_precision="round_trip")
    env = env[env.environment > 1]
    per_run = env.groupby(["case", "alg", "mode", "seed"]).agg(
        end_error=("end_of_environment_error", "mean"),
        post_change_error=("post_change_error", "mean"),
        recovery_frac_50=("recovery_frac_50", "mean"),
        recovery_frac_90=("recovery_frac_90", "mean"),
        late_half_fraction=("late_half_fraction", "mean")).reset_index()
    div = raw[["case", "alg", "mode", "seed", "mean_population_diversity"]]
    diag = per_run.merge(div, on=["case", "alg", "mode", "seed"])
    diag.to_csv(os.path.join(outdir, "gmpb_temporal_behavioral_diagnostics.csv"),
                index=False)

    # ---------------- pattern classification -----------------------------
    cls = []
    for c in CASES:
        for name in EFF:
            e = eff[(eff.case == c) & (eff.effect == name)].set_index("alg")
            lr = {a: float(e.loc[a, "median_log_ratio"]) for a in ALGS}
            sig = {a: bool(e.loc[a, "significant_holm"]) for a in ALGS}
            x = cr[(cr.case == c) & (cr.effect == name)]
            n_sig_diff = int(x.significant_holm.sum())
            same = all(np.sign(lr[a]) == np.sign(lr["SGO"]) for a in ALGS)
            # "amplified" means SGO's effect is further from zero in the SAME
            # direction, not merely that a signed difference is positive.
            further = all(abs(lr["SGO"]) > abs(lr[a]) for a in ALGS if a != "SGO")
            weaker = all(abs(lr["SGO"]) < abs(lr[a]) for a in ALGS if a != "SGO")
            inside = (min(lr[a] for a in ALGS if a != "SGO") <= lr["SGO"]
                      <= max(lr[a] for a in ALGS if a != "SGO"))
            if not same:
                lab = "ALGORITHM-DEPENDENT"
            elif n_sig_diff == 0:
                lab = "GENERAL"
            elif further and n_sig_diff == 3:
                lab = "SGO-AMPLIFIED"
            elif inside:
                lab = "GENERAL"          # SGO lies within the comparator spread
            else:
                lab = "ALGORITHM-DEPENDENT"
            cls.append(dict(case=c, effect=name,
                            **{f"log_ratio_{a}": lr[a] for a in ALGS},
                            **{f"significant_{a}": sig[a] for a in ALGS},
                            all_same_direction=same,
                            n_cross_comparisons_significant=n_sig_diff,
                            sgo_further_from_zero_than_all=further,
                            sgo_weaker_than_all=weaker,
                            sgo_inside_comparator_range=inside,
                            classification=lab))
    pd.DataFrame(cls).to_csv(
        os.path.join(outdir, "gmpb_temporal_pattern_classification.csv"), index=False)

    fig_modes(summ, outdir)
    fig_effects(eff, outdir)
    fig_cross(cr, outdir)
    fig_behaviour(diag, outdir)
    print("analysis done")


def fig_modes(summ, outdir):
    fig, ax = plt.subplots(1, 4, figsize=(7.3, 2.6), layout="constrained")
    for i, c in enumerate(CASES):
        a = ax[i]
        x = np.arange(3)
        w = 0.2
        for j, alg in enumerate(ALGS):
            s = summ[(summ.case == c) & (summ.alg == alg)].set_index("mode").loc[MODES]
            a.bar(x + (j - 1.5) * w, s["median"], width=w, color=COL[alg],
                  edgecolor="white", lw=0.5, label=alg,
                  yerr=[s["median"] - s["q1"], s["q3"] - s["median"]],
                  error_kw=dict(lw=0.5, ecolor=ACCENT, capsize=1.0))
        a.set_xticks(x)
        a.set_xticklabels([MLAB[m] for m in MODES], fontsize=6.5, rotation=20, ha="right")
        a.set_title(c, fontsize=8)
        a.set_yscale("log")
        if i == 0:
            a.set_ylabel("offline error $E_o$ (median, IQR)")
            a.legend(ncol=2, fontsize=6)
    save(fig, outdir, "temporal_modes_by_optimizer")


def fig_effects(eff, outdir):
    names = ["E1_prevbest_minus_cold", "E3_persist_minus_prevbest"]
    titles = ["(a) previous-best seeded vs cold restart",
              "(b) persistence vs previous-best seeded"]
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.8), layout="constrained")
    for k, (nm, ttl) in enumerate(zip(names, titles)):
        a = ax[k]
        rows = [(c, alg) for c in CASES for alg in ALGS]
        y = np.arange(len(rows))[::-1]
        for yy, (c, alg) in zip(y, rows):
            r = eff[(eff.case == c) & (eff.alg == alg) & (eff.effect == nm)].iloc[0]
            sig = r.significant_holm
            a.plot([r.log_ratio_ci_lo, r.log_ratio_ci_hi], [yy, yy], color=COL[alg],
                   lw=1.3 if sig else 0.8, alpha=1.0 if sig else 0.55)
            a.plot(r.median_log_ratio, yy, MK[alg], ms=4.2 if sig else 3.2,
                   mfc=COL[alg] if sig else "white", mec=COL[alg], mew=1.0)
        a.axvline(0, color=ACCENT, ls="--", lw=0.8)
        a.set_yticks(y)
        a.set_yticklabels([f"{c} {alg}" for c, alg in rows], fontsize=5.8)
        a.set_xlabel("median paired log ratio\n(negative: the first mode is better)")
        a.set_title(ttl, loc="left", fontsize=7.5)
    save(fig, outdir, "temporal_transfer_effect_sizes")


def fig_cross(cr, outdir):
    names = ["E1_prevbest_minus_cold", "E2_persist_minus_cold",
             "E3_persist_minus_prevbest"]
    short = {"E1_prevbest_minus_cold": "E1 prev-best vs cold",
             "E2_persist_minus_cold": "E2 persist vs cold",
             "E3_persist_minus_prevbest": "E3 persist vs prev-best"}
    fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.6), layout="constrained")
    for k, nm in enumerate(names):
        a = ax[k]
        rows = [(c, o) for c in CASES for o in ["GWO", "PSO", "WOA"]]
        y = np.arange(len(rows))[::-1]
        for yy, (c, o) in zip(y, rows):
            r = cr[(cr.case == c) & (cr.effect == nm)
                   & (cr.comparison == f"SGO vs {o}")].iloc[0]
            sig = r.significant_holm
            a.plot([r.diff_ci_lo, r.diff_ci_hi], [yy, yy], color=COL[o],
                   lw=1.3 if sig else 0.8, alpha=1.0 if sig else 0.55)
            a.plot(r.median_difference, yy, MK[o], ms=4.2 if sig else 3.2,
                   mfc=COL[o] if sig else "white", mec=COL[o], mew=1.0)
        a.axvline(0, color=ACCENT, ls="--", lw=0.8)
        a.set_yticks(y)
        a.set_yticklabels([f"{c} vs {o}" for c, o in rows], fontsize=5.6)
        a.set_xlabel("SGO effect $-$ comparator effect\n(log-ratio units)")
        a.set_title(short[nm], loc="left", fontsize=7.5)
    save(fig, outdir, "temporal_effect_comparison")


def fig_behaviour(diag, outdir):
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), layout="constrained")
    metrics = [("mean_population_diversity", "normalised diversity", True),
               ("recovery_frac_90", "fraction of environment to 90% recovery", False),
               ("late_half_fraction", "late-half share of improvement", False)]
    for k, (m, lab, logy) in enumerate(metrics):
        a = ax[k]
        x = np.arange(len(MODES))
        w = 0.2
        for j, alg in enumerate(ALGS):
            v = [diag[(diag.alg == alg) & (diag["mode"] == md)][m].median()
                 for md in MODES]
            a.bar(x + (j - 1.5) * w, v, width=w, color=COL[alg], edgecolor="white",
                  lw=0.5, label=alg)
        a.set_xticks(x)
        a.set_xticklabels([MLAB[md] for md in MODES], fontsize=6.5)
        a.set_ylabel(lab, fontsize=7)
        if logy:
            a.set_yscale("log")
        if k == 0:
            a.legend(ncol=2, fontsize=6)
        a.set_title(f"({'abc'[k]})", loc="left", fontsize=7.5)
    save(fig, outdir, "temporal_behavior_profiles")


if __name__ == "__main__":
    main()
