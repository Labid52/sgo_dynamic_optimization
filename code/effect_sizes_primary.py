#!/usr/bin/env python3
"""Phase-2B step 1: effect sizes and confidence intervals for the Phase-2A
primary results (reviewer R2.11).

Phase 2A produced numerical rankings. This script decides which of those
orderings are inferentially supported, because several of them differ only in
the 4th-6th significant digit and the manuscript must not present them as
findings until that is settled.

For every (system, regime) and every unordered optimizer pair:

  * two-sided Mann-Whitney U  (independent samples: the Phase-2A seed scheme
    gives each optimizer its own seeds, so the runs are NOT paired)
  * Holm-Bonferroni adjustment, FAMILY = the 6 pairwise comparisons within one
    (system, regime). Rationale: each (system, regime) is a separate experiment
    with its own conclusion; correcting across systems would be over-conservative
    and correcting within an optimizer across systems would mix experiments.
  * Cliff's delta with a bootstrap 95% CI
  * Hodges-Lehmann location shift with a bootstrap 95% CI
  * absolute mean / median difference and relative percentage difference

Practical-difference flags are reported at 0.5%, 1% and 2% and are labelled
DESCRIPTIVE. No engineering-significance threshold is asserted: none of these
benchmarks comes with an independently justified tolerance on normalized RMSE.

Evidence classification (deterministic decision tree, fixed before running):
    adjusted p >= 0.05                       -> numerically ordered but
                                                statistically uncertain
    adjusted p <  0.05 and rel < 0.5%        -> practically negligible under
                                                descriptive thresholds
    adjusted p <  0.05 and rel >= 2% and
        |Cliff's delta| >= 0.474 (large)     -> clear separation
    otherwise                                -> small but statistically
                                                supported difference

Outputs:
    data/effect_sizes_primary.csv
    data/effect_sizes_primary_summary.csv
    data/tables/effect_sizes_primary.tex
    data/figures/effect_size_forest_diagnostic.eps
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(PROJECT, "data")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
sys.path.insert(0, ROOT)
import phase2_protocol as P2                                          # noqa: E402

BOOT = 10000
RNG_SEED = 20260816
ALPHA = 0.05
DELTA_LARGE = 0.474          # Romano et al. thresholds: .147 small, .33 medium, .474 large
DESCRIPTIVE_THRESHOLDS = [0.5, 1.0, 2.0]


def cliffs_delta(x, y):
    """P(x>y) - P(x<y), computed in O(n log n) via ranks."""
    nx, ny = len(x), len(y)
    r = rankdata(np.concatenate([x, y]))
    rx = r[:nx].sum()
    # Mann-Whitney U for x, then delta = 2U/(nx*ny) - 1
    u = rx - nx * (nx + 1) / 2.0
    return 2.0 * u / (nx * ny) - 1.0


def hodges_lehmann(x, y):
    """Median of all pairwise differences x_i - y_j."""
    return float(np.median(np.subtract.outer(x, y).ravel()))


def boot_ci(x, y, stat, n_boot=BOOT, seed=RNG_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    nx, ny = len(x), len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        xb = x[rng.integers(0, nx, nx)]
        yb = y[rng.integers(0, ny, ny)]
        vals[b] = stat(xb, yb)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, monotone step-down."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * float(p))
        running = max(running, adj)
        out[k] = min(1.0, running)
    return out


def classify(adj_p, rel_pct, delta):
    if not np.isfinite(adj_p) or adj_p >= ALPHA:
        return "numerically ordered but statistically uncertain"
    if rel_pct < DESCRIPTIVE_THRESHOLDS[0]:
        return "practically negligible under descriptive thresholds"
    if rel_pct >= DESCRIPTIVE_THRESHOLDS[2] and abs(delta) >= DELTA_LARGE:
        return "clear separation"
    return "small but statistically supported difference"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()
    t0 = time.perf_counter()

    raw = pd.read_csv(os.path.join(OUT, "primary_closed_loop_raw.csv"))
    rows = []
    for (system, regime), g in raw.groupby(["system", "regime"]):
        samples = {a: g[g.alg == a].sort_values("trial").nrmse.to_numpy(dtype=float)
                   for a in P2.ALGS}
        pvals, recs = {}, []
        for i, a in enumerate(P2.ALGS):
            for b in P2.ALGS[i + 1:]:
                xa, xb = samples[a], samples[b]
                try:
                    p = float(mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
                except ValueError:              # identical constant samples
                    p = 1.0
                d = cliffs_delta(xa, xb)
                d_lo, d_hi = boot_ci(xa, xb, cliffs_delta, args.boot)
                hl = hodges_lehmann(xa, xb)
                hl_lo, hl_hi = boot_ci(xa, xb, hodges_lehmann, args.boot)
                mean_diff = float(np.mean(xa) - np.mean(xb))
                med_diff = float(np.median(xa) - np.median(xb))
                base = max(abs(float(np.mean(xb))), 1e-300)
                rel_pct = abs(mean_diff) / base * 100.0
                pvals[(a, b)] = p
                recs.append(dict(
                    system=system, regime=regime, alg_a=a, alg_b=b,
                    n_a=len(xa), n_b=len(xb),
                    mean_a=float(np.mean(xa)), mean_b=float(np.mean(xb)),
                    median_a=float(np.median(xa)), median_b=float(np.median(xb)),
                    std_a=float(np.std(xa, ddof=1)), std_b=float(np.std(xb, ddof=1)),
                    mean_diff_abs=mean_diff, median_diff_abs=med_diff,
                    relative_diff_pct=rel_pct,
                    p_raw=p, cliffs_delta=d, cliffs_delta_lo=d_lo, cliffs_delta_hi=d_hi,
                    delta_ci_excludes_zero=bool(d_lo > 0 or d_hi < 0),
                    hodges_lehmann=hl, hl_lo=hl_lo, hl_hi=hl_hi,
                    hl_ci_excludes_zero=bool(hl_lo > 0 or hl_hi < 0),
                    better=(a if mean_diff < 0 else b)))
        adj = holm(pvals)
        for r in recs:
            r["p_holm"] = float(adj[(r["alg_a"], r["alg_b"])])
            r["significant_holm"] = bool(r["p_holm"] < ALPHA)
            for th in DESCRIPTIVE_THRESHOLDS:
                r[f"exceeds_{th}pct_descriptive"] = bool(r["relative_diff_pct"] >= th)
            r["evidence"] = classify(r["p_holm"], r["relative_diff_pct"], r["cliffs_delta"])
            r["holm_family"] = f"6 pairwise comparisons within {system}/{regime}"
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "effect_sizes_primary.csv"), index=False)

    # SGO-centred summary
    sgo = df[(df.alg_a == "SGO") | (df.alg_b == "SGO")].copy()
    sgo["other"] = np.where(sgo.alg_a == "SGO", sgo.alg_b, sgo.alg_a)
    # orient every comparison as SGO minus other
    flip = sgo.alg_b == "SGO"
    sgo["sgo_minus_other_mean"] = np.where(flip, -sgo.mean_diff_abs, sgo.mean_diff_abs)
    sgo["delta_sgo_vs_other"] = np.where(flip, -sgo.cliffs_delta, sgo.cliffs_delta)
    sgo["delta_lo"] = np.where(flip, -sgo.cliffs_delta_hi, sgo.cliffs_delta_lo)
    sgo["delta_hi"] = np.where(flip, -sgo.cliffs_delta_lo, sgo.cliffs_delta_hi)
    sgo["sgo_lower"] = sgo.sgo_minus_other_mean < 0
    summ = sgo[["system", "regime", "other", "sgo_minus_other_mean", "relative_diff_pct",
                "delta_sgo_vs_other", "delta_lo", "delta_hi", "p_holm",
                "significant_holm", "sgo_lower", "evidence"]]
    summ.to_csv(os.path.join(OUT, "effect_sizes_primary_summary.csv"), index=False)

    # LaTeX table: SGO-centred
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{SGO versus each reference optimizer under the fixed-MPC primary "
             r"experiment. Negative mean difference means SGO has the lower closed-loop "
             r"nRMSE. Cliff's $\delta$ is oriented SGO$-$other with a bootstrap 95\% CI "
             r"(10{,}000 resamples); $p$ is Holm--Bonferroni adjusted within the six "
             r"pairwise comparisons of each system and regime. Practical-difference "
             r"thresholds are descriptive only.}",
             r"\label{tab:effect_sizes_primary}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}llcrrrcl@{}}", r"\toprule",
             r"System & Regime & vs & $\Delta$mean & rel.\ \% & Cliff's $\delta$ "
             r"[95\% CI] & $p_{\mathrm{Holm}}$ & Evidence \\", r"\midrule"]
    for system in P2.SYSTEMS:
        for regime in ["A_tuned_NP", "B_common_NP"]:
            gg = summ[(summ.system == system) & (summ.regime == regime)]
            for _, r in gg.iterrows():
                lines.append(
                    f"{system} & {'A' if regime.startswith('A') else 'B'} & {r.other} & "
                    f"${r.sgo_minus_other_mean:+.2e}$ & {r.relative_diff_pct:.3f} & "
                    f"${r.delta_sgo_vs_other:+.2f}$ [{r.delta_lo:+.2f}, {r.delta_hi:+.2f}] & "
                    f"{r.p_holm:.3g} & {r.evidence.replace('_',' ')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "effect_sizes_primary.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Diagnostic forest plot of Cliff's delta (SGO vs other)
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True)
    for ax, regime in zip(axes, ["A_tuned_NP", "B_common_NP"]):
        gg = summ[summ.regime == regime].reset_index(drop=True)
        ypos = np.arange(len(gg))
        ax.errorbar(gg.delta_sgo_vs_other, ypos,
                    xerr=[gg.delta_sgo_vs_other - gg.delta_lo,
                          gg.delta_hi - gg.delta_sgo_vs_other],
                    fmt="o", ms=4, capsize=2, lw=1,
                    color="#1f77b4")
        ax.axvline(0, color="k", lw=1)
        for v in (-DELTA_LARGE, DELTA_LARGE):
            ax.axvline(v, color="grey", ls=":", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{r.system}/{r.other}" for _, r in gg.iterrows()], fontsize=7)
        ax.set_xlabel(r"Cliff's $\delta$ (SGO $-$ other); negative = SGO lower nRMSE")
        ax.set_title("Regime A (tuned NP)" if regime.startswith("A") else
                     "Regime B (common NP=20)")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "effect_size_forest_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    counts = df.evidence.value_counts().to_dict()
    meta = {"comparisons": int(len(df)), "bootstrap_resamples": args.boot,
            "alpha": ALPHA, "holm_family": "6 pairwise comparisons per (system, regime)",
            "descriptive_thresholds_pct": DESCRIPTIVE_THRESHOLDS,
            "evidence_counts": counts,
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "effect_sizes_primary_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
