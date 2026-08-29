#!/usr/bin/env python3
"""Phase-2 closure: ANALYSIS-ONLY regeneration of the tail-convention statistics.

Reads committed CSVs only. Runs no simulation, calls no optimizer, builds no MPC
problem, and writes no new raw result. Its purpose is to make every number and
the tail-convention diagnostic figure reproducible from
a committed code path, since the paired statistics and the figure were originally
produced interactively.

Inputs (all committed):
    data/tail_reference_raw.csv
    data/tail_deterministic_closed_loop.csv
    data/tail_stochastic_raw.csv
    data/tail_stochastic_summary.csv

Outputs:
    data/tail_paired_effects.csv          paired hold_last - zero effects + bootstrap CIs
    data/tail_ordering_comparison.csv     ordering under each convention, both metric families
    data/tables/tail_convention_sensitivity.tex
    data/figures/tail_convention_sensitivity.eps   (regenerated, identical inputs)

Bootstrap seed is fixed at BOOT_SEED so the CIs are deterministic.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ROOT), "data")
ALGS = ["SGO", "GWO", "PSO", "WOA"]
BOOT_SEED = 20260817
N_BOOT = 4000


def boot_ci_mean(v, rng, n=N_BOOT, alpha=0.05):
    v = np.asarray(v, float)
    idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(BOOT_SEED)

    ref = pd.read_csv(os.path.join(OUT, "tail_reference_raw.csv"))
    det = pd.read_csv(os.path.join(OUT, "tail_deterministic_closed_loop.csv"))
    raw = pd.read_csv(os.path.join(OUT, "tail_stochastic_raw.csv"))
    summ = pd.read_csv(os.path.join(OUT, "tail_stochastic_summary.csv"))

    # ---- solver agreement, recomputed from the raw file (not remembered) ----
    agree_max = float(max(ref.lbfgsb_rel_zero.max(), ref.lbfgsb_rel_hold.max()))
    per_sys = ref.groupby("system")[["lbfgsb_rel_zero", "lbfgsb_rel_hold"]].max()
    print(f"OSQP vs multistart L-BFGS-B, max relative disagreement: {agree_max:.6e}")
    print(per_sys.to_string())

    # ---- paired effects (same seed under both conventions) -----------------
    rows = []
    for (sy, alg), g in raw.groupby(["system", "alg"]):
        z = g[g["tail"] == "zero"].sort_values("trial")
        h = g[g["tail"] == "hold_last"].sort_values("trial")
        assert (z.seed.values == h.seed.values).all(), f"seeds not paired for {sy}/{alg}"
        rel = (h.nrmse.values - z.nrmse.values) / z.nrmse.values * 100.0
        lo, hi = boot_ci_mean(rel, rng)
        rows.append(dict(system=sy, alg=alg, n_pairs=len(z),
                         d_nrmse_rel_pct=float(rel.mean()),
                         ci95_lo=lo, ci95_hi=hi,
                         d_mean_gap=float(h.mean_norm_gap.mean() - z.mean_norm_gap.mean()),
                         d_first_input_error=float(h.mean_first_input_error.mean()
                                                   - z.mean_first_input_error.mean())))
    paired = pd.DataFrame(rows).sort_values(["system", "alg"])
    paired.to_csv(os.path.join(OUT, "tail_paired_effects.csv"), index=False)

    # ---- ordering under each convention, both metric families --------------
    orows = []
    for sy in sorted(summ.system.unique()):
        for label, col in [("optimization_accuracy_mean_gap", "mean_gap"),
                           ("closed_loop_nrmse", "nrmse_mean")]:
            o = {t: list(summ[(summ.system == sy) & (summ["tail"] == t)]
                         .sort_values(col).alg) for t in ["zero", "hold_last"]}
            sgo = {t: int(summ[(summ.system == sy) & (summ["tail"] == t)
                               & (summ.alg == "SGO")]
                          ["rank_gap" if col == "mean_gap" else "rank_nrmse"].iloc[0])
                   for t in ["zero", "hold_last"]}
            orows.append(dict(system=sy, metric_family=label,
                              ordering_zero=",".join(o["zero"]),
                              ordering_hold_last=",".join(o["hold_last"]),
                              ordering_unchanged=bool(o["zero"] == o["hold_last"]),
                              sgo_rank_zero=sgo["zero"], sgo_rank_hold_last=sgo["hold_last"],
                              sgo_rank_unchanged=bool(sgo["zero"] == sgo["hold_last"])))
    order = pd.DataFrame(orows)
    order.to_csv(os.path.join(OUT, "tail_ordering_comparison.csv"), index=False)

    # ---- LaTeX table --------------------------------------------------------
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Zero-tail versus hold-last prediction-tail sensitivity. Upper block: "
         r"deterministic effect on all five systems. Lower block: paired-seed stochastic effect on "
         r"the two tested systems, with bootstrap 95\% confidence intervals. Nothing was retuned for "
         r"hold-last, so this measures sensitivity, not the relative merit of the two conventions.}",
         r"\label{tab:tail_convention_sensitivity}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"\multicolumn{6}{@{}l}{\textit{Deterministic (all five systems)}} \\",
         r"System & & nRMSE zero-tail & nRMSE hold-last & Rel.\ change (\%) & Mean $|\Delta u_0|$ \\",
         r"\midrule"]
    for _, r in det.iterrows():
        L.append(f"{r.system} & & {r.nrmse_zero:.6g} & {r.nrmse_hold:.6g} & "
                 f"{100*r.nrmse_rel_diff:+.1f} & {r.mean_first_control_diff:.3e} \\\\")
    L += [r"\midrule",
          r"\multicolumn{6}{@{}l}{\textit{Stochastic, paired seeds (flight, UAV)}} \\",
          r"System & Opt. & $\Delta$nRMSE (\%) & 95\% CI & $\Delta$ mean gap & "
          r"$\Delta$ first-input err. \\", r"\midrule"]
    for _, r in paired.iterrows():
        L.append(f"{r.system} & {r.alg} & {r.d_nrmse_rel_pct:+.1f} & "
                 f"[{r.ci95_lo:+.1f}, {r.ci95_hi:+.1f}] & {r.d_mean_gap:+.2e} & "
                 f"{r.d_first_input_error:+.2e} \\\\")
    L += [r"\midrule",
          r"\multicolumn{6}{@{}l}{\textit{SGO rank under each convention}} \\",
          r"System & Metric family & zero-tail & hold-last & \multicolumn{2}{c}{Unchanged?} \\",
          r"\midrule"]
    for _, r in order.iterrows():
        fam = "optimization accuracy" if "gap" in r.metric_family else "closed-loop nRMSE"
        L.append(f"{r.system} & {fam} & {r.sgo_rank_zero} & {r.sgo_rank_hold_last} & "
                 f"\\multicolumn{{2}}{{c}}{{{'yes' if r.sgo_rank_unchanged else 'no'}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "tail_convention_sensitivity.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    # ---- diagnostic figure (regenerated from the same committed inputs) -----
    if not args.no_figure:
        plt.rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].bar(det.system, det.nrmse_rel_diff * 100, color="#607d8b")
        ax[0].set_yscale("log"); ax[0].axhline(1, color="crimson", ls="--", lw=.8)
        ax[0].set_ylabel("|nRMSE change| (%)"); ax[0].set_title("(a) deterministic closed loop")
        ax[0].tick_params(axis="x", labelrotation=45)
        data, labs = [], []
        for (sy, alg), g in raw.groupby(["system", "alg"]):
            z = g[g["tail"] == "zero"].sort_values("trial")
            h = g[g["tail"] == "hold_last"].sort_values("trial")
            data.append((h.nrmse.values - z.nrmse.values) / z.nrmse.values * 100)
            labs.append(f"{sy[:4]}/{alg}")
        ax[1].boxplot(data, tick_labels=labs, showfliers=False)
        ax[1].axhline(0, color="k", lw=1); ax[1].set_ylabel("paired nRMSE change (%)")
        ax[1].set_title("(b) stochastic, paired seeds")
        ax[1].tick_params(axis="x", labelrotation=90, labelsize=7)
        lbl = [f"{r.system[:4]}\n{'gap' if 'gap' in r.metric_family else 'nRMSE'}"
               for _, r in order.iterrows()]
        x = np.arange(len(lbl))
        ax[2].bar(x - 0.2, order.sgo_rank_zero, 0.4, label="zero-tail")
        ax[2].bar(x + 0.2, order.sgo_rank_hold_last, 0.4, label="hold-last")
        ax[2].set_xticks(x); ax[2].set_xticklabels(lbl, fontsize=8)
        ax[2].set_ylabel("SGO rank (1=best)"); ax[2].set_ylim(0, 4.6)
        ax[2].set_title("(c) SGO rank by metric family"); ax[2].legend(fontsize=7)
        for a in ax:
            a.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", "tail_convention_sensitivity.eps"),
                    format="eps", bbox_inches="tight")
        plt.close(fig)

    pd.set_option("display.width", 220)
    print("\n" + paired.to_string(index=False))
    print("\n" + order.to_string(index=False))
    print("\nwrote tail_paired_effects.csv, tail_ordering_comparison.csv, "
          "tables/tail_convention_sensitivity.tex, figures/tail_convention_sensitivity.eps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
