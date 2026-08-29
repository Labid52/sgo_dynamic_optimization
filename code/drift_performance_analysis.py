#!/usr/bin/env python3
"""Phase-2B steps 6 and 11-12: does optimizer behaviour depend on drift severity?

This is the central Phase-2B scientific question. It is answered two ways, both
decided before the frozen sweep was run:

  (a) per-stratum statistics on the drift strata frozen in
      drift_strata_definition.json (median, IQR, convergence rate, bootstrap CIs);
  (b) a CONTINUOUS Spearman association between the drift severity of an instance
      (warm_gap_norm) and the optimizer's error on that instance.

(b) exists because the Severe stratum is UAV-dominated (16 of 19 transitions) and
HVAC contributes no non-benign transition at all. A conclusion resting only on
stratum means would partly be a statement about which system supplies the
instances; the continuous measure is computed per system, so it cannot be.

Correlation is not causation and is not claimed to be: the reported quantity is
association between an instance property fixed in advance and an optimizer
outcome measured afterwards.

Also produces the warm-start mechanism analysis (reviewer R1.7): start-gap
distributions, trivial-start fractions, incumbent-return rates and realized
improvement, per system and start condition.

Outputs:
    data/drift_performance_association.csv
    data/warm_start_mechanism.csv
    data/tables/frozen_sweep_by_drift.tex
    data/tables/drift_performance_association.tex
    data/figures/frozen_gap_vs_drift_diagnostic.eps
    data/figures/convergence_vs_drift_diagnostic.eps
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(PROJECT, "data")
sys.path.insert(0, ROOT)
import phase2_protocol as P2                                           # noqa: E402

STRATA_ORDER = ["Benign", "Moderate", "High", "Severe"]
BOOT = 4000
RNG = np.random.default_rng(20260816)


def boot_ci_median(v, n=BOOT, alpha=0.05):
    v = np.asarray(v, float)
    if len(v) == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, len(v), (n, len(v)))
    meds = np.median(v[idx], axis=1)
    return float(np.percentile(meds, 100 * alpha / 2)), float(np.percentile(meds, 100 * (1 - alpha / 2)))


def boot_ci_mean(v, n=BOOT, alpha=0.05):
    v = np.asarray(v, float)
    if len(v) == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, len(v), (n, len(v)))
    ms = v[idx].mean(axis=1)
    return float(np.percentile(ms, 100 * alpha / 2)), float(np.percentile(ms, 100 * (1 - alpha / 2)))


def main():
    t0 = time.perf_counter()
    df = pd.read_csv(os.path.join(OUT, "frozen_sweep_v2_raw.csv"))

    # ---------------- per-stratum statistics ----------------
    rows = []
    for (system, alg, start, stratum), g in df.groupby(["system", "alg", "start", "stratum"]):
        gm_lo, gm_hi = boot_ci_median(g.final_gap_norm.to_numpy())
        cr_lo, cr_hi = boot_ci_mean(g.converged_1e4.to_numpy(dtype=float))
        rows.append(dict(
            system=system, alg=alg, start=start, stratum=stratum,
            n=len(g), instances=int(g.k.nunique()),
            start_gap_median=float(g.start_gap_norm.median()),
            final_gap_median=float(g.final_gap_norm.median()),
            final_gap_q1=float(np.percentile(g.final_gap_norm, 25)),
            final_gap_q3=float(np.percentile(g.final_gap_norm, 75)),
            final_gap_median_lo=gm_lo, final_gap_median_hi=gm_hi,
            conv_rate=float(g.converged_1e4.mean()),
            conv_rate_lo=cr_lo, conv_rate_hi=cr_hi,
            trivial_start_rate=float(g.trivial_start.mean()),
            incumbent_return_rate=float(g.returned_incumbent.mean()),
            improvement_median=float(g.improvement_fraction.median(skipna=True)),
            first_ctrl_err_median=float(g.first_control_error.median()),
            time_ms_median=float(g.time_ms.median())))
    strat = pd.DataFrame(rows)
    strat["stratum"] = pd.Categorical(strat.stratum, STRATA_ORDER, ordered=True)
    strat = strat.sort_values(["system", "alg", "start", "stratum"])
    strat.to_csv(os.path.join(OUT, "frozen_sweep_by_drift.csv"), index=False)

    # ---------------- continuous association ----------------
    assoc = []
    for (system, alg, start), g in df.groupby(["system", "alg", "start"]):
        # aggregate to one value per instance so trials do not inflate n
        per_inst = g.groupby("k").agg(
            drift=("warm_gap_norm", "first"),
            gap=("final_gap_norm", "median"),
            conv=("converged_1e4", "mean"),
            ctrl=("first_control_error", "median"),
            inc=("returned_incumbent", "mean"),
            tms=("time_ms", "median")).reset_index()
        if per_inst.drift.nunique() < 3:
            continue
        out = {"system": system, "alg": alg, "start": start,
               "n_instances": int(len(per_inst))}
        for label, col in [("gap", "gap"), ("conv", "conv"), ("ctrl", "ctrl"),
                           ("incumbent", "inc"), ("time", "tms")]:
            r, p = spearmanr(per_inst.drift, per_inst[col])
            out[f"spearman_drift_vs_{label}"] = float(r)
            out[f"p_drift_vs_{label}"] = float(p)
        assoc.append(out)
    assoc = pd.DataFrame(assoc)
    assoc.to_csv(os.path.join(OUT, "drift_performance_association.csv"), index=False)

    # ---------------- warm-start mechanism ----------------
    wsm = []
    for (system, alg, start), g in df.groupby(["system", "alg", "start"]):
        wsm.append(dict(
            system=system, alg=alg, start=start, n=len(g),
            start_gap_median=float(g.start_gap_norm.median()),
            start_gap_p90=float(np.percentile(g.start_gap_norm, 90)),
            frac_start_within_1e4=float(g.trivial_start.mean()),
            incumbent_return_rate=float(g.returned_incumbent.mean()),
            improvement_median=float(g.improvement_fraction.median(skipna=True)),
            final_gap_median=float(g.final_gap_norm.median()),
            conv_rate=float(g.converged_1e4.mean())))
    wsm = pd.DataFrame(wsm)
    wsm.to_csv(os.path.join(OUT, "warm_start_mechanism.csv"), index=False)

    # ---------------- tables ----------------
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Frozen-subproblem accuracy by drift stratum, cold start. Median "
             r"normalized optimality gap (IQR) and strict convergence rate at the $10^{-4}$ "
             r"criterion. Strata were fixed from drift metrics alone before any optimizer "
             r"trial. HVAC contributes only Benign instances and the Severe stratum is "
             r"UAV-dominated, so per-system reading is required.}",
             r"\label{tab:frozen_sweep_by_drift}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"Stratum & Optimizer & Median gap & IQR & Conv.\ rate & Instances \\",
             r"\midrule"]
    pooled = (df[df.start == "cold"].groupby(["stratum", "alg"])
              .agg(med=("final_gap_norm", "median"),
                   q1=("final_gap_norm", lambda s: float(np.percentile(s, 25))),
                   q3=("final_gap_norm", lambda s: float(np.percentile(s, 75))),
                   cr=("converged_1e4", "mean"),
                   ni=("k", "nunique")).reset_index())
    for st in STRATA_ORDER:
        for _, r in pooled[pooled.stratum == st].iterrows():
            lines.append(f"{st} & {r.alg} & {r.med:.3e} & "
                         f"[{r.q1:.2e}, {r.q3:.2e}] & {100*r.cr:.1f}\\% & {int(r.ni)} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "frozen_sweep_by_drift.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Spearman association between instance drift severity (warm-start "
             r"suboptimality, fixed in advance) and optimizer outcome, computed per system so "
             r"the result cannot be an artefact of which system supplies the instances. "
             r"Cold start.}",
             r"\label{tab:drift_performance_association}",
             r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"System & Optimizer & $\rho$(drift, gap) & $p$ & $\rho$(drift, conv.) & $p$ \\",
             r"\midrule"]
    for _, r in assoc[assoc.start == "cold"].iterrows():
        lines.append(f"{r.system} & {r.alg} & {r.spearman_drift_vs_gap:+.3f} & "
                     f"{r.p_drift_vs_gap:.2g} & {r.spearman_drift_vs_conv:+.3f} & "
                     f"{r.p_drift_vs_conv:.2g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "drift_performance_association.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 9})
    cold = df[df.start == "cold"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
    for ax, key in zip(axes, P2.SYSTEMS):
        g = cold[cold.system == key]
        for alg in P2.ALGS:
            gg = g[g.alg == alg].groupby("k").agg(
                d=("warm_gap_norm", "first"), gp=("final_gap_norm", "median")).reset_index()
            ax.loglog(np.maximum(gg.d, 1e-18), np.maximum(gg.gp, 1e-18), ".", ms=4, label=alg)
        ax.axhline(1e-4, color="crimson", ls="--", lw=.8)
        ax.axvline(1e-4, color="grey", ls=":", lw=.8)
        ax.set_title(key); ax.set_xlabel("instance drift (warm-gap)")
        if key == P2.SYSTEMS[0]:
            ax.set_ylabel("median final gap"); ax.legend(fontsize=6)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "frozen_gap_vs_drift_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, start in zip(axes, ["cold", "reference_warm", "optimizer_warm"]):
        sub = df[df.start == start]
        w = 0.2
        xs = np.arange(len(STRATA_ORDER))
        for j, alg in enumerate(P2.ALGS):
            vals = [float(sub[(sub.stratum == st) & (sub.alg == alg)].converged_1e4.mean())
                    if len(sub[(sub.stratum == st) & (sub.alg == alg)]) else np.nan
                    for st in STRATA_ORDER]
            ax.bar(xs + (j - 1.5) * w, vals, w, label=alg)
        ax.set_xticks(xs); ax.set_xticklabels(STRATA_ORDER)
        ax.set_title(start.replace("_", " ")); ax.grid(alpha=.3, axis="y")
        if start == "cold":
            ax.set_ylabel("convergence rate at 1e-4"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "convergence_vs_drift_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    meta = {"rows": int(len(df)), "instances": int(df.k.nunique()),
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "drift_performance_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.set_option("display.width", 250)
    print("=== pooled cold-start gap and convergence by stratum ===")
    print(pooled.to_string(index=False))
    print("\n=== Spearman drift vs gap (cold) ===")
    print(assoc[assoc.start == "cold"][
        ["system", "alg", "n_instances", "spearman_drift_vs_gap", "p_drift_vs_gap",
         "spearman_drift_vs_conv", "p_drift_vs_conv"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
