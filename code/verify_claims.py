#!/usr/bin/env python3
"""Independent recomputation of the headline reported numbers from the raw data.

This script deliberately does NOT reuse the analysis modules. It reads the raw
per-run CSVs under data/ and recomputes each reported quantity from scratch, so
that a reader can check the paper without trusting the table generators.

Every temporal-comparator CSV is read with float_precision="round_trip": the
default pandas parser reproduces a small subset of these rows one ULP off, which
is enough to flip an exact equality check.

    python code/verify_claims.py

Exit code 0 if every check passes.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(ROOT), "data")
RT = dict(float_precision="round_trip")
CASES = [f"F{i}" for i in range(1, 13)]

_results = []


def check(name, got, want, tol=None, note=""):
    if tol is None:
        ok = got == want
    else:
        ok = abs(float(got) - float(want)) <= tol
    _results.append(ok)
    flag = "PASS" if ok else "FAIL"
    extra = f"  [{note}]" if note else ""
    print(f"  {flag}  {name}: got {got!r}, expected {want!r}{extra}")
    return ok


def head(title):
    print(f"\n--- {title} ---")


def d(*p):
    return os.path.join(DATA, *p)


# ---------------------------------------------------------------- GMPB primary
def gmpb_primary():
    head("GMPB primary protocol (population persistence)")
    raw = pd.read_csv(d("gmpb", "gmpb_primary_raw.csv"), **RT)
    check("raw rows", len(raw), 1488)
    check("runs per case/optimizer", sorted(set(raw.groupby(["case", "alg"]).size())), [31])
    check("duplicate runs", int(raw.duplicated(["case", "seed", "alg", "mode"]).sum()), 0)
    check("every run exactly on budget", bool((raw.realized_FE == raw.max_evals).all()), True)
    check("environments completed per run", sorted(raw.environments_completed.unique()), [100])

    med = raw.pivot_table(index="case", columns="alg",
                          values="offline_error", aggfunc="median").loc[CASES]
    rank = med.rank(axis=1)["SGO"]
    check("SGO ranks last (4/4)", int((rank == 4).sum()), 11)
    check("SGO ranks third on F10", int(rank.loc["F10"]), 3)
    lower = [(c, a) for c in CASES for a in ("GWO", "PSO", "WOA")
             if med.loc[c, "SGO"] < med.loc[c, a]]
    check("cells with SGO median below a control", lower, [("F10", "PSO")])

    st = pd.read_csv(d("gmpb", "gmpb_statistics.csv"), **RT)
    p = st[st.family == "primary"]
    check("pairwise comparisons", len(p), 36)
    check("significantly unfavourable to SGO after Holm",
          int(((~p.sgo_lower_offline_error) & p.significant_holm).sum()), 32)
    check("significantly favourable to SGO after Holm",
          int((p.sgo_lower_offline_error & p.significant_holm).sum()), 0)
    unres = sorted(zip(p[~p.significant_holm].case, p[~p.significant_holm].comparison))
    check("unresolved comparisons", unres,
          [("F10", "SGO vs PSO"), ("F12", "SGO vs PSO"),
           ("F7", "SGO vs WOA"), ("F8", "SGO vs WOA")])


# ------------------------------------------------- GMPB matched temporal modes
def gmpb_temporal():
    head("GMPB matched temporal-transfer controls")
    tp = d("gmpb_temporal_comparators")
    raw = pd.read_csv(os.path.join(tp, "gmpb_temporal_comparator_raw.csv"), **RT)
    check("raw rows (4 optimizers x 4 cases x 3 modes x 31 runs)", len(raw), 1488)
    check("cells", raw.groupby(["case", "alg", "mode"]).ngroups, 48)
    check("every run exactly on budget", bool((raw.realized_FE == raw.max_evals).all()), True)

    eff = pd.read_csv(os.path.join(tp, "gmpb_temporal_effects.csv"), **RT)
    e1 = eff[eff.effect.str.startswith("E1")]
    check("previous-best beats cold restart", int(e1.A_better.sum()), 16)
    check("  ... and all 16 significant after Holm", int(e1.significant_holm.sum()), 16)
    e3 = eff[eff.effect.str.startswith("E3")]
    check("persistence worse than previous-best", int((~e3.A_better).sum()), 15)
    exc = e3[e3.A_better]
    check("the single exception", sorted(zip(exc.case, exc.alg)), [("F8", "PSO")])

    med = raw.pivot_table(index=["case", "alg"], columns="mode",
                          values="offline_error", aggfunc="median")
    order = ((med.previous_best_seeded < med.cold_restart)
             & (med.cold_restart < med.population_persistence))
    check("complete ordering prev < cold < persist", int(order.sum()), 14)
    check("  ... violated only by", sorted(med[~order].index.tolist()),
          [("F8", "GWO"), ("F8", "PSO")])

    cls = pd.read_csv(os.path.join(tp, "gmpb_temporal_pattern_classification.csv"), **RT)
    vc = cls.classification.value_counts().to_dict()
    check("patterns classified GENERAL", vc.get("GENERAL", 0), 6)
    check("patterns classified ALGORITHM-DEPENDENT", vc.get("ALGORITHM-DEPENDENT", 0), 6)
    check("patterns SGO-amplified or SGO-distinctive",
          len(cls) - vc.get("GENERAL", 0) - vc.get("ALGORITHM-DEPENDENT", 0), 0)

    r = raw.pivot_table(index=["case", "mode"], columns="alg",
                        values="offline_error", aggfunc="median").rank(axis=1)["SGO"]
    check("SGO never ranks better than third in any case x mode", float(r.min()), 3.0)

    pen = e3.set_index(["case", "alg"]).percent_change
    for case, want in [("F10", dict(SGO=43.7, GWO=225.5, PSO=192.0, WOA=12.2)),
                       ("F12", dict(SGO=117.0, GWO=166.6, PSO=196.9, WOA=47.9))]:
        for alg, v in want.items():
            check(f"{case} persistence penalty vs previous-best, {alg} (%)",
                  round(float(pen.loc[(case, alg)]), 1), v, tol=0.05)

    bd = pd.read_csv(os.path.join(tp, "gmpb_temporal_behavioral_diagnostics.csv"), **RT)
    pp = bd[bd["mode"] == "population_persistence"].groupby("alg")
    for col, want, dec in [("mean_population_diversity",
                            dict(SGO=0.057, GWO=0.021, PSO=0.018, WOA=0.061), 3),
                           ("recovery_frac_90",
                            dict(SGO=0.090, GWO=0.323, PSO=0.330, WOA=0.093), 3),
                           ("late_half_fraction", dict(SGO=0.020, WOA=0.0027), 4)]:
        m = pp[col].median()
        for alg, v in want.items():
            check(f"pooled persistence {col}, {alg}", round(float(m[alg]), dec), v, tol=1e-9)


# ------------------------------------------------- GMPB SGO grouping diagnostic
def gmpb_grouping():
    head("SGO grouping diagnostic on GMPB")
    mp = d("gmpb_mechanism")
    base = pd.read_csv(os.path.join(mp, "gmpb_baseline_repro_raw.csv"), **RT)
    prim = pd.read_csv(d("gmpb", "gmpb_primary_raw.csv"), **RT)
    a = base[["case", "seed", "offline_error"]].sort_values(["case", "seed"]).reset_index(drop=True)
    b = (prim[prim.alg == "SGO"][["case", "seed", "offline_error"]]
         .sort_values(["case", "seed"]).reset_index(drop=True))
    check("random-grouping baseline reproduces the primary SGO runs exactly",
          float(np.abs(a.offline_error.values - b.offline_error.values).max()), 0.0)

    st = pd.read_csv(os.path.join(mp, "gmpb_mechanism_statistics.csv"), **RT)
    oe = st[st.family == "primary_offline_error"]
    check("cases tested", len(oe), 12)
    check("cases where fitness grouping lowers median offline error",
          int((oe.median_diff < 0).sum()), 12)
    check("all significant after Holm", bool(oe.significant_holm.all()), True)
    lo, hi = -oe.relative_median_diff.max() * 100, -oe.relative_median_diff.min() * 100
    check("relative median reduction, min (%)", round(float(lo), 1), 44.7, tol=0.05)
    check("relative median reduction, max (%)", round(float(hi), 1), 66.3, tol=0.05)
    check("Cliff's delta most positive", round(float(oe.cliffs_delta.max()), 2), -0.85, tol=5e-3)
    check("Cliff's delta most negative", float(oe.cliffs_delta.min()), -1.0, tol=1e-9)

    dv = st[st.quantity == "mean population diversity"]
    check("cases where diversity increases under fitness grouping",
          int((dv.median_diff > 0).sum()), 12)
    check("  ... all significant after Holm", bool(dv.significant_holm.all()), True)
    dvi = dv.set_index("case")
    for c, r0, r1 in [("F2", 0.083, 0.192), ("F8", 0.079, 0.201),
                      ("F10", 0.031, 0.141), ("F12", 0.062, 0.197)]:
        check(f"{c} median diversity, random", round(float(dvi.loc[c, "median_random"]), 3), r0, tol=1e-9,
              note="F10 is 0.031456" if c == "F10" else "")
        check(f"{c} median diversity, fitness", round(float(dvi.loc[c, "median_fitness"]), 3), r1, tol=1e-9)

    it = st[st.family == "interaction"].set_index(["case", "comparison"])
    persist = "fitness vs random within population_persistence"
    prev = "fitness vs random within previous_best_seeded"
    dod = "difference of differences (effect under persistence minus effect under previous-best)"
    for c, wp, wq in [("F2", -25.3, 4.3), ("F8", -20.7, 8.4),
                      ("F10", -120.9, 13.5), ("F12", -33.3, 2.6)]:
        check(f"{c} grouping effect under persistence",
              round(float(it.loc[(c, persist), "median_diff"]), 1), wp, tol=0.05)
        check(f"{c} grouping effect under previous-best seeding",
              round(float(it.loc[(c, prev), "median_diff"]), 1), wq, tol=0.05)
        check(f"{c} both within-mode effects significant after Holm",
              bool(it.loc[(c, persist), "significant_holm"] and it.loc[(c, prev), "significant_holm"]), True)
        lo_, hi_ = it.loc[(c, dod), ["median_diff_ci_lo", "median_diff_ci_hi"]]
        check(f"{c} difference-of-differences CI excludes zero", bool(lo_ * hi_ > 0), True)


# ------------------------------------------------------------------ MPC claims
def mpc():
    head("MPC objective drift and finite-horizon difficulty")
    dr = pd.read_csv(d("objective_drift_full_raw.csv"), **RT)
    check("consecutive transitions", len(dr), 2400)
    check("overall benign fraction (%)", round(float(dr.benign.mean()) * 100, 1), 92.9, tol=0.05)
    bs = dr.groupby("system").benign.mean() * 100
    check("lowest per-system benign fraction (%)", round(float(bs.min()), 1), 65.3, tol=0.05)
    check("highest per-system benign fraction (%)", round(float(bs.max()), 1), 100.0, tol=0.05)

    fz = pd.read_csv(d("frozen_sweep_v2_raw.csv"), **RT)
    check("frozen subproblems", fz.groupby(["system", "k"]).ngroups, 215)
    check("trials per optimizer and start condition", fz.trial.nunique(), 20)
    check("every frozen trial exactly on budget", bool(fz.fe_exact.all()), True)
    check("frozen budget", sorted(fz.requested_fe.unique()), [1000])
    cold = fz[fz.start == "cold"]
    cv = cold.groupby(["alg", "stratum"]).converged_1e4.mean() * 100
    for alg, ben, sev in [("SGO", 84.9, 28.2), ("GWO", 96.9, 52.7),
                          ("PSO", 98.7, 45.0), ("WOA", 91.2, 31.4)]:
        check(f"{alg} cold convergence, Benign (%)", round(float(cv[(alg, "Benign")]), 1), ben, tol=0.05)
        check(f"{alg} cold convergence, Severe (%)", round(float(cv[(alg, "Severe")]), 1), sev, tol=0.05)
    g = cold[cold.alg == "SGO"].groupby("stratum").final_gap_norm.median()
    check("SGO median normalized gap, Benign", f"{g['Benign']:.2e}", "3.89e-10")
    check("SGO median normalized gap, Severe", f"{g['Severe']:.2e}", "5.75e-03")

    head("MPC protocol, budget and reference checks")
    pr = pd.read_csv(d("primary_closed_loop_raw.csv"), **RT)
    check("closed-loop runs per system/regime/optimizer",
          sorted(set(pr.groupby(["system", "regime", "alg"]).size())), [20])
    check("every closed-loop run exactly on budget", bool(pr.nfe_exactly_on_budget.all()), True)
    check("objective calls per MPC step", sorted(pr.fe_budget.unique()), [1000])

    nc = pd.read_csv(d("reference_nc_sweep_summary.csv"), **RT)
    check("certified (system, Nc) cells", int(nc.certified.sum()), 22)
    check("tested (system, Nc) cells", len(nc), 25)
    bad = sorted(zip(nc[~nc.certified].system, nc[~nc.certified].Nc))
    check("uncertified cells", bad, [("uav", 3), ("uav", 5), ("uav", 8)])
    p1 = nc[nc.Nc == 1]
    check("all primary Nc=1 cells certified", bool(p1.certified.all()), True)
    check("max solver disagreement at Nc=1", f"{p1.max_solver_rel_disagreement.max():.2e}", "2.25e-10")

    ef = pd.read_csv(d("effect_sizes_primary.csv"), **RT)
    vc = ef.evidence.value_counts().to_dict()
    check("pairwise closed-loop comparisons", len(ef), 60)
    check("clear separation", vc.get("clear separation", 0), 9)
    check("  ... all on the UAV",
          sorted(ef[ef.evidence == "clear separation"].system.unique()), ["uav"])
    check("statistically uncertain",
          vc.get("numerically ordered but statistically uncertain", 0), 29)
    check("resolved but practically negligible",
          vc.get("practically negligible under descriptive thresholds", 0), 22)
    check("flight-control comparisons all uncertain",
          set(ef[ef.system == "flight"].evidence),
          {"numerically ordered but statistically uncertain"})

    sv = pd.read_csv(d("submitted_vs_revised_comparison.csv"), **RT)
    check("system/regime cells whose ranking changes", int(sv.ranking_changed.sum()), 9)
    check("  ... out of", len(sv), 10)

    head("MPC robustness analyses")
    bg = pd.read_csv(d("budget_sensitivity_v2_summary.csv"), **RT)
    s5 = bg[(bg.alg == "SGO") & (bg.budget_fe == 5000)].set_index(["system", "k"]).conv_rate
    check("SGO convergence at 5000 FE, both CSTR instances (%)",
          sorted(round(float(v) * 100, 1) for (sy, k), v in s5.items() if sy == "cstr"), [100.0, 100.0])
    check("SGO convergence at 5000 FE, two flight instances (%)",
          sorted(round(float(v) * 100, 1) for (sy, k), v in s5.items() if sy == "flight"), [6.7, 16.7])
    check("SGO convergence at 5000 FE, worst of the three UAV instances (%)",
          round(max(float(v) for (sy, k), v in s5.items() if sy == "uav") * 100, 1), 3.3)

    ds = pd.read_csv(d("dimension_sweep_association.csv"), **RT).set_index("alg")
    check("accepted (system, Nc) cells in the dimension sweep",
          sorted(ds.n_cells.unique()), [22])
    for alg, v in [("SGO", -0.69), ("GWO", -0.85), ("PSO", -0.82), ("WOA", -0.63)]:
        check(f"rho(dimension, convergence) for {alg}",
              round(float(ds.loc[alg, "rho_dim_vs_conv"]), 2), v, tol=5e-3)
    check("rho(cond(H), median gap) significant for any optimizer",
          bool((ds.p_cond_vs_gap < 0.05).any()), False)

    ws = pd.read_csv(d("warm_start_mechanism.csv"), **RT)
    u = ws[ws.system == "uav"].pivot(index="alg", columns="start", values="conv_rate") * 100
    for alg, ref, opt in [("SGO", 56.2, 8.8), ("PSO", 58.3, 2.7)]:
        check(f"UAV {alg} convergence, reference warm start (%)",
              round(float(u.loc[alg, "reference_warm"]), 1), ref, tol=0.05)
        check(f"UAV {alg} convergence, optimizer warm start (%)",
              round(float(u.loc[alg, "optimizer_warm"]), 1), opt, tol=0.05)

    td = pd.read_csv(d("tail_deterministic_summary.csv"), **RT).set_index("system")
    hv = (td.loc["hvac", "nrmse_hold"] - td.loc["hvac", "nrmse_zero"]) / td.loc["hvac", "nrmse_zero"]
    pc = (td.loc["pendcart", "nrmse_hold"] - td.loc["pendcart", "nrmse_zero"]) / td.loc["pendcart", "nrmse_zero"]
    check("deterministic nRMSE change, HVAC (%)", round(float(hv) * 100, 1), -5.3, tol=0.05)
    check("deterministic nRMSE change, pendulum-cart (%)", round(float(pc) * 100), 1707, tol=0.5)
    to = pd.read_csv(d("tail_ordering_comparison.csv"), **RT)
    fl = to[to.system == "flight"].set_index("metric_family")
    check("flight: SGO least accurate finite-horizon under both conventions",
          [int(fl.loc["optimization_accuracy_mean_gap", "sgo_rank_zero"]),
           int(fl.loc["optimization_accuracy_mean_gap", "sgo_rank_hold_last"])], [4, 4])
    check("flight: SGO closed-loop rank moves best -> worst",
          [int(fl.loc["closed_loop_nrmse", "sgo_rank_zero"]),
           int(fl.loc["closed_loop_nrmse", "sgo_rank_hold_last"])], [1, 4])

    pt = pd.read_csv(d("perturbation_summary.csv"), **RT)
    h = pt[pt.system == "hvac"]
    check("HVAC non-benign transitions, nominal (%)",
          round(float(h[h.condition == "nominal"].frac_non_benign.iloc[0]) * 100, 1), 0.0)
    check("HVAC non-benign transitions, process disturbance (%)",
          round(float(h[h.condition == "process_disturbance"].frac_non_benign.iloc[0]) * 100, 1), 86.7, tol=0.05)
    pdis = h[h.condition == "process_disturbance"].relative_excess_vs_reference * 100
    check("closed-loop excess over reference, min (%)", round(float(pdis.min()), 1), 13.8, tol=0.05)
    check("closed-loop excess over reference, max (%)", round(float(pdis.max()), 1), 21.9, tol=0.05)
    check("SGO last in all three perturbed HVAC conditions",
          sorted(h[(h.alg == "SGO") & (h.condition != "nominal")]["rank"].tolist()), [4, 4, 4])

    mi = pd.read_csv(d("multi_ic_raw.csv"), **RT)
    check("multi-initial-condition runs", len(mi), 500)
    check("seeds per cell", sorted(set(mi.groupby(["system", "ic", "alg"]).size())), [5])
    ab = pd.read_csv(d("sgo_ablation_summary.csv"), **RT)
    fg = ab[(ab.system == "flight") & (ab.budget_fe == 5000)].pivot(
        index="k", columns="variant", values="conv_rate") * 100
    check("flight plateau instances, baseline at 5000 FE (%)",
          sorted(round(float(v), 1) for v in fg["baseline"]), [30.0, 43.3])
    check("flight plateau instances, fitness grouping at 5000 FE (%)",
          sorted(round(float(v), 1) for v in fg["C_fitness_group"]), [86.7, 96.7])
    uv = ab[(ab.system == "uav") & (ab.budget_fe == 5000)]
    check("UAV at 5000 FE: fitness grouping never converges (%)",
          round(float(uv[uv.variant == "C_fitness_group"].conv_rate.max()) * 100, 1), 0.0)


# ---------------------------------------- original SGO static-benchmark rerun
def sgo_reproduction():
    head("Original-SGO static benchmark reproduction")
    raw = pd.read_csv(d("sgo_benchmark_raw.csv"), **RT)
    check("benchmark functions", raw.fid.nunique(), 13)
    check("independent runs per function and convention",
          sorted(set(raw.groupby(["fid", "variant"]).size())), [100])
    check("dimension", sorted(raw.dim.unique()), [100])
    check("evaluation budget", sorted(raw.budget.unique()), [150000])
    check("population size", sorted(raw.NP.unique()), [50])
    s = pd.read_csv(d("sgo_benchmark_summary.csv"), **RT)
    q = s[(s.fid == "F22") & (s.variant == "def_le_off")].iloc[0]
    check("Quintic mean, this implementation", round(float(q.our_mean), 1), 306.5, tol=0.05)
    check("Quintic mean, published", float(q.published_mean), 23.6, tol=1e-9)

    def agree(our, pub, fmin):
        a, b = abs(our - fmin), abs(pub - fmin)
        if b == 0:
            return a <= 1e-10
        return 0.5 <= a / b <= 2.0

    lit = s[s.variant == "def_le_off"]
    n = sum(agree(r.our_mean, r.published_mean, r.known_min) for _, r in lit.iterrows())
    check("functions matching the published mean (as-implemented convention)", int(n), 11)
    off = sorted(r.fid for _, r in lit.iterrows()
                 if not agree(r.our_mean, r.published_mean, r.known_min))
    check("functions outside the agreement band", off, ["F2", "F22"],
          note="F22 Quintic is worse than published; F2 Alpine 1 is closer to the optimum")


def main():
    print("Recomputing the reported numbers from the raw data under data/ ...")
    for fn in (gmpb_primary, gmpb_temporal, gmpb_grouping, mpc, sgo_reproduction):
        fn()
    n, k = len(_results), sum(_results)
    print(f"\n{k}/{n} checks passed")
    return 0 if k == n else 1


if __name__ == "__main__":
    sys.exit(main())
