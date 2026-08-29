#!/usr/bin/env python3
"""Phase-2B step 4: define and FREEZE the drift strata and the frozen-instance list.

Selection uses ONLY drift metrics computed in step 3. No optimizer result is
consulted, and this file is committed before any frozen optimizer trial is run.

Stratification variable: the shifted-reference warm-start normalized
suboptimality, because it directly measures the optimization consequence of
moving from one receding-horizon problem to the next.

Quantile strata were rejected after inspecting the drift distribution ALONE:
93 % of transitions sit below the 1e-4 optimization-relevance criterion, so a
50/75/90 percentile split would place the Low/Moderate boundary in numerical
noise (medians range from 1e-15 to 1e-6) and would not isolate the severe tail.
Absolute strata on the same scale as the reported optimality gaps are used
instead:

    Benign     warm_gap <= 1e-4        (below the convergence criterion)
    Moderate   1e-4 < warm_gap <= 1e-2
    High       1e-2 < warm_gap <= 1e-1
    Severe     warm_gap  > 1e-1

Known imbalance, recorded here rather than engineered away: the Severe stratum
holds 19 transitions of which 16 are UAV, and HVAC contributes none at all
(its maximum warm gap over 1500 transitions is 6.3e-5). The analysis therefore
also uses a CONTINUOUS association measure (Spearman on warm_gap vs optimizer
error), which does not depend on stratum balance.

Instance selection per system:
  * uniform coverage: 25 evenly spaced steps over the trajectory
  * stratified coverage: up to 8 instances per stratum, taken at evenly spaced
    ranks within that stratum's warm-gap ordering (deterministic, drift-only)
  * duplicates between the two sets are recorded, not silently dropped

Outputs:
    data/drift_strata_definition.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(PROJECT, "data")
sys.path.insert(0, ROOT)
import phase2_protocol as P2                                           # noqa: E402

STRATA = [("Benign", -np.inf, 1e-4), ("Moderate", 1e-4, 1e-2),
          ("High", 1e-2, 1e-1), ("Severe", 1e-1, np.inf)]
N_UNIFORM = 25
N_PER_STRATUM = 8


def assign(w):
    for name, lo, hi in STRATA:
        if lo < w <= hi:
            return name
    return "Benign"


def main():
    t0 = time.perf_counter()
    df = pd.read_csv(os.path.join(OUT, "objective_drift_full_raw.csv"))
    df["stratum"] = df.warm_gap_norm.map(assign)

    payload = {
        "stratification_variable": "warm_gap_norm (shifted-reference warm-start "
                                   "normalized suboptimality)",
        "selection_uses_optimizer_results": False,
        "strata_definition": {n: {"lower_exclusive": (None if np.isneginf(lo) else lo),
                                  "upper_inclusive": (None if np.isposinf(hi) else hi)}
                              for n, lo, hi in STRATA},
        "why_absolute_not_quantile":
            "93% of all consecutive transitions fall below the 1e-4 optimization-relevance "
            "criterion, so quantile boundaries at the 50th/75th percentile would sit in "
            "numerical noise (per-system medians span 1e-15 to 1e-6) and would not isolate "
            "the severe tail. Absolute cut points on the same scale as the reported "
            "optimality gaps keep the strata interpretable.",
        "known_imbalance":
            "The Severe stratum contains 19 transitions, 16 of them UAV; HVAC contributes no "
            "non-benign transition at all (max warm gap 6.3e-5 over 1500 transitions). This is "
            "reported as a finding and handled by also using a continuous Spearman association "
            "between warm-gap severity and optimizer error, which does not depend on stratum "
            "balance. The strata were NOT redrawn to balance systems.",
        "n_uniform_per_system": N_UNIFORM,
        "n_per_stratum_per_system": N_PER_STRATUM,
        "global_counts": df.stratum.value_counts().to_dict(),
        "systems": {},
    }

    for key in P2.SYSTEMS:
        g = df[df.system == key].sort_values("k").reset_index(drop=True)
        Nsim = int(g.Nsim.iloc[0])
        uniform = sorted(set(np.linspace(0, Nsim - 1, N_UNIFORM).astype(int).tolist()))
        strat_sel = {}
        for name, _, _ in STRATA:
            gs = g[g.stratum == name].sort_values("warm_gap_norm")
            if len(gs) == 0:
                strat_sel[name] = []
                continue
            idx = np.unique(np.linspace(0, len(gs) - 1, min(N_PER_STRATUM, len(gs))).astype(int))
            strat_sel[name] = sorted(int(gs.iloc[i].k) for i in idx)
        stratified = sorted({k for v in strat_sel.values() for k in v})
        dup = sorted(set(uniform) & set(stratified))
        allk = sorted(set(uniform) | set(stratified))
        info = {int(r.k): {"warm_gap_norm": float(r.warm_gap_norm),
                           "stratum": r.stratum,
                           "cold_gap_norm": float(r.cold_gap_norm),
                           "df_over_max": float(r.df_over_max),
                           "dU_shift": float(r.dU_shift)}
                for _, r in g.iterrows() if int(r.k) in allk}
        payload["systems"][key] = {
            "Nsim": Nsim,
            "stratum_counts_full_trajectory": g.stratum.value_counts().to_dict(),
            "uniform_instances": uniform,
            "stratified_instances_by_stratum": strat_sel,
            "duplicates_uniform_and_stratified": dup,
            "selected_instances": allk,
            "n_selected": len(allk),
            "instance_details": info,
        }
        print(f"  {key:9s} Nsim={Nsim:5d} selected={len(allk):3d} "
              f"(uniform {len(uniform)}, stratified {len(stratified)}, dup {len(dup)}) "
              f"strata={g.stratum.value_counts().to_dict()}")

    payload["total_selected_instances"] = sum(v["n_selected"] for v in payload["systems"].values())
    payload["wall_seconds"] = round(time.perf_counter() - t0, 1)
    path = os.path.join(OUT, "drift_strata_definition.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\ntotal frozen instances: {payload['total_selected_instances']}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
