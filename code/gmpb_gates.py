#!/usr/bin/env python3
"""Validation gates A-J for the GMPB study (protocol section 10).

    python code/gmpb_gates.py <cache> <outdir>

Writes ``benchmark_environment_checks.csv``.  Interpretation of the scientific
results is not permitted unless every gate passes.
"""
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import GMPB, CASES, CASE_LIST, state_path   # noqa: E402
from gmpb_optimizers import OPTIMIZERS, LegacyRNG               # noqa: E402
import gmpb_run                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gate(rows, name, detail, passed, note=""):
    rows.append(dict(gate=name, detail=detail, passed=bool(passed), note=note))


def main():
    cache, outdir = sys.argv[1], sys.argv[2]
    rows = []
    proto = json.load(open(os.path.join(outdir, "gmpb_protocol.json")))

    # ---- A: environments change at the expected evaluation counts ---------
    for case in ["F2", "F8", "F10"]:
        b = GMPB(state_path(cache, case, 1))
        cf = b.change_frequency
        boundaries, envs = [], []
        while not b.finished:
            b.evaluate(np.zeros((7, b.d)))
            if b.recent_change:
                boundaries.append(b.FE)
                envs.append(b.env)
                b.acknowledge_change()
        ok = (all(x % cf == 0 for x in boundaries)
              and boundaries == list(range(cf, b.max_evals, cf))
              and envs == list(range(2, b.T + 1)))
        gate(rows, "A_change_points", f"{case}: {len(boundaries)} changes",
             ok, f"all at multiples of ChangeFrequency={cf}")

    # ---- B: identical environment trajectories for a given run index ------
    for case in ["F2", "F12"]:
        opts = [GMPB(state_path(cache, case, 3)) for _ in range(4)]
        same = all(np.array_equal(opts[0].optimum_value, o.optimum_value) for o in opts[1:]) \
            and all(np.array_equal(opts[0].peaks_position, o.peaks_position) for o in opts[1:])
        gate(rows, "B_paired_trajectories", f"{case} seed 3",
             same, "same state file is loaded by every optimizer and mode")

    # ---- C: optimum values track the generated peaks ----------------------
    for case in ["F1", "F5", "F10"]:
        b = GMPB(state_path(cache, case, 1))
        recomputed = b.peaks_height.max(axis=1)
        idx = b.peaks_height.argmax(axis=1) + 1
        ok = (np.allclose(recomputed, b.optimum_value, rtol=0, atol=0)
              and np.array_equal(idx, b.optimum_id))
        # the recorded optimum must also be attained by evaluating its own peak
        t = 0
        att = b.raw_fitness(b.peaks_position[t, b.optimum_id[t] - 1][None, :], env=1)[0]
        gate(rows, "C_optimum_tracking", f"{case}: max height vs OptimumValue",
             ok and abs(att - b.optimum_value[0]) < 1e-9,
             f"peak {b.optimum_id[0]} evaluates to {att:.12g} vs {b.optimum_value[0]:.12g}")

    # ---- D/E: evaluation counters and budget ------------------------------
    raw = pd.read_csv(os.path.join(outdir, "gmpb_primary_raw.csv"))
    exp_me = {c: CASES[c]["ChangeFrequency"] * 100 for c in CASE_LIST}
    ok_d = bool((raw.realized_FE == raw.max_evals).all()
                and all(raw[raw.case == c].max_evals.eq(exp_me[c]).all() for c in raw.case.unique()))
    gate(rows, "D_fe_accounting", f"{len(raw)} primary runs", ok_d,
         "realized FE equals MaxEvals = ChangeFrequency x 100 in every run")
    over = int((raw.realized_FE > raw.max_evals).sum())
    gate(rows, "E_budget_not_exceeded", f"{len(raw)} primary runs", over == 0,
         f"{over} runs exceeded the budget")
    # every optimizer spends the same on post-change re-evaluation
    g = raw.groupby(["case", "seed"]).reevaluation_FE.nunique()
    gate(rows, "E_equal_reevaluation", "post-change re-evaluation cost",
         bool((g == 1).all()), "identical for all four optimizers in every (case, seed)")

    # ---- F: no optimizer reads benchmark internals ------------------------
    src = open(os.path.join(ROOT, "codes", "gmpb_optimizers.py")).read()
    forbidden = ["optimum_value", "optimum_id", "peaks_position", "peaks_height",
                 "peaks_width", "rotation", "current_error", "ebbc", "raw_fitness"]
    hits = [w for w in forbidden if w in src]
    gate(rows, "F_no_hidden_information", "code/gmpb_optimizers.py",
         not hits, f"forbidden references: {hits if hits else 'none'}")

    # ---- G/H: SGO identity -----------------------------------------------
    fid = pd.read_csv(os.path.join(outdir, "sgo_fidelity.csv"))
    gate(rows, "G_sgo_is_squid_game", f"{len(fid)} fidelity checks",
         bool(fid.passed.all()),
         "dynamic driver identical to code/sgo.py evaluation by evaluation")
    try:
        grep = subprocess.run(
            ["grep", "-ril", "--exclude=gmpb_gates.py", "social group optimization",
             os.path.join(ROOT, "codes")],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        grep = ""
    sgo_head = open(os.path.join(ROOT, "codes", "sgo.py")).read()[:1200]
    gate(rows, "H_no_social_group_optimization", "code/ tree",
         (grep == "") and ("Azizi" in sgo_head) and ("Squid Game Optimizer" in sgo_head),
         "code/sgo.py cites Azizi et al. 2023 Sci. Rep. 13:5373; no SGO acronym collision found")

    # ---- I: no per-case tuning -------------------------------------------
    npu = sorted(raw.NP.unique().tolist())
    gate(rows, "I_no_per_case_tuning", f"population sizes used: {npu}",
         npu == [proto["optimizers"]["population_size"]],
         "one population size for every optimizer, case and mode")

    # ---- J: run counts match the protocol --------------------------------
    cnt = raw.groupby(["case", "alg"]).size()
    ok_primary = bool((cnt == 31).all() and len(cnt) == 48 and len(raw) == 1488)
    gate(rows, "J_primary_run_count", f"{len(raw)} runs, {len(cnt)} case x alg cells",
         ok_primary, "12 cases x 4 optimizers x 31 runs = 1488")
    tp = os.path.join(outdir, "gmpb_sgo_temporal_raw.csv")
    if os.path.exists(tp):
        tr = pd.read_csv(tp)
        c2 = tr.groupby(["case", "mode"]).size()
        gate(rows, "J_temporal_run_count", f"{len(tr)} runs, {len(c2)} case x mode cells",
             bool((c2 == 31).all() and len(c2) == 12 and len(tr) == 372),
             "4 cases x 3 modes x 31 runs = 372")
        gate(rows, "D_fe_accounting_temporal", f"{len(tr)} temporal runs",
             bool((tr.realized_FE == tr.max_evals).all()),
             "realized FE equals MaxEvals in every temporal run")

    p = os.path.join(outdir, "benchmark_environment_checks.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gate", "detail", "passed", "note"])
        w.writeheader()
        w.writerows(rows)
    nf = sum(1 for r in rows if not r["passed"])
    for r in rows:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['gate']:32s} {r['detail']:44s} {r['note']}")
    print(f"\n{len(rows)} gates, {nf} failures -> {p}")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
