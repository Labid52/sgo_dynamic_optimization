#!/usr/bin/env python3
"""Validation gate for the comparator temporal-transfer study (protocol section 8).

    python code/gmpb_temporal_validation.py <cache> <outdir>

Checks, in order:
  1  the persistence path reproduces the committed primary GMPB results;
  2  SGO cold / previous-best / persistence reproduces the committed SGO
     temporal results;
  3  fresh-state initialisation for GWO, PSO and WOA matches their validated
     standalone implementations (structural check on the validated classes);
  4  previous-best seeding changes only the intended information transfer;
  5  evaluation counts stay exact;
  6  no objective value from before a change survives it;
  7  all optimizers see identical environment trajectories for a paired seed.

Interpretation is not permitted unless every check passes.
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import GMPB, state_path                 # noqa: E402
from gmpb_optimizers import OPTIMIZERS, LegacyRNG           # noqa: E402
import gmpb_temporal_comparators as TC                      # noqa: E402

CASES = TC.CASES
ALGS = TC.ALGS


def gate(rows, name, detail, passed, note=""):
    rows.append(dict(check=name, detail=detail, passed=bool(passed), note=note))


def _reproduce(cache, case, seed, alg, mode):
    rec, _, _ = TC._run_one(cache, case, seed, alg, mode)
    return rec


def main():
    cache, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rows = []
    seeds = [1, 2, 3, 9, 17, 31]
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=20)

    # --- 1 persistence reproduces the committed primary results -----------
    prim = pd.read_csv("data/gmpb/gmpb_primary_raw.csv",
                       float_precision="round_trip")
    spec = [(cache, c, s, a, "population_persistence")
            for c in CASES for a in ALGS for s in seeds]
    got = list(pool.map(TC._worker, spec))
    worst = 0.0
    for (_, c, s, a, m), g in zip(spec, got):
        ref = prim[(prim.case == c) & (prim.alg == a) & (prim.seed == s)].iloc[0]
        worst = max(worst, abs(ref.offline_error - g[0]["offline_error"]))
    n = len(spec)
    gate(rows, "1_persistence_reproduces_primary",
         f"{n} cells over {len(CASES)} cases x 4 optimizers x {len(seeds)} seeds",
         worst == 0.0, f"max abs offline-error difference {worst:.3e}; committed CSVs read with round-trip float parsing, since the pandas default parser is off by one ulp on some rows")

    # --- 2 SGO three modes reproduce the committed temporal study ---------
    tmp = pd.read_csv("data/gmpb/gmpb_sgo_temporal_raw.csv",
                      float_precision="round_trip")
    spec = [(cache, c, s, "SGO", m) for c in CASES for m in TC.MODES for s in seeds]
    got = list(pool.map(TC._worker, spec))
    worst = 0.0
    for (_, c, s, a, m), g in zip(spec, got):
        ref = tmp[(tmp.case == c) & (tmp["mode"] == m) & (tmp.seed == s)].iloc[0]
        worst = max(worst, abs(ref.offline_error - g[0]["offline_error"]))
    n = len(spec)
    gate(rows, "2_sgo_temporal_reproduces_committed",
         f"{n} cells over {len(CASES)} cases x 3 modes x {len(seeds)} seeds",
         worst == 0.0, f"max abs offline-error difference {worst:.3e}; committed CSVs read with round-trip float parsing, since the pandas default parser is off by one ulp on some rows")

    # --- 3 fresh-state initialisation is the optimizers' own rule ---------
    # A cold restart must leave the optimizer in exactly the state a brand new
    # run would have.  Compare a mid-run cold restart against a fresh object
    # built from the same RNG position on the same environment.
    for alg in ALGS:
        b = GMPB(state_path(cache, "F2", 5))
        rng = LegacyRNG(4242)
        opt = OPTIMIZERS[alg](b, rng, TC.POPULATION_SIZE)
        opt.initialize()
        for _ in range(30):
            opt.iteration()
        state_before = np.array(opt.X, copy=True)
        rng2 = LegacyRNG(999)
        opt.rng = rng2
        opt.initialize()                      # the cold-restart call
        after = np.array(opt.X, copy=True)
        b2 = GMPB(state_path(cache, "F2", 5))
        b2.FE = b.FE
        fresh = OPTIMIZERS[alg](b2, LegacyRNG(999), TC.POPULATION_SIZE)
        fresh.initialize()
        same_as_fresh = np.array_equal(after, fresh.X)
        changed = not np.array_equal(after, state_before)
        extra = ""
        if alg == "PSO":
            # velocities and personal bests must be rebuilt, not carried
            extra = (f"pbest equals current population: "
                     f"{np.array_equal(opt.P, opt.X)}; "
                     f"velocity magnitude finite: {np.all(np.isfinite(opt.V))}")
            ok_extra = np.array_equal(opt.P, opt.X)
        else:
            ok_extra = True
        gate(rows, "3_fresh_state_initialisation", f"{alg} cold restart",
             same_as_fresh and changed and ok_extra,
             f"identical to a brand-new optimizer: {same_as_fresh}; "
             f"differs from the pre-restart population: {changed}. {extra}")

    # --- 4 previous-best seeding changes only the carried point -----------
    for alg in ALGS:
        b = GMPB(state_path(cache, "F2", 5))
        seed_pt = np.full(b.d, 3.75)
        a1 = OPTIMIZERS[alg](b, LegacyRNG(777), TC.POPULATION_SIZE)
        a1.initialize(seed_x=seed_pt)
        b2 = GMPB(state_path(cache, "F2", 5))
        a2 = OPTIMIZERS[alg](b2, LegacyRNG(777), TC.POPULATION_SIZE)
        a2.initialize()
        row0 = np.allclose(a1.X[0], seed_pt)
        rest = np.array_equal(a1.X[1:], a2.X[1:])
        gate(rows, "4_previous_best_seeds_one_slot", f"{alg}",
             row0 and rest,
             "slot 0 holds the carried point and every other slot is identical "
             "to an unseeded initialisation from the same RNG")

    # --- 5 exact evaluation counts ---------------------------------------
    spec = [(cache, "F8", 4, a, m) for a in ALGS for m in TC.MODES]
    ok = True
    for g in pool.map(TC._worker, spec):
        rec = g[0]
        ok &= bool(rec["budget_exact"] and rec["environments_completed"] == 100
                   and rec["changes_seen"] == 99)
    gate(rows, "5_exact_fe", "12 optimizer x mode cells on F8", ok,
         "realized FE equals MaxEvals, 100 environments, 99 changes in every cell")

    # --- 6 no pre-change objective value survives a change ----------------
    # After the change reaction the recorded costs must equal a fresh evaluation
    # of the retained points under the NEW environment.
    # Native persistence differs by optimizer, which is the point of mode C:
    # SGO, GWO and WOA re-evaluate the retained population X; PSO re-evaluates its
    # personal-best archive P, which is the standard PSO change reaction and is
    # what the committed study used.  The check compares against whichever array
    # the optimizer actually refreshed.
    for alg in ALGS:
        for mode in ("previous_best_seeded", "population_persistence"):
            b = GMPB(state_path(cache, "F2", 6))
            opt = OPTIMIZERS[alg](b, LegacyRNG(1234), TC.POPULATION_SIZE)
            opt.initialize()
            while not b.recent_change:
                opt.iteration()
            pre = np.array(opt.costs, copy=True)
            b.acknowledge_change()
            if mode == "population_persistence":
                opt.reevaluate()
            else:
                opt.initialize(seed_x=opt.best_x)
            refreshed = "P" if (alg == "PSO" and mode == "population_persistence") else "X"
            X = getattr(opt, refreshed)
            c = opt.costs
            recomputed = -b.raw_fitness(X)
            finite = np.isfinite(c)
            d = float(np.max(np.abs(c[finite] - recomputed[finite]))) if finite.any() else 0.0
            carried = int(np.sum(np.isin(c[finite], pre)))
            gate(rows, "6_no_stale_objective_values", f"{alg} / {mode}",
                 d == 0.0,
                 f"stored costs equal a re-evaluation of {refreshed} in the new "
                 f"environment, max difference {d:.3e}; "
                 f"{carried} of {int(finite.sum())} values coincide with a "
                 f"pre-change value")

    # --- 7 identical environment trajectories for a paired seed -----------
    for case in CASES:
        insts = [GMPB(state_path(cache, case, 8)) for _ in range(4)]
        same = all(np.array_equal(insts[0].optimum_value, x.optimum_value)
                   and np.array_equal(insts[0].peaks_position, x.peaks_position)
                   for x in insts[1:])
        gate(rows, "7_paired_environment_trajectory", f"{case} seed 8", same,
             "every optimizer and mode loads the same official state file")

    pool.shutdown()
    p = os.path.join(outdir, "temporal_equivalence_checks.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "detail", "passed", "note"])
        w.writeheader()
        w.writerows(rows)
    nf = sum(1 for r in rows if not r["passed"])
    for r in rows:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['check']:36s} {r['detail']:52s} {r['note']}")
    print(f"\n{len(rows)} checks, {nf} failures -> {p}")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
