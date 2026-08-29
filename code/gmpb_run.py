#!/usr/bin/env python3
"""Run the GMPB dynamic study.

    python code/gmpb_run.py primary  <cache> <outdir> <cases> <seeds> [nproc]
    python code/gmpb_run.py temporal <cache> <outdir> <cases> <seeds> [nproc]

``cases`` is a comma-separated list such as F1,F2 or ALL; ``seeds`` is either
``N`` (meaning 1..N) or ``a-b``.  The benchmark state for every (case, seed) must
already have been produced by the official Octave generator into ``cache``.

The main loop follows the official EDOLAB template: run one optimizer iteration;
if the benchmark reports an environmental change, acknowledge it and run the
change reaction; stop when the evaluation budget is exhausted.
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import GMPB, CASE_LIST, state_path      # noqa: E402
from gmpb_optimizers import OPTIMIZERS, LegacyRNG           # noqa: E402

# ---- frozen protocol constants (see data/gmpb/GMPB_PROTOCOL.md) -----
POPULATION_SIZE = 20            # identical for all optimizers and all cases
N_RUNS = 31                     # official CEC competition requirement
OPTIMIZER_SEED_BASE = 20260900  # optimizer stream; benchmark stream is the run index
ALG_OFFSET = {"SGO": 1, "GWO": 2, "PSO": 3, "WOA": 4}   # deterministic, no hashing
MODE_OFFSET = {"population_persistence": 0, "cold_restart": 10, "previous_best_seeded": 20}
TEMPORAL_CASES = ["F2", "F8", "F10", "F12"]
TEMPORAL_MODES = ["cold_restart", "previous_best_seeded", "population_persistence"]
RECOVERY_LEVELS = [0.5, 0.9]    # fractions of the within-environment error reduction


def _diagnostics(bench, opt, mode_extra=None):
    """Per-environment diagnostics, all derived from the recorded error trace.

    Every definition is parameter free.  Recovery is measured against each
    environment's own error reduction rather than an absolute error level,
    because the GMPB error scale varies by orders of magnitude across cases and
    no scale-free absolute threshold can be fixed in advance.
    """
    cf, T = bench.change_frequency, bench.T
    ce = bench.current_error.reshape(T, cf)
    rows = []
    for t in range(T):
        seg = ce[t]
        e0, e1 = float(seg[0]), float(seg[-1])
        red = e0 - e1
        rec = {}
        for lv in RECOVERY_LEVELS:
            tag = f"{int(lv * 100)}"
            if red > 0:
                idx = np.nonzero(seg <= e0 - lv * red)[0]
                rec[f"recovery_evals_{tag}"] = int(idx[0] + 1) if idx.size else -1
                rec[f"recovery_success_{tag}"] = bool(idx.size)
            else:
                rec[f"recovery_evals_{tag}"] = -1
                rec[f"recovery_success_{tag}"] = False
        rows.append(dict(environment=t + 1,
                         post_change_error=e0,
                         end_of_environment_error=e1,
                         mean_error=float(seg.mean()),
                         error_reduction=float(red),
                         error_reduction_fraction=float(red / e0) if e0 > 0 else float("nan"),
                         **rec))
    return rows


def _run_one(cache, case, seed, alg_name, mode="population_persistence",
             collect_env=False):
    """One independent run: benchmark seed = run index, optimizer seed derived."""
    bench = GMPB(state_path(cache, case, seed))
    rng = LegacyRNG(OPTIMIZER_SEED_BASE + 1000 * seed
                    + 10 * ALG_OFFSET[alg_name] + MODE_OFFSET[mode])
    opt = OPTIMIZERS[alg_name](bench, rng, POPULATION_SIZE)
    t0 = time.perf_counter()
    opt.initialize()
    n_reeval = 0
    n_changes = 0
    div = []
    while not bench.finished:
        opt.iteration()
        if bench.recent_change:
            bench.acknowledge_change()
            n_changes += 1
            fe_before = bench.FE
            if mode == "population_persistence":
                opt.reevaluate()
            elif mode == "cold_restart":
                opt.initialize()
            elif mode == "previous_best_seeded":
                opt.initialize(seed_x=opt.best_x)
            else:
                raise ValueError(mode)
            n_reeval += bench.FE - fe_before
            if collect_env and hasattr(opt, "X"):
                span = bench.ub - bench.lb
                div.append(float(np.mean(np.std(opt.X, axis=0)) / span))
    wall = time.perf_counter() - t0
    rec = dict(case=case, seed=seed, alg=alg_name, mode=mode,
               NP=POPULATION_SIZE, dimension=bench.d, peaks=bench.m,
               change_frequency=bench.change_frequency,
               shift_severity=bench.shift_severity,
               environments=bench.T, max_evals=bench.max_evals,
               realized_FE=bench.FE, budget_exact=bool(bench.FE == bench.max_evals),
               environments_completed=bench.env, changes_seen=n_changes,
               reevaluation_FE=n_reeval,
               offline_error=bench.offline_error,
               best_error_before_change=bench.best_error_before_change,
               mean_population_diversity=float(np.mean(div)) if div else float("nan"),
               wall_seconds=wall)
    env_rows = None
    if collect_env:
        env_rows = [dict(case=case, seed=seed, alg=alg_name, mode=mode, **r)
                    for r in _diagnostics(bench, opt)]
    return rec, env_rows


def _parse(cases, seeds):
    cs = CASE_LIST if cases.upper() == "ALL" else cases.split(",")
    if "-" in seeds:
        a, b = seeds.split("-")
        sd = list(range(int(a), int(b) + 1))
    else:
        sd = list(range(1, int(seeds) + 1))
    return cs, sd


def _write(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def _done_set(path, keys):
    """Which (case, seed, alg, mode) combinations are already in the output file."""
    if not os.path.exists(path):
        return set()
    import pandas as pd
    try:
        d = pd.read_csv(path)
    except Exception:
        return set()
    if d.empty:
        return set()
    return set(map(tuple, d[list(keys)].astype(str).values.tolist()))


def main():
    what, cache, outdir, cases, seeds = sys.argv[1:6]
    nproc = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    cs, sd = _parse(cases, seeds)
    os.makedirs(outdir, exist_ok=True)

    if what == "primary":
        raw = os.path.join(outdir, "gmpb_primary_raw.csv")
        envp = os.path.join(outdir, "gmpb_primary_environments.csv")
        jobs = [(c, s, a, "population_persistence")
                for c in cs for s in sd for a in ["SGO", "GWO", "PSO", "WOA"]]
    else:
        raw = os.path.join(outdir, "gmpb_sgo_temporal_raw.csv")
        envp = os.path.join(outdir, "gmpb_sgo_temporal_environments.csv")
        jobs = [(c, s, "SGO", m)
                for c in cs if c in TEMPORAL_CASES for s in sd for m in TEMPORAL_MODES]

    # resume: skip anything already recorded
    done = _done_set(raw, ("case", "seed", "alg", "mode"))
    todo = [j for j in jobs if (str(j[0]), str(j[1]), str(j[2]), str(j[3])) not in done]
    print(f"{what}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run", flush=True)

    t0 = time.perf_counter()
    n = 0
    if nproc > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = {ex.submit(_worker, (cache,) + j): j for j in todo}
            for fut in as_completed(futs):
                rec, env_rows = fut.result()
                _write(raw, [rec])
                if env_rows:
                    _write(envp, env_rows)
                n += 1
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} runs, "
                          f"{time.perf_counter() - t0:.0f}s", flush=True)
    else:
        for j in todo:
            rec, env_rows = _worker((cache,) + j)
            _write(raw, [rec])
            if env_rows:
                _write(envp, env_rows)
            n += 1
    print(json.dumps(dict(kind=what, jobs=len(jobs), completed_now=n,
                          wall_seconds=round(time.perf_counter() - t0, 1),
                          raw=raw), indent=1))


def _worker(args):
    cache, c, s, a, m = args
    return _run_one(cache, c, s, a, m, collect_env=True)


if __name__ == "__main__":
    main()
