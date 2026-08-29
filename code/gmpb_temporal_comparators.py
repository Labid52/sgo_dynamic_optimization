#!/usr/bin/env python3
"""Comparator temporal-transfer study on GMPB.

Asks whether the temporal-information effects already measured for the Squid Game
Optimizer (Azizi et al., Sci. Rep. 13:5373, 2023) are peculiar to it or are a
general property of population-based optimizers on this benchmark.

    python code/gmpb_temporal_comparators.py <cache> <outdir> <cases> <seeds> \
                                              <algs> <modes> [nproc]

Uses the validated optimizer implementations in ``code/gmpb_optimizers.py``
unchanged, the validated benchmark in ``code/gmpb_benchmark.py`` unchanged, and
the seed recipe of ``code/gmpb_run.py`` unchanged, so runs that duplicate an
already committed cell must reproduce it exactly.  Writes incrementally and
resumes.
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import GMPB, state_path                  # noqa: E402
from gmpb_optimizers import OPTIMIZERS, LegacyRNG            # noqa: E402
import gmpb_run                                              # noqa: E402

POPULATION_SIZE = gmpb_run.POPULATION_SIZE           # 20
RECOVERY_LEVELS = gmpb_run.RECOVERY_LEVELS           # [0.5, 0.9]
OPTIMIZER_SEED_BASE = gmpb_run.OPTIMIZER_SEED_BASE
ALG_OFFSET = gmpb_run.ALG_OFFSET
MODE_OFFSET = gmpb_run.MODE_OFFSET
FRACTIONS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75]
CASES = ["F2", "F8", "F10", "F12"]
ALGS = ["SGO", "GWO", "PSO", "WOA"]
MODES = ["cold_restart", "previous_best_seeded", "population_persistence"]


def _optimizer_seed(seed, alg, mode):
    """Identical to the committed studies, so duplicated cells reproduce exactly."""
    return (OPTIMIZER_SEED_BASE + 1000 * seed
            + 10 * ALG_OFFSET[alg] + MODE_OFFSET[mode])


def _diversity(opt, bench):
    """Normalised positional diversity, same definition as the GMPB study."""
    return float(np.mean(np.std(opt.X, axis=0)) / (bench.ub - bench.lb))


def _env_rows(bench, case, seed, alg, mode):
    cf, T = bench.change_frequency, bench.T
    ce = bench.current_error.reshape(T, cf)
    idx = {f: min(int(round(f * cf)), cf - 1) for f in FRACTIONS}
    rows = []
    for t in range(T):
        seg = ce[t]
        e0, e1 = float(seg[0]), float(seg[-1])
        red = e0 - e1
        r = dict(case=case, seed=seed, alg=alg, mode=mode, environment=t + 1,
                 change_frequency=cf, post_change_error=e0,
                 end_of_environment_error=e1, mean_error=float(seg.mean()),
                 error_reduction=float(red),
                 error_reduction_fraction=float(red / e0) if e0 > 0 else float("nan"))
        for f in FRACTIONS:
            r[f"error_at_{int(f * 100)}"] = float(seg[idx[f]])
        e50 = float(seg[idx[0.50]])
        r["late_half_reduction"] = e50 - e1
        r["late_half_fraction"] = float((e50 - e1) / red) if red > 0 else float("nan")
        for lv in RECOVERY_LEVELS:
            tag = str(int(lv * 100))
            if red > 0:
                hit = np.nonzero(seg <= e0 - lv * red)[0]
                r[f"recovery_evals_{tag}"] = int(hit[0] + 1) if hit.size else -1
                r[f"recovery_success_{tag}"] = bool(hit.size)
                r[f"recovery_frac_{tag}"] = (float((hit[0] + 1) / cf) if hit.size
                                             else float("nan"))
            else:
                r[f"recovery_evals_{tag}"] = -1
                r[f"recovery_success_{tag}"] = False
                r[f"recovery_frac_{tag}"] = float("nan")
        rows.append(r)
    return rows


def _run_one(cache, case, seed, alg, mode):
    """One run.  The three modes differ only in what survives a change.

    cold_restart          : opt.initialize() rebuilds population and all
                            algorithm state from scratch and evaluates it in the
                            new environment.
    previous_best_seeded  : the same fresh rebuild, with exactly one individual
                            replaced by that optimizer's best point from the
                            previous environment, re-evaluated in the new one.
                            No old objective value survives.
    population_persistence: opt.reevaluate(), the native persistence path already
                            used by the committed primary study.
    """
    bench = GMPB(state_path(cache, case, seed))
    rng = LegacyRNG(_optimizer_seed(seed, alg, mode))
    opt = OPTIMIZERS[alg](bench, rng, POPULATION_SIZE)
    cf = bench.change_frequency
    targets = [int(round(f * cf)) for f in FRACTIONS]
    div_rows = []

    def _record(env, k, frac_label):
        div_rows.append(dict(case=case, seed=seed, alg=alg, mode=mode,
                             environment=env, target_fraction=frac_label,
                             actual_fe_in_env=k, actual_fraction=float(k / cf),
                             diversity=_diversity(opt, bench)))

    t0 = time.perf_counter()
    opt.initialize()
    init_fe = bench.FE
    _record(1, bench.FE % cf, 0.0)
    nxt = 1
    n_reeval = n_changes = 0
    while not bench.finished:
        opt.iteration()
        k = bench.FE % cf
        env = bench.env
        if bench.recent_change:
            _record(env - 1, cf, 1.0)
            bench.acknowledge_change()
            n_changes += 1
            fe0 = bench.FE
            if mode == "population_persistence":
                opt.reevaluate()
            elif mode == "cold_restart":
                opt.initialize()
            elif mode == "previous_best_seeded":
                opt.initialize(seed_x=opt.best_x)
            else:
                raise ValueError(mode)
            n_reeval += bench.FE - fe0
            _record(bench.env, bench.FE % cf, 0.0)
            nxt = 1
            continue
        while nxt < len(FRACTIONS) and k >= targets[nxt]:
            _record(env, k, FRACTIONS[nxt])
            nxt += 1
    _record(bench.T, cf, 1.0)

    rec = dict(case=case, seed=seed, alg=alg, mode=mode, NP=POPULATION_SIZE,
               dimension=bench.d, peaks=bench.m, change_frequency=cf,
               shift_severity=bench.shift_severity, environments=bench.T,
               max_evals=bench.max_evals, realized_FE=bench.FE,
               budget_exact=bool(bench.FE == bench.max_evals),
               environments_completed=bench.env, changes_seen=n_changes,
               initialization_FE=init_fe, reevaluation_FE=n_reeval,
               offline_error=bench.offline_error,
               best_error_before_change=bench.best_error_before_change,
               mean_population_diversity=float(np.mean([r["diversity"] for r in div_rows])),
               wall_seconds=time.perf_counter() - t0)
    return rec, _env_rows(bench, case, seed, alg, mode), div_rows


def _worker(a):
    return _run_one(*a)


def _write(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def _done(path, keys):
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
    cache, outdir, cases, seeds, algs, modes = sys.argv[1:7]
    nproc = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    cs = CASES if cases.upper() == "ALL" else cases.split(",")
    al = ALGS if algs.upper() == "ALL" else algs.split(",")
    md = MODES if modes.upper() == "ALL" else modes.split(",")
    if "-" in seeds:
        a, b = seeds.split("-")
        sd = list(range(int(a), int(b) + 1))
    else:
        sd = list(range(1, int(seeds) + 1))
    os.makedirs(outdir, exist_ok=True)

    raw = os.path.join(outdir, "gmpb_temporal_comparator_raw.csv")
    envp = os.path.join(outdir, "gmpb_temporal_comparator_environments.csv")
    divp = os.path.join(outdir, "gmpb_temporal_comparator_diversity_trace.csv")

    jobs = [(cache, c, s, a, m) for c in cs for s in sd for a in al for m in md]
    have = _done(raw, ("case", "seed", "alg", "mode"))
    todo = [j for j in jobs
            if (str(j[1]), str(j[2]), str(j[3]), str(j[4])) not in have]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run",
          flush=True)

    t0 = time.perf_counter()
    n = 0
    if nproc > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = [ex.submit(_worker, j) for j in todo]
            for fut in as_completed(futs):
                rec, er, dr = fut.result()
                _write(raw, [rec]); _write(envp, er); _write(divp, dr)
                n += 1
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}, {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        for j in todo:
            rec, er, dr = _worker(j)
            _write(raw, [rec]); _write(envp, er); _write(divp, dr)
            n += 1
    print(json.dumps(dict(jobs=len(jobs), completed_now=n,
                          wall_seconds=round(time.perf_counter() - t0, 1)), indent=1))


if __name__ == "__main__":
    main()
