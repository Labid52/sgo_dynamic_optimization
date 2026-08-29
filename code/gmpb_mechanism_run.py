#!/usr/bin/env python3
"""Run the GMPB grouping-mechanism ablation.

    python code/gmpb_mechanism_run.py <cache> <outdir> <cases> <seeds> \
                                       <grouping> <mode> [nproc]

Writes incrementally and resumes, so a long study survives interruption.
Diagnostics follow data/gmpb_mechanism/GMPB_MECHANISM_PROTOCOL.md.
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import GMPB, CASE_LIST, state_path       # noqa: E402
from gmpb_optimizers import LegacyRNG                        # noqa: E402
from gmpb_sgo_mechanism import SGOGrouping                   # noqa: E402
import gmpb_run                                              # noqa: E402

# inherited verbatim from the completed GMPB study
POPULATION_SIZE = gmpb_run.POPULATION_SIZE          # 20
RECOVERY_LEVELS = gmpb_run.RECOVERY_LEVELS          # [0.5, 0.9]
OPTIMIZER_SEED_BASE = gmpb_run.OPTIMIZER_SEED_BASE
ALG_OFFSET = gmpb_run.ALG_OFFSET
MODE_OFFSET = gmpb_run.MODE_OFFSET
# frozen before outcomes
FRACTIONS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75]     # 1.0 recorded as "end"
INTERACTION_CASES = ["F2", "F8", "F10", "F12"]


def _optimizer_seed(seed, mode):
    """Identical to the baseline study, so paired runs share the seed recipe."""
    return (OPTIMIZER_SEED_BASE + 1000 * seed
            + 10 * ALG_OFFSET["SGO"] + MODE_OFFSET[mode])


def _env_rows(bench, case, seed, grouping, mode):
    """Per-environment error diagnostics from the official CurrentError trace."""
    cf, T = bench.change_frequency, bench.T
    ce = bench.current_error.reshape(T, cf)
    idx = {f: min(int(round(f * cf)), cf - 1) for f in FRACTIONS}
    rows = []
    for t in range(T):
        seg = ce[t]
        e0, e1 = float(seg[0]), float(seg[-1])
        red = e0 - e1
        r = dict(case=case, seed=seed, grouping=grouping, mode=mode,
                 environment=t + 1, change_frequency=cf,
                 post_change_error=e0, end_of_environment_error=e1,
                 mean_error=float(seg.mean()),
                 error_reduction=float(red),
                 error_reduction_fraction=float(red / e0) if e0 > 0 else float("nan"))
        for f in FRACTIONS:
            r[f"error_at_{int(f * 100)}"] = float(seg[idx[f]])
        # late-stage improvement, defined before outcomes
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


def _run_one(cache, case, seed, grouping, mode):
    bench = GMPB(state_path(cache, case, seed))
    rng = LegacyRNG(_optimizer_seed(seed, mode))
    opt = SGOGrouping(bench, rng, POPULATION_SIZE, grouping=grouping)
    cf = bench.change_frequency
    targets = [int(round(f * cf)) for f in FRACTIONS]

    div_rows = []

    def _record(env, k, target_fe, frac_label):
        div_rows.append(dict(case=case, seed=seed, grouping=grouping, mode=mode,
                             environment=env, target_fraction=frac_label,
                             actual_fe_in_env=k,
                             actual_fraction=float(k / cf),
                             diversity=opt.diversity()))

    t0 = time.perf_counter()
    opt.initialize()
    _record(1, bench.FE % cf, targets[0], 0.0)
    nxt = 1                      # index of the next target fraction to record
    n_reeval = n_changes = 0
    while not bench.finished:
        opt.iteration()
        k = bench.FE % cf
        env = bench.env
        if bench.recent_change:
            _record(env - 1, cf, cf, 1.0)          # end of the environment
            bench.acknowledge_change()
            n_changes += 1
            fe0 = bench.FE
            if mode == "population_persistence":
                opt.reevaluate()
            elif mode == "previous_best_seeded":
                opt.initialize(seed_x=opt.best_x)
            else:
                raise ValueError(mode)
            n_reeval += bench.FE - fe0
            _record(bench.env, bench.FE % cf, targets[0], 0.0)
            nxt = 1
            continue
        while nxt < len(FRACTIONS) and k >= targets[nxt]:
            _record(env, k, targets[nxt], FRACTIONS[nxt])
            nxt += 1
    _record(bench.T, cf, cf, 1.0)                  # end of the final environment

    rec = dict(case=case, seed=seed, grouping=grouping, mode=mode,
               NP=POPULATION_SIZE, dimension=bench.d, peaks=bench.m,
               change_frequency=cf, shift_severity=bench.shift_severity,
               environments=bench.T, max_evals=bench.max_evals,
               realized_FE=bench.FE, budget_exact=bool(bench.FE == bench.max_evals),
               environments_completed=bench.env, changes_seen=n_changes,
               reevaluation_FE=n_reeval,
               offline_error=bench.offline_error,
               best_error_before_change=bench.best_error_before_change,
               mean_population_diversity=float(np.mean([r["diversity"] for r in div_rows])),
               wall_seconds=time.perf_counter() - t0)
    return rec, _env_rows(bench, case, seed, grouping, mode), div_rows


def _worker(args):
    return _run_one(*args)


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
    cache, outdir, cases, seeds, grouping, mode = sys.argv[1:7]
    nproc = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    cs = CASE_LIST if cases.upper() == "ALL" else cases.split(",")
    if "-" in seeds:
        a, b = seeds.split("-")
        sd = list(range(int(a), int(b) + 1))
    else:
        sd = list(range(1, int(seeds) + 1))
    os.makedirs(outdir, exist_ok=True)

    if grouping == "random" and mode == "population_persistence":
        # baseline reproduction through the same code path, used only to obtain the
        # fractional-error and diversity diagnostics the committed baseline lacks.
        # The offline errors it produces must equal the committed ones exactly.
        raw = os.path.join(outdir, "gmpb_baseline_repro_raw.csv")
        envp = os.path.join(outdir, "gmpb_baseline_repro_environments.csv")
        divp = os.path.join(outdir, "gmpb_baseline_repro_diversity_trace.csv")
    elif mode == "population_persistence":
        raw = os.path.join(outdir, "gmpb_fitness_group_raw.csv")
        envp = os.path.join(outdir, "gmpb_fitness_group_environments.csv")
        divp = os.path.join(outdir, "gmpb_fitness_group_diversity_trace.csv")
    else:
        raw = os.path.join(outdir, "gmpb_temporal_interaction_raw.csv")
        envp = os.path.join(outdir, "gmpb_temporal_interaction_environments.csv")
        divp = os.path.join(outdir, "gmpb_temporal_interaction_diversity_trace.csv")
        cs = [c for c in cs if c in INTERACTION_CASES]

    jobs = [(cache, c, s, grouping, mode) for c in cs for s in sd]
    have = _done(raw, ("case", "seed", "grouping", "mode"))
    todo = [j for j in jobs
            if (str(j[1]), str(j[2]), str(j[3]), str(j[4])) not in have]
    print(f"{grouping}/{mode}: {len(jobs)} jobs, {len(jobs) - len(todo)} done, "
          f"{len(todo)} to run", flush=True)

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
                if n % 20 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}, {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        for j in todo:
            rec, er, dr = _worker(j)
            _write(raw, [rec]); _write(envp, er); _write(divp, dr)
            n += 1
    print(json.dumps(dict(grouping=grouping, mode=mode, jobs=len(jobs),
                          completed_now=n,
                          wall_seconds=round(time.perf_counter() - t0, 1)), indent=1))


if __name__ == "__main__":
    main()
