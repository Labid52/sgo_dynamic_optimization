#!/usr/bin/env python3
"""Gate: the ablation driver with grouping="random" must be the validated SGO.

Runs ``code/gmpb_optimizers.py:SGO`` and ``code/gmpb_sgo_mechanism.py:SGOGrouping``
with ``grouping="random"`` on the same official GMPB instances, from the same
optimizer seed, and compares them evaluation by evaluation.  Any change to an
operator, to a constant, or to the order in which random numbers are drawn would
show up here.

    python code/gmpb_mechanism_equivalence.py <cache> <outdir>

Nothing downstream may run unless every check passes.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import GMPB, state_path            # noqa: E402
from gmpb_optimizers import SGO, LegacyRNG             # noqa: E402
from gmpb_sgo_mechanism import SGOGrouping             # noqa: E402
import gmpb_run                                        # noqa: E402


class _Recording(GMPB):
    """GMPB that also records every point it is asked to evaluate."""

    def __init__(self, path):
        super().__init__(path)
        self.log_x = []
        self.log_f = []

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        before = self.FE
        out = super().evaluate(X)
        n = self.FE - before
        if n:
            self.log_x.append(np.array(X[:n], copy=True))
            self.log_f.append(np.array(out[:n], copy=True))
        return out


def _run(cls, path, seed, NP, n_env, mode, **kw):
    """Run one optimizer for n_env environments under the given temporal mode."""
    b = _Recording(path)
    b.T = min(b.T, n_env)
    b.max_evals = b.change_frequency * b.T
    b.reset()
    opt = cls(b, LegacyRNG(seed), NP, **kw)
    opt.initialize()
    while not b.finished:
        opt.iteration()
        if b.recent_change:
            b.acknowledge_change()
            if mode == "population_persistence":
                opt.reevaluate()
            else:
                opt.initialize(seed_x=opt.best_x)
    return b, opt


def main():
    cache, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rows = []
    NP = gmpb_run.POPULATION_SIZE
    specs = []
    for case in ["F2", "F8", "F10", "F12"]:
        for seed in (1, 2, 7):
            for mode in ("population_persistence", "previous_best_seeded"):
                specs.append((case, seed, mode))
    # a couple of extra cases for breadth over peaks and dimension
    for case, seed, mode in [("F1", 3, "population_persistence"),
                             ("F5", 4, "population_persistence"),
                             ("F9", 5, "population_persistence"),
                             ("F7", 6, "previous_best_seeded")]:
        specs.append((case, seed, mode))

    for case, seed, mode in specs:
        path = state_path(cache, case, seed)
        oseed = (gmpb_run.OPTIMIZER_SEED_BASE + 1000 * seed
                 + 10 * gmpb_run.ALG_OFFSET["SGO"] + gmpb_run.MODE_OFFSET[mode])
        n_env = 4 if case != "F8" else 8
        ba, oa = _run(SGO, path, oseed, NP, n_env, mode)
        bb, ob = _run(SGOGrouping, path, oseed, NP, n_env, mode, grouping="random")

        xa = np.concatenate(ba.log_x) if ba.log_x else np.zeros((0, ba.d))
        xb = np.concatenate(bb.log_x) if bb.log_x else np.zeros((0, bb.d))
        fa = np.concatenate(ba.log_f) if ba.log_f else np.zeros(0)
        fb = np.concatenate(bb.log_f) if bb.log_f else np.zeros(0)
        n = min(len(fa), len(fb))

        checks = [
            ("evaluated points", float(np.max(np.abs(xa[:n] - xb[:n]))) if n else 0.0,
             len(xa) == len(xb)),
            ("objective values", float(np.max(np.abs(fa[:n] - fb[:n]))) if n else 0.0,
             len(fa) == len(fb)),
            ("FE counter", float(abs(ba.FE - bb.FE)), ba.FE == bb.FE),
            ("environment counter", float(abs(ba.env - bb.env)), ba.env == bb.env),
            ("population after run", float(np.max(np.abs(oa.X - ob.X))), True),
            ("population costs", float(np.max(np.abs(oa.costs - ob.costs))), True),
            ("best solution", float(np.max(np.abs(oa.best_x - ob.best_x))), True),
            ("best cost", float(abs(oa.best_cost - ob.best_cost)), True),
            ("CurrentError trace",
             float(np.max(np.abs(ba.current_error - bb.current_error))), True),
            ("Ebbc vector", float(np.nanmax(np.abs(ba.ebbc - bb.ebbc))), True),
            ("offline error", float(abs(ba.offline_error - bb.offline_error)), True),
        ]
        for name, diff, lens_ok in checks:
            rows.append(dict(case=case, seed=seed, temporal_mode=mode,
                             environments=n_env, optimizer_seed=oseed,
                             quantity=name, n_compared=n,
                             max_abs_diff=diff,
                             passed=bool(lens_ok and diff == 0.0)))

    p = os.path.join(outdir, "baseline_equivalence.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    nf = sum(1 for r in rows if not r["passed"])
    for r in rows:
        if r["quantity"] in ("objective values", "offline error", "population after run"):
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['case']:4s} seed={r['seed']} "
                  f"{r['temporal_mode']:22s} {r['quantity']:22s} n={r['n_compared']:>6} "
                  f"maxdiff={r['max_abs_diff']:.3e}")
    print(f"\n{len(rows)} checks over {len(specs)} configurations, {nf} failures -> {p}")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
