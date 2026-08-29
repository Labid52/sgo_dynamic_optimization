#!/usr/bin/env python3
"""Fidelity gate: the dynamic SGO driver must be the published SGO.

Runs ``code/sgo.py``'s ``sgo()`` (the implementation used for every result in
the manuscript, verified against Azizi et al. 2023 on 11 of 13 benchmark
functions) and the dynamic driver ``code/gmpb_optimizers.py:SGO`` on the same
static objective, from the same seed, and compares the complete sequence of
evaluated points and objective values.  If the driver had altered any operator,
introduced a new one, or changed the order in which random numbers are drawn,
the sequences would diverge.

    python code/gmpb_sgo_fidelity.py <outdir>
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgo as sgo_mod                                   # noqa: E402
from gmpb_optimizers import SGO, LegacyRNG              # noqa: E402


class _StaticProblem:
    """Minimal problem object with no environmental change, for the gate only."""

    def __init__(self, fn, d, lb, ub):
        self.fn = fn
        self.d = d
        self.lb = lb
        self.ub = ub
        self.change_frequency = 10 ** 12
        self.FE = 0
        self.recent_change = False
        self.log = []

    @property
    def finished(self):
        return False

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty(len(X))
        for i, x in enumerate(X):
            v = float(self.fn(x))
            self.log.append((x.copy(), v))
            self.FE += 1
            out[i] = -v          # driver minimises -fitness
        return out


def _objectives():
    def sphere(x):
        return float(np.sum(x ** 2))

    def rosen(x):
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))

    def rastrigin(x):
        return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))

    return {"sphere": sphere, "rosenbrock": rosen, "rastrigin": rastrigin}


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for name, fn in _objectives().items():
        for d, NP, seed, iters in [(5, 10, 11, 6), (5, 20, 12, 5),
                                   (10, 20, 13, 5), (20, 30, 14, 4)]:
            lb = np.full(d, -50.0)
            ub = np.full(d, 50.0)

            # ---- published implementation -------------------------------
            ref = []
            np.random.seed(seed)

            def logged(x, _ref=ref):
                v = fn(np.asarray(x, dtype=float))
                _ref.append((np.asarray(x, dtype=float).copy(), float(v)))
                return v

            sgo_mod.sgo(logged, NP, iters, lb, ub, tol=0.0,
                        stagnant_limit=10 ** 9, max_evals=None)

            # ---- dynamic driver -----------------------------------------
            prob = _StaticProblem(fn, d, -50.0, 50.0)
            opt = SGO(prob, LegacyRNG(seed), NP)
            opt.initialize()
            for _ in range(iters):
                opt.iteration()
            got = prob.log

            n = min(len(ref), len(got))
            same_len = len(ref) == len(got)
            dx = max((float(np.max(np.abs(ref[i][0] - got[i][0]))) for i in range(n)),
                     default=0.0)
            dv = max((abs(ref[i][1] - got[i][1]) for i in range(n)), default=0.0)
            rows.append(dict(objective=name, d=d, NP=NP, seed=seed, iterations=iters,
                             n_evals_reference=len(ref), n_evals_driver=len(got),
                             same_evaluation_count=same_len,
                             max_abs_diff_point=dx, max_abs_diff_value=dv,
                             passed=bool(same_len and dx == 0.0 and dv == 0.0)))

    p = os.path.join(outdir, "sgo_fidelity.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    nf = sum(1 for r in rows if not r["passed"])
    for r in rows:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['objective']:11s} d={r['d']:>2} "
              f"NP={r['NP']:>2} seed={r['seed']}  evals ref={r['n_evals_reference']:>4} "
              f"drv={r['n_evals_driver']:>4}  dx={r['max_abs_diff_point']:.1e} "
              f"dv={r['max_abs_diff_value']:.1e}")
    print(f"\n{len(rows)} checks, {nf} failures -> {p}")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
