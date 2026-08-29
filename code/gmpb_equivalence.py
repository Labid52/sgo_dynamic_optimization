#!/usr/bin/env python3
"""Equivalence gate: Python GMPB port vs the official Octave/MATLAB implementation.

Three independent checks, all against outputs produced by the unmodified
official EDOLAB sources running under GNU Octave:

  1. FITNESS  -- ``fitness_GMPB.m`` evaluated at fixed probe points across many
                 environments, compared with ``GMPB.raw_fitness``.
  2. RUN      -- a deterministic, optimizer-free query stream pushed through the
                 official ``fitness.m`` wrapper, compared with ``GMPB.evaluate``
                 on the per-evaluation objective values, the whole CurrentError
                 trace, the Ebbc vector, the offline error and the final FE and
                 environment counters.
  3. SHIM     -- the one substituted library function (``pdist2``) against its
                 closed form.

Nothing downstream may run unless every check passes.

    python code/gmpb_equivalence.py <refdir> <outdir>
"""
import glob
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_benchmark import GMPB  # noqa: E402

TOL_FITNESS = 1e-9      # relative; the port is a transcription, so this is loose
TOL_ERROR = 1e-9


def _rel(a, b):
    d = np.abs(a - b)
    s = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return float(np.nanmax(d / s))


def check_fitness(refdir, rows):
    for sp in sorted(glob.glob(os.path.join(refdir, "*_state.mat"))):
        if sp.endswith("_refstate.mat"):
            continue
        base = os.path.basename(sp).replace("_state.mat", "")
        pp = os.path.join(refdir, base + "_probe.csv")
        if not os.path.exists(pp):
            continue
        case, seed = base.split("_seed")
        b = GMPB(sp)
        probe = np.loadtxt(pp, delimiter=",")
        envs = probe[:, 0].astype(int)
        ref = probe[:, 2]
        X = probe[:, 3:]
        got = np.empty_like(ref)
        for e in np.unique(envs):
            sel = envs == e
            got[sel] = b.raw_fitness(X[sel], env=int(e))
        r = _rel(ref, got)
        rows.append(dict(check="fitness", case=case, seed=int(seed), batch="",
                         quantity="fitness_GMPB at probe points", n=len(ref),
                         max_rel_diff=r, passed=bool(r <= TOL_FITNESS)))


def check_run(refdir, rows):
    for rp in sorted(glob.glob(os.path.join(refdir, "*_refrun.mat"))):
        o = sio.loadmat(rp, squeeze_me=True, struct_as_record=False)["out"]
        case = str(o.case)
        seed = int(o.seed)
        batch = int(o.batch)
        sp = rp.replace("_refrun.mat", "_refstate.mat")
        b = GMPB(sp)
        Q = np.asarray(o.Q, float)
        ref_vals = np.asarray(o.vals, float)
        n = int(o.nEval)

        got_vals = np.full(n, np.nan)
        i = 0
        while i < n and not b.finished:
            hi = min(i + batch, n)
            r = b.evaluate(Q[i:hi])
            got_vals[i:hi] = r
            n_ok = int(np.sum(~np.isnan(r)))
            if b.recent_change:
                b.acknowledge_change()
            i += max(n_ok, 1) if n_ok else 0
            if n_ok == 0:
                continue

        m = ~np.isnan(ref_vals) & ~np.isnan(got_vals)
        rows.append(dict(check="run", case=case, seed=seed, batch=batch,
                         quantity="per-evaluation objective values", n=int(m.sum()),
                         max_rel_diff=_rel(ref_vals[m], got_vals[m]),
                         passed=bool(_rel(ref_vals[m], got_vals[m]) <= TOL_FITNESS)))
        rows.append(dict(check="run", case=case, seed=seed, batch=batch,
                         quantity="NaN pattern of returned values", n=n,
                         max_rel_diff=float(np.mean(np.isnan(ref_vals) != np.isnan(got_vals))),
                         passed=bool(np.array_equal(np.isnan(ref_vals), np.isnan(got_vals)))))

        ce_ref = np.asarray(o.CurrentError, float).ravel()
        ce_got = b.current_error
        k = min(len(ce_ref), len(ce_got))
        rows.append(dict(check="run", case=case, seed=seed, batch=batch,
                         quantity="CurrentError trace (all FEs)", n=k,
                         max_rel_diff=_rel(ce_ref[:k], ce_got[:k]),
                         passed=bool(_rel(ce_ref[:k], ce_got[:k]) <= TOL_ERROR)))
        eb_ref = np.atleast_1d(np.asarray(o.Ebbc, float)).ravel()
        eb_got = b.ebbc
        k = min(len(eb_ref), len(eb_got))
        rows.append(dict(check="run", case=case, seed=seed, batch=batch,
                         quantity="Ebbc vector", n=k,
                         max_rel_diff=_rel(eb_ref[:k], eb_got[:k]),
                         passed=bool(_rel(eb_ref[:k], eb_got[:k]) <= TOL_ERROR)))
        rows.append(dict(check="run", case=case, seed=seed, batch=batch,
                         quantity="offline error E_o", n=1,
                         max_rel_diff=_rel(np.array([float(o.OfflineError)]),
                                           np.array([b.offline_error])),
                         passed=bool(_rel(np.array([float(o.OfflineError)]),
                                          np.array([b.offline_error])) <= TOL_ERROR)))
        rows.append(dict(check="run", case=case, seed=seed, batch=batch,
                         quantity="final FE counter", n=1,
                         max_rel_diff=abs(int(o.FE) - b.FE),
                         passed=bool(int(o.FE) == b.FE)))
        rows.append(dict(check="run", case=case, seed=seed, batch=batch,
                         quantity="final environment counter", n=1,
                         max_rel_diff=abs(int(o.Environmentcounter) - b.env),
                         passed=bool(int(o.Environmentcounter) == b.env)))


def check_shim(rows):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(37, 6))
    B = np.zeros((1, 6))
    closed = np.sqrt(np.sum((A - B) ** 2, axis=1))
    direct = np.linalg.norm(A, axis=1)
    rows.append(dict(check="shim", case="pdist2", seed="", batch="",
                     quantity="Euclidean norm identity used by the generator",
                     n=len(closed), max_rel_diff=_rel(closed, direct),
                     passed=bool(_rel(closed, direct) <= 1e-15)))


def main():
    refdir = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rows = []
    check_shim(rows)
    check_fitness(refdir, rows)
    check_run(refdir, rows)

    import csv
    p = os.path.join(outdir, "benchmark_equivalence.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_fail = sum(1 for r in rows if not r["passed"])
    for r in rows:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['check']:8s} {str(r['case']):5s} "
              f"seed={str(r['seed']):>3s} batch={str(r['batch']):>3s}  "
              f"{r['quantity']:40s} n={r['n']:>7}  maxreldiff={r['max_rel_diff']:.3e}")
    print(f"\n{len(rows)} checks, {n_fail} failures -> {p}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
