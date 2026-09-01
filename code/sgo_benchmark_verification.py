#!/usr/bin/env python3
"""Phase-1 Step F: verify this repository's SGO against the original paper
(reviewer R2.7).

Protocol and reference values are transcribed from
`s41598-023-32465-z.pdf` (Azizi et al., Sci. Rep. 13:5373, 2023):

  * Table 1  (p. 8)   function names, search ranges, known minima
  * text     (p. 8)   100 dimensions, 150,000 objective evaluations,
                      1e-12 tolerance stopping criterion, 100 independent runs
  * Table 4  (p. 11)  mean and standard deviation per function per algorithm

Function DEFINITIONS are not printed in the SGO paper (it cites Jamil & Yang and
related compendia), so each function is implemented from its standard published
definition and then VALIDATED against the search range and known minimum that
the SGO paper's Table 1 does give. A function whose implementation fails that
validation is excluded and reported as excluded rather than silently adjusted.

Interpretation D1 (the direction of the winning condition) is the one ambiguity
that cannot be resolved from the pseudocode alone, so both readings are run:

  variant "literal"  -- calls code/sgo.py unchanged; the interpretation behind
                        every MPC result in the manuscript
  variant "flipped"  -- D1-alt, offensive wins when it is the better candidate

`code/sgo.py` is NOT modified: the flipped variant lives in this file as a
clearly marked copy, and a self-check asserts that the copy with the literal
setting reproduces code/sgo.py exactly on a fixed seed. That keeps the
implementation under test byte-identical to the submitted version.

Outputs (data/):
    sgo_benchmark_raw.csv, sgo_benchmark_summary.csv, sgo_benchmark_verification.json

Usage:
    python sgo_benchmark_verification.py [--runs 20] [--dim 100] [--budget 150000]
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
# Output root. Defaults to the released data/ directory; set SGO_OUTPUT_DIR to
# redirect every generated file elsewhere (e.g. a scratch directory) so that a
# smoke test cannot overwrite the committed production evidence.
OUT = os.environ.get("SGO_OUTPUT_DIR") or os.path.join(PROJECT, "data")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
from sgo import sgo as sgo_reference                                   # noqa: E402

SEED_BASE = 6_000_000
PAPER_NP = 50            # decision D8: not stated in the paper, assumed
PAPER_TOL = 1e-12


# --------------------------------------------------------------------------
# Benchmark functions. Range and known minimum are taken from Table 1 of the
# SGO paper; the algebraic form is the standard published definition.
# --------------------------------------------------------------------------
def f_ackley1(x):
    n = x.size
    return float(-20 * np.exp(-0.02 * np.sqrt(np.sum(x * x) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def f_alpine1(x):
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))


def f_chung_reynolds(x):
    s = float(np.sum(x * x))
    return s * s


def f_csendes(x):
    # x^6 (2 + sin(1/x)); the limit at x = 0 is 0, which is the global minimum.
    nz = x[x != 0.0]
    if nz.size == 0:
        return 0.0
    return float(np.sum(nz ** 6 * (2.0 + np.sin(1.0 / nz))))


def f_deb1(x):
    return float(-np.mean(np.sin(5.0 * np.pi * x) ** 6))


def f_dixon_price(x):
    i = np.arange(2, x.size + 1)
    return float((x[0] - 1) ** 2 + np.sum(i * (2 * x[1:] ** 2 - x[:-1]) ** 2))


def f_griewank(x):
    i = np.arange(1, x.size + 1)
    return float(np.sum(x * x) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def f_powell_sum(x):
    i = np.arange(1, x.size + 1)
    return float(np.sum(np.abs(x) ** (i + 1)))


def f_rastrigin(x):
    return float(np.sum(x * x - 10 * np.cos(2 * np.pi * x) + 10))


def f_quintic(x):
    return float(np.sum(np.abs(x ** 5 - 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2 - 10 * x - 4)))


def f_rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def f_salomon(x):
    r = np.sqrt(np.sum(x * x))
    return float(1 - np.cos(2 * np.pi * r) + 0.1 * r)


def f_schumer_steiglitz(x):
    return float(np.sum(x ** 4))


# id -> (name, fn, lb, ub, known_min, optimiser_value, published_mean, published_std)
# published_* transcribed from Table 4 (p. 11), SGO column.
BENCHMARKS = {
    "F1":  ("Ackley 1",          f_ackley1,          -35.0,   35.0,   0.0, 0.0,  0.00e+00, 0.00e+00),
    "F2":  ("Alpine 1",          f_alpine1,          -10.0,   10.0,   0.0, 0.0,  6.15e-08, 6.15e-07),
    "F4":  ("Chung Reynolds",    f_chung_reynolds,  -100.0,  100.0,   0.0, 0.0,  0.00e+00, 0.00e+00),
    "F5":  ("Csendes",           f_csendes,           -1.0,    1.0,   0.0, 0.0,  0.00e+00, 0.00e+00),
    "F6":  ("Deb 1",             f_deb1,              -1.0,    1.0,  -1.0, 0.1, -6.00e-01, 2.26e-01),
    "F7":  ("Dixon & Price",     f_dixon_price,      -10.0,   10.0,   0.0, None, 1.00e+00, 2.07e-04),
    "F10": ("Griewank",          f_griewank,        -100.0,  100.0,   0.0, 0.0,  0.00e+00, 0.00e+00),
    "F19": ("Powell Sum",        f_powell_sum,        -1.0,    1.0,   0.0, 0.0,  0.00e+00, 0.00e+00),
    "F20": ("Rastrigin",         f_rastrigin,        -5.12,   5.12,   0.0, 0.0,  0.00e+00, 0.00e+00),
    "F22": ("Quintic",           f_quintic,          -10.0,   10.0,   0.0, -1.0, 2.36e+01, 9.40e+01),
    "F23": ("Rosenbrock",        f_rosenbrock,       -30.0,   30.0,   0.0, 1.0,  9.61e+01, 1.38e+01),
    "F24": ("Salomon",           f_salomon,         -100.0,  100.0,   0.0, 0.0,  0.00e+00, 0.00e+00),
    "F25": ("Schumer Steiglitz", f_schumer_steiglitz, -100.0, 100.0,  0.0, 0.0,  0.00e+00, 0.00e+00),
}


def validate_functions(dim):
    """Check each implementation against Table 1's range and known minimum."""
    rows = []
    for fid, (name, fn, lb, ub, fmin, xopt, _, _) in BENCHMARKS.items():
        if xopt is None:
            rows.append(dict(fid=fid, name=name, known_min=fmin, value_at_optimiser=np.nan,
                             validated=None,
                             note="optimiser location is dimension-dependent; not validated pointwise"))
            continue
        x = np.full(dim, float(xopt))
        val = fn(x)
        ok = bool(abs(val - fmin) <= 1e-9 * max(1.0, abs(fmin)))
        rows.append(dict(fid=fid, name=name, known_min=fmin, value_at_optimiser=val,
                         validated=ok, note="" if ok else "EXCLUDED: does not attain tabulated minimum"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# SGO with the D1 switch. This mirrors code/sgo.py exactly; the ONLY behavioural
# difference is the direction of the winning comparison, selected by
# `offensive_wins_when`. code/sgo.py itself is left untouched, and
# selfcheck_variant() asserts that this copy with "def_le_off" reproduces it.
# --------------------------------------------------------------------------
def sgo_switchable(cost_fn, NP, max_iter, lb, ub, x0_warm=None, tol=1e-10,
                   stagnant_limit=15, max_evals=None, offensive_wins_when="def_le_off"):
    if NP % 2 != 0:
        NP += 1
    lb = np.asarray(lb, dtype=float); ub = np.asarray(ub, dtype=float)
    d = len(lb); m = NP // 2
    t0 = time.perf_counter(); evals = 0

    def eval_cost(x):
        nonlocal evals
        evals += 1
        return float(cost_fn(np.asarray(x, dtype=float)))

    def budget_left():
        return max_evals is None or evals < max_evals

    def offensive_wins(ws_off, ws_def):
        if offensive_wins_when == "def_le_off":     # literal pseudocode (D1)
            return ws_def <= ws_off
        return ws_def >= ws_off                     # D1-alt

    X = lb + np.random.rand(NP, d) * (ub - lb)
    if x0_warm is not None:
        X[0] = np.clip(x0_warm, lb, ub)
    costs = np.empty(NP)
    for i in range(NP):
        costs[i] = eval_cost(X[i])
    bi = int(np.argmin(costs)); BS = X[bi].copy(); best_cost = float(costs[bi])
    cost_hist = []; stagnant = 0; prev_best = best_cost

    for it in range(max_iter):
        if not budget_left(): break
        perm = np.random.permutation(NP); off_idx = perm[:m]; def_idx = perm[m:]
        X_off = X[off_idx].copy(); X_def = X[def_idx].copy()
        c_off = costs[off_idx].copy(); c_def = costs[def_idx].copy()
        DG = X_def.mean(axis=0); OG = X_off.mean(axis=0)
        sog_slots = []; sog_pos = []; sog_cost = []; sdg_pos = []

        for i in range(m):
            if not budget_left(): break
            r3 = np.random.randint(m)
            r1, r2 = np.random.rand(2)
            X_O1 = (X_off[i] + r1 * DG - r2 * X_def[r3]) / 2.0
            X_O1 = np.clip(X_O1, lb, ub)
            if offensive_wins(c_off[i], c_def[r3]):
                c_O1 = eval_cost(X_O1)
                X_best_local = X_O1.copy(); c_best_local = c_O1
                sog_temp = sog_pos + [X_O1.copy()]
                SOG_mean = np.mean(sog_temp, axis=0)
                if budget_left():
                    r1b, r2b = np.random.rand(2)
                    X_O2 = np.clip(X_O1 + r1b * SOG_mean - r2b * BS, lb, ub)
                    c_O2 = eval_cost(X_O2)
                    if c_O2 < c_best_local:
                        X_best_local = X_O2.copy(); c_best_local = c_O2
                X_off[i] = X_best_local; c_off[i] = c_best_local
                sog_slots.append(i); sog_pos.append(X_best_local.copy()); sog_cost.append(c_best_local)
            else:
                sdg_pos.append(X_def[r3].copy())
                if budget_left():
                    r4 = np.random.randint(m); r1d, r2d = np.random.rand(2)
                    X_D1 = np.clip(X_def[r3] + r1d * OG - r2d * X_off[r4], lb, ub)
                    c_D1 = eval_cost(X_D1)
                    if c_D1 < c_def[r3]:
                        X_def[r3] = X_D1; c_def[r3] = c_D1

        if sog_pos and budget_left():
            for jj, slot in enumerate(sog_slots):
                if not budget_left(): break
                X_scd = (sdg_pos[np.random.randint(len(sdg_pos))] if sdg_pos
                         else X_def[np.random.randint(m)])
                r1b, r2b = np.random.rand(2)
                X_O3 = np.clip(sog_pos[jj] + r1b * BS - r2b * X_scd, lb, ub)
                c_O3 = eval_cost(X_O3)
                if c_O3 < sog_cost[jj]:
                    X_off[slot] = X_O3; c_off[slot] = c_O3

        X[off_idx] = X_off; X[def_idx] = X_def
        costs[off_idx] = c_off; costs[def_idx] = c_def
        bi2 = int(np.argmin(costs))
        if costs[bi2] < best_cost:
            best_cost = float(costs[bi2]); BS = X[bi2].copy()
        cost_hist.append(best_cost)
        if abs(prev_best - best_cost) < tol: stagnant += 1
        else: stagnant = 0
        prev_best = best_cost
        if stagnant >= stagnant_limit: break
    return BS, best_cost, np.asarray(cost_hist), len(cost_hist), time.perf_counter() - t0


def selfcheck_variant():
    """The local copy with the literal setting must equal code/sgo.py exactly."""
    def quad(x):
        return float(np.sum((x - 0.3) ** 2))
    lb, ub = -np.ones(5), np.ones(5)
    np.random.seed(12345)
    a = sgo_reference(quad, 10, 25, lb, ub)
    np.random.seed(12345)
    b = sgo_switchable(quad, 10, 25, lb, ub, offensive_wins_when="def_le_off")
    same = bool(np.array_equal(a[0], b[0]) and a[1] == b[1] and
                np.array_equal(a[2], b[2]) and a[3] == b[3])
    return same, float(a[1]), float(b[1])


class TargetReached(Exception):
    pass


def run_one(fn, lb, ub, dim, budget, fmin, seed, variant):
    """One independent run under the paper's protocol.

    Stagnation stopping is disabled (D12); the run ends at the evaluation budget
    or when the 1e-12 tolerance to the known minimum is reached.
    """
    best = {"f": np.inf, "evals": 0}

    def wrapped(x):
        v = fn(np.asarray(x, dtype=float))
        best["evals"] += 1
        if v < best["f"]:
            best["f"] = v
        if v - fmin <= PAPER_TOL:
            raise TargetReached
        return v

    np.random.seed(seed)
    lbv, ubv = np.full(dim, lb), np.full(dim, ub)
    try:
        sgo_switchable(wrapped, PAPER_NP, 10 ** 9, lbv, ubv, tol=0.0,
                       stagnant_limit=10 ** 9, max_evals=budget,
                       offensive_wins_when=variant)
    except TargetReached:
        pass
    return best["f"], best["evals"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--budget", type=int, default=150000)
    ap.add_argument("--functions", nargs="*", default=None)
    args = ap.parse_args()

    ok, va, vb = selfcheck_variant()
    print(f"self-check (local copy == code/sgo.py): {ok}  ({va!r} vs {vb!r})")
    if not ok:
        raise SystemExit("ABORT: the switchable copy does not reproduce code/sgo.py")

    val = validate_functions(args.dim)
    val.to_csv(os.path.join(OUT, "sgo_benchmark_function_validation.csv"), index=False)
    print(val.to_string(index=False))
    excluded = set(val[val.validated == False].fid)                    # noqa: E712

    fids = args.functions or [f for f in BENCHMARKS if f not in excluded]
    t0 = time.perf_counter()
    rows = []
    for fid in fids:
        name, fn, lb, ub, fmin, _, pmean, pstd = BENCHMARKS[fid]
        for variant in ("def_le_off", "def_ge_off"):
            vals = []
            for r in range(args.runs):
                seed = SEED_BASE + 1_000_000 * list(BENCHMARKS).index(fid) \
                       + 1_000 * (variant == "def_ge_off") + r
                f, ne = run_one(fn, lb, ub, args.dim, args.budget, fmin, seed, variant)
                vals.append(f)
                rows.append(dict(fid=fid, name=name, variant=variant, run=r,
                                 best=f, evals=ne, dim=args.dim, budget=args.budget,
                                 NP=PAPER_NP, known_min=fmin,
                                 published_mean=pmean, published_std=pstd))
            v = np.asarray(vals)
            print(f"  {fid:4s} {name:18s} {variant:10s} "
                  f"mean {v.mean():.3e} std {v.std(ddof=0):.3e} "
                  f"| published mean {pmean:.3e} std {pstd:.3e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sgo_benchmark_raw.csv"), index=False)

    summ = (df.groupby(["fid", "name", "variant"])
              .agg(runs=("best", "size"), our_best=("best", "min"),
                   our_mean=("best", "mean"), our_std=("best", "std"),
                   our_median=("best", "median"), mean_evals=("evals", "mean"),
                   known_min=("known_min", "first"),
                   published_mean=("published_mean", "first"),
                   published_std=("published_std", "first")).reset_index())
    # Agreement measure that works when the published value is exactly 0.
    summ["abs_diff_mean"] = (summ.our_mean - summ.published_mean).abs()
    summ["rel_diff_mean"] = summ.abs_diff_mean / summ.published_mean.abs().clip(lower=1e-12)
    summ["order_of_magnitude_diff"] = np.log10(
        (summ.our_mean - summ.known_min).abs().clip(lower=1e-300) /
        (summ.published_mean - summ.known_min).abs().clip(lower=1e-300))
    summ.to_csv(os.path.join(OUT, "sgo_benchmark_summary.csv"), index=False)

    verdict = {
        "protocol": {"dim": args.dim, "budget_evals": args.budget, "runs": args.runs,
                     "NP": PAPER_NP, "tolerance": PAPER_TOL,
                     "paper_runs": 100,
                     "deviation_from_paper": ("runs reduced from 100 to %d for tractability"
                                              % args.runs) if args.runs != 100 else "none"},
        "selfcheck_local_copy_matches_sgo_py": ok,
        "functions_validated": int((val.validated == True).sum()),      # noqa: E712
        "functions_excluded": sorted(excluded),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(OUT, "sgo_benchmark_verification.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
