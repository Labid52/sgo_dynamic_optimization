#!/usr/bin/env python3
"""Phase-2A step 6: verify exact objective-call-level FE budgeting (reviewer R2.6).

The submitted "equal-FE" mode was not equal: every optimizer shared a stagnation
rule, so realized evaluation counts inside a nominal 1000-evaluation budget
ranged from 341 to 796 depending on the algorithm. Reviewer 2 asked for strictly
identical budgets with no algorithm gaining an implicit advantage from early
stopping.

Strict-FE, as implemented in sim_common.CountedCost:
  * every objective call increments one counter, including the initial
    population sweep;
  * call number B+1 raises instead of being evaluated, so an optimizer that
    proposes more candidates than the remaining budget allows gets only the
    allowable ones evaluated;
  * the best point actually evaluated is returned;
  * stagnation stopping is disabled (tol = 0, stagnant_limit = infinity).

This script checks the property empirically for every optimizer at several
budgets on real MPC subproblems, and records any shortfall rather than hiding it.
A shortfall (realized < B) is possible in principle when an algorithm's outer
loop terminates for a reason other than the budget; an OVERSHOOT (realized > B)
would be a defect and is a stop condition.

Outputs:
    data/strict_fe_budget_verification.csv
    data/strict_fe_budget_verification.json

Usage:
    python strict_fe_verification.py [--budgets 200 500 1000 2000] [--trials 5]
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

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, build_problem, solve_one_step         # noqa: E402

ALGS = ["SGO", "GWO", "PSO", "WOA"]
SEED_BASE = 7_100_000


import phase2_protocol as P2


def strict_fe_params(NP, budget, alg=None):
    """Delegate to the frozen protocol, so this verification tests exactly the
    sizing rule the production experiments use rather than a private copy."""
    return P2.strict_fe_params(NP, budget, alg=alg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="*", default=[200, 500, 1000, 2000])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--NP", type=int, nargs="*", default=[8, 10, 14, 20, 30])
    args = ap.parse_args()
    t0 = time.perf_counter()

    rows = []
    for key, sysdef in get_systems().items():
        problem = build_problem(sysdef, 1, 1.0, sysdef.P)
        xh = simulate(key, "QP", params={"Nc": 1, "Q_scale": 1.0}, seed=11,
                      verbose=False)["x_hist"]
        k = 0
        x = xh[:, k]
        for NP in args.NP:
            for B in args.budgets:
                for ai, alg in enumerate(ALGS):
                    p = strict_fe_params(NP, B, alg=alg)
                    for t in range(args.trials):
                        np.random.seed(SEED_BASE + 1_000 * ai + t)
                        sol, J, conv, nit, wall, nfe = solve_one_step(
                            problem, x, k, alg, p, None)
                        rows.append(dict(system=key, alg=alg, NP=NP, requested_FE=B,
                                         realized_FE=int(nfe), trial=t,
                                         difference=int(nfe) - B,
                                         overshoot=max(0, int(nfe) - B),
                                         shortfall=max(0, B - int(nfe)),
                                         J=float(J), interrupted=bool(nit == -1),
                                         termination=("budget cap hit mid-iteration"
                                                      if nit == -1 else
                                                      "optimizer loop ended at budget")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "strict_fe_budget_verification.csv"), index=False)

    summ = (df.groupby(["alg", "requested_FE"])
              .agg(n=("realized_FE", "size"),
                   realized_min=("realized_FE", "min"),
                   realized_max=("realized_FE", "max"),
                   max_overshoot=("overshoot", "max"),
                   max_shortfall=("shortfall", "max"),
                   exact_fraction=("difference", lambda s: float((s == 0).mean()))
                   ).reset_index())
    summ.to_csv(os.path.join(OUT, "strict_fe_budget_verification_summary.csv"), index=False)

    verdict = {
        "budgets": args.budgets, "NP": args.NP, "trials_per_cell": args.trials,
        "systems": sorted(df.system.unique().tolist()),
        "cells": int(len(df)),
        "any_overshoot": bool((df.overshoot > 0).any()),
        "max_overshoot": int(df.overshoot.max()),
        "max_shortfall": int(df.shortfall.max()),
        "fraction_exactly_on_budget": float((df.difference == 0).mean()),
        "shortfall_by_alg": {a: int(g.shortfall.max()) for a, g in df.groupby("alg")},
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    verdict["verdict"] = ("PASS - no optimizer exceeded its budget"
                          if not verdict["any_overshoot"] else
                          "FAIL - budget exceeded; STOP before primary experiments")
    with open(os.path.join(OUT, "strict_fe_budget_verification.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(summ.to_string(index=False))
    print(json.dumps(verdict, indent=2))
    return 0 if not verdict["any_overshoot"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
