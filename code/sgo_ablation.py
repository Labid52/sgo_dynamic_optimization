#!/usr/bin/env python3
"""Phase-2C Task 1: SGO mechanism ablation (reviewer R2.15).

Diagnostic only. No variant is proposed as a new algorithm, and the frozen SGO in
code/sgo.py is neither modified nor replaced.

Three stages:
  --equivalence   seed-matched proof that the ablation-capable copy with default
                  flags reproduces code/sgo.py exactly (solution, objective,
                  NFE, history). Must pass before anything else runs.
  --define        write the instance/variant protocol, to be committed BEFORE
                  any ablation outcome is observed.
  (default)       run the ablation.

Outputs:
    data/sgo_ablation_default_equivalence.csv
    data/sgo_ablation_protocol.json
    data/sgo_ablation_raw.csv
    data/sgo_ablation_summary.csv
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
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import (simulate, build_problem, make_cost,            # noqa: E402
                        CountedCost, BudgetExhausted)
from mpc_utils import normalize_cost                                   # noqa: E402
from sgo import sgo as sgo_frozen                                      # noqa: E402
from sgo_ablation_impl import sgo_ablatable, VARIANTS                  # noqa: E402
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import osqp_reference, problem_linear_term      # noqa: E402

SEED_BASE = 14_000_000
TRIALS = 30
PRIMARY_BUDGET = 1000
PLATEAU_BUDGET = 5000
CONV_TOL = 1e-4
INCUMBENT_RTOL, INCUMBENT_ATOL = 1e-12, 1e-14


# ---------------------------------------------------------------- equivalence
def run_equivalence(n_cases=60):
    """Frozen sgo.py vs ablation copy with default flags, on matched seeds."""
    rows = []
    rng = np.random.default_rng(7)
    for c in range(n_cases):
        d = int(rng.integers(1, 6))
        lb, ub = -np.ones(d), np.ones(d)
        A = rng.normal(size=(d, d)); H = A @ A.T + 0.5 * np.eye(d)
        f = rng.normal(size=d)
        shift = rng.normal(size=d) * 0.3

        def cost(x, H=H, f=f, shift=shift):
            z = np.asarray(x, float) - shift
            return float(0.5 * z @ H @ z + f @ z)

        NP = int(rng.choice([8, 10, 20]))
        mi = int(rng.choice([5, 15, 40]))
        warm = rng.uniform(-1, 1, d) if c % 2 == 0 else None
        cap = int(rng.choice([0, 200, 500]))
        kw = {} if cap == 0 else {"max_evals": cap}

        np.random.seed(1234 + c)
        a = sgo_frozen(cost, NP, mi, lb, ub, x0_warm=warm, **kw)
        np.random.seed(1234 + c)
        b = sgo_ablatable(cost, NP, mi, lb, ub, x0_warm=warm, **kw)
        same = bool(np.array_equal(a[0], b[0]) and a[1] == b[1]
                    and np.array_equal(a[2], b[2]) and a[3] == b[3])
        rows.append(dict(case=c, dim=d, NP=NP, max_iter=mi,
                         warm=warm is not None, max_evals=cap or None,
                         frozen_best=float(a[1]), ablatable_best=float(b[1]),
                         solution_identical=bool(np.array_equal(a[0], b[0])),
                         objective_identical=bool(a[1] == b[1]),
                         history_identical=bool(np.array_equal(a[2], b[2])),
                         iters_identical=bool(a[3] == b[3]), identical=same))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sgo_ablation_default_equivalence.csv"), index=False)
    ok = bool(df.identical.all())
    print(f"equivalence: {int(df.identical.sum())}/{len(df)} cases identical -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- protocol
def define_protocol():
    """Instances chosen from Phase-2B results, before any ablation outcome."""
    sweep = pd.read_csv(os.path.join(OUT, "frozen_sweep_v2_raw.csv"))
    cold = sweep[sweep.start == "cold"]
    sgo_cold = (cold[cold.alg == "SGO"].groupby(["system", "k"])
                .agg(stratum=("stratum", "first"), drift=("warm_gap_norm", "first"),
                     sgo_gap=("final_gap_norm", "median"),
                     sgo_conv=("converged_1e4", "mean"),
                     sgo_incumbent=("returned_incumbent", "mean")).reset_index())
    budget = pd.read_csv(os.path.join(OUT, "budget_sensitivity_v2_summary.csv"))
    bsgo = budget[budget.alg == "SGO"]
    plateau = []
    for (sy, k), g in bsgo.groupby(["system", "k"]):
        g = g.sort_values("budget_fe")
        lo = float(g[g.budget_fe == 1000].median_gap.iloc[0])
        hi = float(g[g.budget_fe == 5000].median_gap.iloc[0])
        # plateau := 5000 FE buys less than one order of magnitude over 1000 FE
        if hi > 0.1 * lo:
            plateau.append((sy, int(k)))
    plateau_set = set(plateau)

    chosen = []
    # 1) plateau instances (flight + UAV expected)
    for sy, k in sorted(plateau_set):
        r = sgo_cold[(sgo_cold.system == sy) & (sgo_cold.k == k)]
        if len(r):
            chosen.append((sy, k, "plateau", r.iloc[0]))
    # 2) recoverable difficult instances (CSTR / pendcart from the budget study)
    for sy, k in sorted(set(zip(bsgo.system, bsgo.k)) - plateau_set):
        r = sgo_cold[(sgo_cold.system == sy) & (sgo_cold.k == int(k))]
        if len(r):
            chosen.append((sy, int(k), "recoverable_difficult", r.iloc[0]))
    # 3) one reliable benign instance per system (SGO converges in every trial)
    for sy, g in sgo_cold.groupby("system"):
        rel = g[(g.stratum == "Benign") & (g.sgo_conv >= 0.99)]
        if len(rel):
            r = rel.nsmallest(1, "sgo_gap").iloc[0]
            chosen.append((sy, int(r.k), "benign_reliable", r))
    # 4) ensure Moderate/High representation outside the budget set
    for st in ("Moderate", "High", "Severe"):
        g = sgo_cold[sgo_cold.stratum == st]
        if len(g):
            r = g.nlargest(1, "drift").iloc[0]
            key = (r.system, int(r.k))
            if key not in [(c[0], c[1]) for c in chosen]:
                chosen.append((r.system, int(r.k), f"stratum_{st}", r))

    seen, insts = set(), []
    for sy, k, role, r in chosen:
        if (sy, k) in seen:
            continue
        seen.add((sy, k))
        insts.append(dict(system=sy, k=int(k), role=role, stratum=str(r.stratum),
                          drift_warm_gap=float(r.drift),
                          sgo_cold_gap_median=float(r.sgo_gap),
                          sgo_cold_conv=float(r.sgo_conv),
                          sgo_incumbent_return=float(r.sgo_incumbent),
                          plateau=bool((sy, k) in plateau_set)))
    payload = {
        "purpose": "Diagnostic ablation of SGO mechanisms (R2.15). No variant is "
                   "proposed as a new algorithm.",
        "frozen_sgo_untouched": True,
        "variants": {k: (v or "frozen defaults") for k, v in VARIANTS.items()},
        "variant_C_note":
            "The Phase-2C brief described the frozen grouping as cost-based and proposed "
            "random grouping as the ablation. That premise is inverted: sgo.py:50 already "
            "groups by a fresh random permutation each iteration (decision D11). The "
            "meaningful predeclared alternative is therefore the fitness-based split "
            "(best half offensive), which is D11's documented rejected alternative.",
        "harness_note":
            "Under strict FE the solve harness returns the best point actually evaluated, "
            "so disabling greedy acceptance changes the SEARCH TRAJECTORY rather than the "
            "best-found bookkeeping. Variant A therefore tests whether greedy acceptance "
            "helps the search, not whether it prevents losing good solutions.",
        "budgets": {"primary": PRIMARY_BUDGET, "plateau_only": PLATEAU_BUDGET},
        "trials": TRIALS, "seeds": "paired across variants; SEED_BASE=%d" % SEED_BASE,
        "instances": insts,
        "n_instances": len(insts),
        "systems_covered": sorted({i["system"] for i in insts}),
        "strata_covered": sorted({i["stratum"] for i in insts}),
    }
    with open(os.path.join(OUT, "sgo_ablation_protocol.json"), "w") as f:
        json.dump(payload, f, indent=2)
    for i in insts:
        print(f"  {i['system']:9s} k={i['k']:4d} {i['stratum']:9s} {i['role']:22s} "
              f"drift={i['drift_warm_gap']:.2e} SGO cold gap={i['sgo_cold_gap_median']:.2e} "
              f"conv={100*i['sgo_cold_conv']:.0f}% plateau={i['plateau']}")
    print(f"n_instances={len(insts)} systems={payload['systems_covered']}")
    return payload


# ------------------------------------------------------------------------ run
def solve_variant(problem, x, k, variant_kw, NP, budget, warm, seed):
    """One frozen-subproblem solve with an ablation variant, exact FE budget."""
    raw = make_cost(problem, x, k)
    cost_norm, lb_n, ub_n, unscale = normalize_cost(raw, problem.lb_seq, problem.ub_seq)
    counted = CountedCost(cost_norm, fe_cap=budget)
    warm_n = None
    if warm is not None:
        warm_n = (warm - problem.lb_seq) / (problem.ub_seq - problem.lb_seq) * 2.0 - 1.0
    per_iter = max(NP // 2, 1)
    max_iter = int(np.ceil(budget / per_iter)) + 5
    np.random.seed(seed)
    t0 = time.perf_counter()
    try:
        sol_n, best, _, _, _ = sgo_ablatable(
            counted, NP, max_iter, lb_n, ub_n, x0_warm=warm_n,
            tol=0.0, stagnant_limit=10 ** 9, max_evals=budget, **variant_kw)
    except BudgetExhausted:
        sol_n, best = counted.best_x, counted.best_f
    wall = time.perf_counter() - t0
    if counted.best_x is not None and float(counted.best_f) < float(cost_norm(sol_n)):
        sol_n = counted.best_x
    sol = np.clip(unscale(sol_n), problem.lb_seq, problem.ub_seq)
    return sol, float(raw(sol)), counted.count, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equivalence", action="store_true")
    ap.add_argument("--define", action="store_true")
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()

    if args.equivalence:
        raise SystemExit(0 if run_equivalence() else 1)
    if args.define:
        define_protocol()
        raise SystemExit(0)

    if not run_equivalence():
        raise SystemExit("ABORT: default-flag equivalence failed")
    with open(os.path.join(OUT, "sgo_ablation_protocol.json")) as f:
        proto = json.load(f)
    ics = P2.load_initial_conditions()
    tuned = P2.load_tuned_np()["selected"]
    systems = get_systems()

    t0 = time.perf_counter()
    rows = []
    for inst_i, inst in enumerate(proto["instances"]):
        key, k = inst["system"], int(inst["k"])
        sysdef = systems[key]
        mpc = P2.fixed_mpc_params(sysdef)
        problem = build_problem(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"])
        lb, ub = problem.lb_seq, problem.ub_seq
        xh = simulate(key, "QP", params=mpc, seed=11, verbose=False,
                      x0_override=ics[key]["eval"])["x_hist"]
        x = xh[:, k]
        f = problem_linear_term(problem, sysdef, x, k)
        u_star, status, _ = osqp_reference(problem, f)
        if u_star is None:
            raise RuntimeError(f"OSQP failed at {key} k={k}: {status}")
        u_star = np.clip(u_star, lb, ub)
        cost = make_cost(problem, x, k)
        J_star = float(cost(u_star))
        NP = int(tuned[key]["SGO"]["NP"])

        # Warm start used for the warm-slot variant: previous deterministic optimum.
        if k > 0:
            f_prev = problem_linear_term(problem, sysdef, xh[:, k - 1], k - 1)
            u_prev, _, _ = osqp_reference(problem, f_prev)
            warm = np.clip(u_prev, lb, ub)
        else:
            warm = np.zeros_like(lb)
        J_start = float(cost(warm))

        budgets = [PRIMARY_BUDGET] + ([PLATEAU_BUDGET] if inst["plateau"] else [])
        for B in budgets:
            for vi, (vname, vkw) in enumerate(VARIANTS.items()):
                for t in range(args.trials):
                    seed = SEED_BASE + 1_000_000 * inst_i + 10_000 * (B == PLATEAU_BUDGET) \
                           + 1_000 * vi + t
                    sol, J, nfe, wall = solve_variant(problem, x, k, vkw, NP, B, warm, seed)
                    gap = abs(J - J_star) / max(1.0, abs(J_star))
                    denom = abs(J_start - J_star)
                    rows.append(dict(
                        system=key, k=k, role=inst["role"], stratum=inst["stratum"],
                        drift_warm_gap=inst["drift_warm_gap"], plateau=inst["plateau"],
                        variant=vname, budget_fe=B, trial=t, seed=seed, NP=NP,
                        J_star=J_star, J_start=J_start, J=J, norm_gap=gap,
                        converged_1e4=bool(gap <= CONV_TOL),
                        first_control_error=float(np.linalg.norm(
                            sol[:sysdef.nu] - u_star[:sysdef.nu])),
                        returned_incumbent=bool(abs(J - J_start) <= max(
                            INCUMBENT_ATOL, INCUMBENT_RTOL * abs(J_start))),
                        improvement_fraction=(float((J_start - J) / denom)
                                              if denom > 1e-12 * max(1.0, abs(J_star))
                                              else np.nan),
                        realized_fe=int(nfe), fe_exact=bool(int(nfe) == B),
                        time_ms=float(wall * 1e3)))
        print(f"  {key:9s} k={k:4d} done ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sgo_ablation_raw.csv"), index=False)
    summ = (df.groupby(["system", "k", "stratum", "role", "plateau", "budget_fe", "variant"])
              .agg(n=("norm_gap", "size"), median_gap=("norm_gap", "median"),
                   q1=("norm_gap", lambda s: float(np.percentile(s, 25))),
                   q3=("norm_gap", lambda s: float(np.percentile(s, 75))),
                   conv_rate=("converged_1e4", "mean"),
                   incumbent_return=("returned_incumbent", "mean"),
                   improvement_median=("improvement_fraction", "median"),
                   first_ctrl_err=("first_control_error", "median"),
                   time_ms=("time_ms", "median"),
                   fe_exact=("fe_exact", "all")).reset_index())
    summ.to_csv(os.path.join(OUT, "sgo_ablation_summary.csv"), index=False)
    meta = {"instances": int(df.groupby(["system", "k"]).ngroups),
            "variants": list(VARIANTS), "trials": args.trials,
            "fe_exact_everywhere": bool(df.fe_exact.all()),
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "sgo_ablation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
