#!/usr/bin/env python3
"""Phase-2B step 5-6: dense frozen-subproblem sweep with three start conditions.

Does NOT touch the submitted convergence_check.py.

Instances come from the frozen drift-strata protocol
(data/drift_strata_definition.json): 25 uniformly spaced steps per system
plus up to 8 per drift stratum, 215 instances in total, selected from drift
metrics alone before any optimizer result was seen.

Start conditions:

  cold             U0 = 0, placed in the optimizer's warm slot.
  reference_warm   the previous step's deterministic OSQP optimum, mapped into
                   the current problem by the receding-horizon shift-and-hold
                   map. At Nc=1 that map is the IDENTITY -- there is no later
                   block to move into first position -- so this is simply the
                   previous OSQP optimum, and it is described as such rather
                   than as a sequence shift.
  optimizer_warm   the SAME optimizer's own solution at step k-1, taken from a
                   matched Phase-2A Regime-A closed-loop run with the SAME seed,
                   then passed through the same shift-and-hold map. One
                   optimizer's history is never used for another, and the
                   deterministic solution is never labelled an optimizer warm
                   start.

Phase 2A stored only the first component of the applied input per step, which is
insufficient at n_u > 1, so the matched trajectories are regenerated here under
the frozen Phase-2A Regime-A protocol (same seeds, same fixed MPC, same exact FE
budget) and cached. Provenance of every optimizer warm start is recorded in the
output rows.

Outputs:
    data/frozen_sweep_v2_raw.csv
    data/frozen_sweep_v2_summary.csv
    data/frozen_sweep_v2_meta.json
    data/cache/optwarm_<system>_<alg>_<seed>.npy   (matched histories)
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(PROJECT, "data")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import (simulate, build_problem, solve_one_step,       # noqa: E402
                        make_cost)
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import osqp_reference, problem_linear_term      # noqa: E402

CONV_TOL = 1e-4
TRIALS = 20
# "Returned incumbent" tolerance: the optimizer gave back its start point to
# within solver noise. Relative to the start objective, with an absolute floor.
INCUMBENT_RTOL = 1e-12
INCUMBENT_ATOL = 1e-14


def shift_hold(U, nu, Nc):
    """Receding-horizon shift-and-hold; identity at Nc == 1."""
    seq = np.asarray(U, dtype=float).copy()
    if Nc > 1:
        seq[:-nu] = seq[nu:]
    return seq


def matched_history(key, sysdef, alg, seed, mpc, x0_eval):
    """Regenerate (and cache) the Phase-2A Regime-A closed-loop input history."""
    path = os.path.join(CACHE, f"optwarm_{key}_{alg}_{seed}.npy")
    if os.path.exists(path):
        return np.load(path)
    NP = int(P2.load_tuned_np()["selected"][key][alg]["NP"])
    params = dict(mpc)
    params.update(P2.strict_fe_params(NP, alg=alg))
    out = simulate(key, alg, params=params, seed=seed, verbose=False, x0_override=x0_eval)
    u = np.asarray(out["u_hist"], dtype=float)
    np.save(path, u)
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--systems", nargs="*", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()

    with open(os.path.join(OUT, "drift_strata_definition.json")) as fh:
        strata = json.load(fh)
    ics = P2.load_initial_conditions()
    tuned = P2.load_tuned_np()["selected"]
    systems = get_systems()
    keys = args.systems or P2.SYSTEMS

    rows = []
    for key in keys:
        sysdef = systems[key]
        mpc = P2.fixed_mpc_params(sysdef)
        Nc, nu = mpc["Nc"], sysdef.nu
        problem = build_problem(sysdef, Nc, mpc["Q_scale"], mpc["P"])
        lb, ub = problem.lb_seq, problem.ub_seq
        x0_eval = ics[key]["eval"]
        inst_meta = strata["systems"][key]["instance_details"]
        instances = strata["systems"][key]["selected_instances"]

        det = simulate(key, "QP", params=mpc, seed=11, verbose=False, x0_override=x0_eval)
        xh = det["x_hist"]

        # Matched optimizer histories, one per (alg, trial) with Phase-2A Regime-A seeds.
        hist = {}
        for alg in P2.ALGS:
            for t in range(args.trials):
                s = P2.seed_for("primary", P2.SYSTEM_INDEX[key], 0, P2.ALG_INDEX[alg], t)
                hist[(alg, t)] = matched_history(key, sysdef, alg, s, mpc, x0_eval)
        print(f"  {key}: matched histories ready ({len(hist)})")

        for k in instances:
            k = int(k)
            x = xh[:, k]
            f = problem_linear_term(problem, sysdef, x, k)
            u_star, status, _ = osqp_reference(problem, f)
            if u_star is None:
                raise RuntimeError(f"OSQP failed at {key} k={k}: {status}")
            u_star = np.clip(u_star, lb, ub)
            cost = make_cost(problem, x, k)
            J_star = float(cost(u_star))

            # Reference warm start: previous deterministic optimum, shift-and-hold.
            if k > 0:
                f_prev = problem_linear_term(problem, sysdef, xh[:, k - 1], k - 1)
                u_prev_ref, _, _ = osqp_reference(problem, f_prev)
                ref_warm = np.clip(shift_hold(np.clip(u_prev_ref, lb, ub), nu, Nc), lb, ub)
            else:
                ref_warm = np.zeros_like(lb)

            meta_k = inst_meta[str(k)]
            for alg in P2.ALGS:
                NP = int(tuned[key][alg]["NP"])
                params = dict(mpc)
                params.update(P2.strict_fe_params(NP, alg=alg))
                for start_name in ("cold", "reference_warm", "optimizer_warm"):
                    for t in range(args.trials):
                        if start_name == "cold":
                            start = np.zeros_like(lb)
                            prov = "zero sequence"
                        elif start_name == "reference_warm":
                            start = ref_warm
                            prov = ("previous OSQP optimum (identity shift at Nc=1)"
                                    if k > 0 else "zero (k=0: no previous step)")
                        else:
                            if k > 0:
                                u_hist = hist[(alg, t)]
                                start = np.clip(shift_hold(
                                    np.tile(u_hist[:, k - 1], Nc), nu, Nc), lb, ub)
                                prov = (f"matched Phase-2A Regime-A run, alg={alg}, "
                                        f"seed={P2.seed_for('primary', P2.SYSTEM_INDEX[key], 0, P2.ALG_INDEX[alg], t)}, step k-1")
                            else:
                                start = np.zeros_like(lb)
                                prov = "zero (k=0: no previous step)"
                        J_start = float(cost(start))
                        start_gap = abs(J_start - J_star) / max(1.0, abs(J_star))

                        seed = P2.seed_for("frozen_v2", P2.SYSTEM_INDEX[key],
                                           ("cold", "reference_warm",
                                            "optimizer_warm").index(start_name),
                                           P2.ALG_INDEX[alg], t)
                        np.random.seed(seed)
                        sol, J_final, _, nit, wall, nfe = solve_one_step(
                            problem, x, k, alg, params, start.copy())
                        J_final = float(J_final)
                        gap = abs(J_final - J_star) / max(1.0, abs(J_star))

                        returned_incumbent = bool(
                            abs(J_final - J_start) <= max(INCUMBENT_ATOL,
                                                          INCUMBENT_RTOL * abs(J_start)))
                        denom = abs(J_start - J_star)
                        improvement = (float((J_start - J_final) / denom)
                                       if denom > max(INCUMBENT_ATOL,
                                                      1e-12 * max(1.0, abs(J_star)))
                                       else np.nan)
                        rows.append(dict(
                            system=key, k=k, stratum=meta_k["stratum"],
                            warm_gap_norm=meta_k["warm_gap_norm"],
                            cold_gap_norm_drift=meta_k["cold_gap_norm"],
                            df_over_max=meta_k["df_over_max"],
                            dU_shift=meta_k["dU_shift"],
                            alg=alg, start=start_name, trial=t, seed=seed, NP=NP,
                            J_star=J_star, J_start=J_start, J_final=J_final,
                            start_gap_norm=start_gap, final_gap_norm=gap,
                            final_gap_abs=abs(J_final - J_star),
                            trivial_start=bool(start_gap <= CONV_TOL),
                            converged_1e4=bool(gap <= CONV_TOL),
                            control_error=float(np.linalg.norm(sol - u_star)),
                            first_control_error=float(np.linalg.norm(
                                sol[:nu] - u_star[:nu])),
                            returned_incumbent=returned_incumbent,
                            improvement_fraction=improvement,
                            time_ms=float(wall * 1e3),
                            requested_fe=int(params["fe_cap"]), realized_fe=int(nfe),
                            fe_exact=bool(int(nfe) == int(params["fe_cap"])),
                            warm_start_provenance=prov))
        print(f"  {key}: {len(instances)} instances done "
              f"({time.perf_counter()-t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "frozen_sweep_v2_raw.csv"), index=False)

    summ = (df.groupby(["system", "stratum", "alg", "start"])
              .agg(n=("final_gap_norm", "size"),
                   instances=("k", "nunique"),
                   start_gap_median=("start_gap_norm", "median"),
                   final_gap_median=("final_gap_norm", "median"),
                   final_gap_iqr=("final_gap_norm",
                                  lambda s: float(np.percentile(s, 75) - np.percentile(s, 25))),
                   conv_rate=("converged_1e4", "mean"),
                   trivial_start_rate=("trivial_start", "mean"),
                   incumbent_return_rate=("returned_incumbent", "mean"),
                   improvement_median=("improvement_fraction", "median"),
                   first_ctrl_err_median=("first_control_error", "median"),
                   time_ms_median=("time_ms", "median"),
                   fe_exact=("fe_exact", "all")).reset_index())
    summ.to_csv(os.path.join(OUT, "frozen_sweep_v2_summary.csv"), index=False)

    meta = {"instances": int(df.k.nunique()), "rows": int(len(df)),
            "trials": args.trials, "fe_budget": P2.FE_BUDGET,
            "fe_exact_everywhere": bool(df.fe_exact.all()),
            "start_conditions": ["cold", "reference_warm", "optimizer_warm"],
            "incumbent_tolerance": {"rtol": INCUMBENT_RTOL, "atol": INCUMBENT_ATOL},
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "frozen_sweep_v2_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
