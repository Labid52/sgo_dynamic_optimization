#!/usr/bin/env python3
"""Phase-2A steps 8-10: the fixed-MPC primary experiment.

This replaces the submitted jointly tuned study as the primary evidence, and
removes the confound reviewer 2 identified (R2.5): the MPC design and the
optimizer parameters were previously tuned together on the evaluation trajectory,
so no result could be attributed to the optimizer.

Two controlled regimes, both with the MPC formulation frozen and identical for
every optimizer and for the deterministic reference:

  Regime A "A_tuned_NP"   population size per optimizer, selected using the
                          held-out tuning IC only; exact common FE budget B.
  Regime B "B_common_NP"  common NP=20 for all optimizers; exact common FE
                          budget B. Separates "equal total evaluation resources"
                          from "equal population structure".

In both regimes early stopping and stagnation stopping are disabled and the
budget is enforced at the objective-call level, so realized evaluations are
exactly B for every optimizer at every MPC step.

The deterministic reference is computed per system with OSQP as primary solver
and multistart L-BFGS-B as an independent cross-check (R2.17, R2.18); both are
scored on the implemented objective and certified by the same strong-convexity
suboptimality bound. If they disagree materially the system is flagged and its
optimality gaps are not reported.

Outputs:
    data/primary_closed_loop_raw.csv     one row per stochastic run
    data/primary_closed_loop_summary.csv per (system, optimizer, regime)
    data/primary_per_step.csv            one row per MPC step per run
    data/primary_reference_crosscheck.csv

Usage:
    python primary_experiment.py [--runs 20] [--systems ...] [--per-step-seeds 3]
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
from sim_common import (simulate, build_problem, normalized_rmse,      # noqa: E402
                        make_cost)
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import deterministic_reference                  # noqa: E402

# A material disagreement between the two deterministic solvers, relative to the
# normalisation used for optimality gaps. Fixed before running.
REF_DISAGREEMENT_TOL = 1e-6


def reference_crosscheck(key, sysdef, x0_eval, n_instants=25):
    """OSQP vs multistart L-BFGS-B along the deterministic trajectory."""
    mpc = P2.fixed_mpc_params(sysdef)
    problem = build_problem(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"])
    traj = simulate(key, "QP", params=mpc, seed=11, verbose=False, x0_override=x0_eval)
    xh, Nsim = traj["x_hist"], traj["Nsim"]
    steps = sorted(set(np.linspace(0, Nsim - 1, n_instants).astype(int).tolist()))
    rows = []
    # The pendulum-cart soft penalty is certified globally inactive over the
    # admissible input box (pendcart_penalty_certificate.py), so every system's
    # subproblems here are genuine bound-constrained QPs and OSQP is admissible.
    for k in steps:
        _, J_ref, rec = deterministic_reference(problem, sysdef, xh[:, k], int(k),
                                                allow_osqp=True, n_restarts=20,
                                                seed=900 + int(k))
        rec.update(system=key, step=int(k), P=mpc["P"], Nc=mpc["Nc"],
                   Q_scale=mpc["Q_scale"])
        rows.append(rec)
    return rows


def run_regime(key, sysdef, regime, x0_eval, tuned_np, n_runs, per_step_seeds):
    mpc = P2.fixed_mpc_params(sysdef)
    inst = P2.SYSTEM_INDEX[key]
    ri = P2.REGIME_INDEX[regime]
    problem = build_problem(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"])
    dt = float(sysdef.dt)

    # Deterministic reference closed loop under the SAME fixed MPC.
    det = simulate(key, "QP", params=mpc, seed=11, verbose=False, x0_override=x0_eval)
    det_nrmse = normalized_rmse(det["x_hist"], sysdef, x0=x0_eval)
    det_u = det["u_hist"]

    run_rows, step_rows = [], []
    for alg in P2.ALGS:
        ai = P2.ALG_INDEX[alg]
        NP = P2.COMMON_NP if regime == "B_common_NP" else int(tuned_np[key][alg]["NP"])
        params = dict(mpc)
        params.update(P2.strict_fe_params(NP, alg=alg))
        for t in range(n_runs):
            seed = P2.seed_for("primary", inst, ri, ai, t)
            out = simulate(key, alg, params=params, seed=seed, verbose=False,
                           x0_override=x0_eval)
            nr = normalized_rmse(out["x_hist"], sysdef, x0=x0_eval)
            tms = np.asarray(out["time_ms"], dtype=float)
            nfe = np.asarray(out["nfe"], dtype=float)
            u0_err = float(np.mean(np.linalg.norm(
                out["u_hist"][:, :det_u.shape[1]] - det_u, axis=0)))
            run_rows.append(dict(
                system=key, alg=alg, regime=regime, seed=seed, trial=t,
                ic="eval", NP=NP, fe_budget=params["fe_cap"],
                maxIter_derived=params["maxIter"],
                realized_nfe_total=float(np.sum(nfe)),
                realized_nfe_per_step_mean=float(np.mean(nfe)),
                realized_nfe_per_step_min=float(np.min(nfe)),
                realized_nfe_per_step_max=float(np.max(nfe)),
                nfe_exactly_on_budget=bool(np.all(nfe == params["fe_cap"])),
                nrmse=nr,
                deterministic_reference_nrmse=det_nrmse,
                nrmse_minus_reference=nr - det_nrmse,
                mean_first_input_error_vs_reference=u0_err,
                time_ms_mean=float(np.mean(tms)), time_ms_median=float(np.median(tms)),
                time_ms_p95=float(np.percentile(tms, 95)), time_ms_max=float(np.max(tms)),
                dt_s=dt, p95_time_over_dt=float(np.percentile(tms, 95) / (dt * 1e3)),
                max_time_over_dt=float(np.max(tms) / (dt * 1e3)),
                Nsim=int(out["Nsim"]), P=mpc["P"], Nc=mpc["Nc"], Q_scale=mpc["Q_scale"],
                termination="exact FE budget; early stopping and stagnation disabled"))
            # Per-step detail for a subset of seeds (full per-step logging for all
            # runs would be ~1.5M rows for HVAC alone).
            if t < per_step_seeds:
                xh, uh = out["x_hist"], out["u_hist"]
                for k in range(int(out["Nsim"])):
                    step_rows.append(dict(
                        system=key, alg=alg, regime=regime, seed=seed, trial=t, k=k,
                        time_ms=float(tms[k]), nfe=float(nfe[k]),
                        objective=float(out["cost"][k]),
                        first_input=float(uh[0, k]),
                        reference_first_input=float(det_u[0, k]),
                        first_input_error=float(np.linalg.norm(uh[:, k] - det_u[:, k])),
                        state_norm=float(np.linalg.norm(xh[:, k])),
                        dt_s=dt))
        print(f"    {key:9s} {regime:12s} {alg:4s} NP={NP:3d} "
              f"nRMSE {np.mean([r['nrmse'] for r in run_rows if r['alg']==alg]):.6e}")
    return run_rows, step_rows, det_nrmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=P2.PRIMARY_RUNS)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--per-step-seeds", type=int, default=3)
    ap.add_argument("--skip-crosscheck", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    ics = P2.load_initial_conditions()
    tuned_np = P2.load_tuned_np()["selected"]
    systems = get_systems()
    keys = args.systems or P2.SYSTEMS

    cross_rows = []
    if not args.skip_crosscheck:
        print("deterministic reference cross-check (OSQP vs multistart L-BFGS-B)")
        for key in keys:
            cross_rows += reference_crosscheck(key, systems[key], ics[key]["eval"])
        cdf = pd.DataFrame(cross_rows)
        cdf.to_csv(os.path.join(OUT, "primary_reference_crosscheck.csv"), index=False)
        bad = cdf[cdf.obj_rel_diff > REF_DISAGREEMENT_TOL]
        for key, g in cdf.groupby("system"):
            print(f"  {key:9s} max |J_OSQP - J_LBFGSB| = {g.obj_abs_diff.max():.3e} "
                  f"(rel {g.obj_rel_diff.max():.3e}), max sol diff "
                  f"{g.sol_inf_diff.max():.3e}, max subopt bound "
                  f"{g.subopt_bound_norm.max():.3e}")
        if len(bad):
            print(f"WARNING: {len(bad)} instances exceed the reference-disagreement "
                  f"tolerance {REF_DISAGREEMENT_TOL:g}; systems: "
                  f"{sorted(bad.system.unique())}")

    all_runs, all_steps = [], []
    for key in keys:
        print(f"primary experiment: {key}")
        for regime in ["A_tuned_NP", "B_common_NP"]:
            r, s, _ = run_regime(key, systems[key], regime, ics[key]["eval"],
                                 tuned_np, args.runs, args.per_step_seeds)
            all_runs += r
            all_steps += s

    df = pd.DataFrame(all_runs)
    df.to_csv(os.path.join(OUT, "primary_closed_loop_raw.csv"), index=False)
    pd.DataFrame(all_steps).to_csv(os.path.join(OUT, "primary_per_step.csv"), index=False)

    summ = (df.groupby(["system", "regime", "alg"])
              .agg(runs=("nrmse", "size"), NP=("NP", "first"),
                   nrmse_mean=("nrmse", "mean"), nrmse_std=("nrmse", "std"),
                   nrmse_median=("nrmse", "median"),
                   nrmse_min=("nrmse", "min"), nrmse_max=("nrmse", "max"),
                   reference_nrmse=("deterministic_reference_nrmse", "first"),
                   nfe_per_step_mean=("realized_nfe_per_step_mean", "mean"),
                   nfe_exact=("nfe_exactly_on_budget", "all"),
                   time_ms_mean=("time_ms_mean", "mean"),
                   time_ms_median=("time_ms_median", "mean"),
                   time_ms_p95=("time_ms_p95", "mean"),
                   time_ms_max=("time_ms_max", "max"),
                   dt_s=("dt_s", "first"),
                   p95_over_dt=("p95_time_over_dt", "mean"),
                   max_over_dt=("max_time_over_dt", "max"),
                   first_input_err=("mean_first_input_error_vs_reference", "mean")
                   ).reset_index())
    summ["rank_in_system_regime"] = (summ.groupby(["system", "regime"])
                                     .nrmse_mean.rank(method="min").astype(int))
    summ.to_csv(os.path.join(OUT, "primary_closed_loop_summary.csv"), index=False)

    meta = {"runs_per_cell": args.runs, "fe_budget": P2.FE_BUDGET,
            "regimes": list(P2.REGIME_INDEX),
            "all_runs_exactly_on_budget": bool(df.nfe_exactly_on_budget.all()),
            "systems": keys, "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "primary_experiment_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
