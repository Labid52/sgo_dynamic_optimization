#!/usr/bin/env python3
"""Phase-2A step 7: optimizer-only tuning on held-out initial conditions.

Reviewer R2.5: the submitted study tuned the MPC design (P, Nc, Q_scale) jointly
with the optimizer parameters, on the same trajectory used for evaluation. Both
problems are fixed here:

  * the MPC formulation is FROZEN (phase2_protocol.fixed_mpc_params) and is
    identical for every optimizer and for the deterministic reference. P, Nc, Q,
    R, Q_scale, bounds, reference, plant and horizon are NOT tuned.
  * tuning runs on the held-out initial condition x0_tune only; the evaluation
    initial condition x0_eval is never seen during tuning.

Only the population size is searched. Under an exact FE budget the iteration
limit is not an independent degree of freedom -- the run must consume exactly B
objective evaluations, so the iteration cap is derived from B and NP. Leaving it
free would allow a configuration to stop before exhausting B, reintroducing the
unequal-budget effect that strict FE exists to remove.

Selection metric: mean closed-loop normalized RMSE over the full simulation
length from x0_tune, averaged over TUNING_SEEDS_PER_CANDIDATE seeds. The nRMSE
scale is computed from x0_tune (not the nominal x0), otherwise the metric would
be silently rescaled.

Outputs:
    data/optimizer_only_tuning_raw.csv
    data/optimizer_only_tuning_summary.csv
    data/tuned_params_optimizer_only.json
    data/tuning_report_optimizer_only.txt

Usage:
    python tune_optimizer_only.py [--systems ...]
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
from sim_common import simulate, normalized_rmse                       # noqa: E402
import phase2_protocol as P2                                           # noqa: E402


def tune_system(key, sysdef, x0_tune, rows):
    mpc = P2.fixed_mpc_params(sysdef)
    inst = P2.SYSTEM_INDEX[key]
    best = {}
    for alg in P2.ALGS:
        ai = P2.ALG_INDEX[alg]
        for np_i, NP in enumerate(P2.TUNING_NP_GRID):
            params = dict(mpc)
            params.update(P2.strict_fe_params(NP, alg=alg))
            vals, nfes, times = [], [], []
            for t in range(P2.TUNING_SEEDS_PER_CANDIDATE):
                seed = P2.seed_for("tuning", inst, np_i, ai, t)
                out = simulate(key, alg, params=params, seed=seed, verbose=False,
                               x0_override=x0_tune)
                vals.append(normalized_rmse(out["x_hist"], sysdef, x0=x0_tune))
                nfes.append(float(np.mean(out["nfe"])))
                times.append(float(np.mean(out["time_ms"])))
                rows.append(dict(system=key, alg=alg, NP=NP,
                                 maxIter_derived=params["maxIter"],
                                 fe_budget=params["fe_cap"], seed=seed, trial=t,
                                 nrmse=vals[-1], mean_nfe_per_step=nfes[-1],
                                 mean_time_ms=times[-1],
                                 P=mpc["P"], Nc=mpc["Nc"], Q_scale=mpc["Q_scale"],
                                 ic="tune"))
            score = float(np.mean(vals))
            if alg not in best or score < best[alg]["score"]:
                best[alg] = {"NP": int(NP), "score": score,
                             "maxIter_derived": int(params["maxIter"]),
                             "mean_nfe_per_step": float(np.mean(nfes)),
                             "mean_time_ms": float(np.mean(times))}
        print(f"  {key:9s} {alg:4s} selected NP={best[alg]['NP']:3d} "
              f"(held-out nRMSE {best[alg]['score']:.6e}, "
              f"realized NFE/step {best[alg]['mean_nfe_per_step']:.1f})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()

    ics = P2.load_initial_conditions()
    systems = get_systems()
    keys = args.systems or P2.SYSTEMS

    rows, selected = [], {}
    for key in keys:
        sysdef = systems[key]
        print(f"tuning {key} (fixed MPC: P={sysdef.P}, Nc={P2.FIXED_MPC['Nc']}, "
              f"Q_scale={P2.FIXED_MPC['Q_scale']}, exact FE budget={P2.FE_BUDGET})")
        selected[key] = tune_system(key, sysdef, ics[key]["tune"], rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "optimizer_only_tuning_raw.csv"), index=False)
    summ = (df.groupby(["system", "alg", "NP"])
              .agg(mean_nrmse=("nrmse", "mean"), std_nrmse=("nrmse", "std"),
                   mean_nfe_per_step=("mean_nfe_per_step", "mean"),
                   mean_time_ms=("mean_time_ms", "mean"),
                   maxIter_derived=("maxIter_derived", "first"),
                   n=("nrmse", "size")).reset_index())
    summ["selected"] = [bool(selected[r.system][r.alg]["NP"] == r.NP)
                        for r in summ.itertuples()]
    summ.to_csv(os.path.join(OUT, "optimizer_only_tuning_summary.csv"), index=False)

    payload = {
        "protocol": {
            "mpc_frozen": True,
            "tuned_parameters": ["NP"],
            "derived_parameters": ["maxIter (from exact FE budget and NP)"],
            "not_tuned": ["P", "Nc", "Q", "R", "Q_scale", "plant", "reference", "horizon"],
            "fe_budget": P2.FE_BUDGET,
            "np_grid": P2.TUNING_NP_GRID,
            "seeds_per_candidate": P2.TUNING_SEEDS_PER_CANDIDATE,
            "selection_metric": "mean closed-loop nRMSE on the held-out tuning IC, "
                                "full simulation length",
            "initial_condition": "x0_tune only; x0_eval never used during tuning",
        },
        "selected": {k: {a: v[a] for a in P2.ALGS} for k, v in selected.items()},
    }
    with open(os.path.join(OUT, "tuned_params_optimizer_only.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    lines = ["OPTIMIZER-ONLY TUNING (fixed MPC, held-out initial condition)", "=" * 72,
             f"FE budget (exact, per MPC step): {P2.FE_BUDGET}",
             f"NP grid: {P2.TUNING_NP_GRID}",
             f"seeds per candidate: {P2.TUNING_SEEDS_PER_CANDIDATE}",
             "MPC parameters were NOT tuned: P, Nc, Q, R, Q_scale are frozen at the "
             "nominal per-system design.", ""]
    for key in keys:
        sysdef = systems[key]
        lines.append(f"{key}  (P={sysdef.P}, Nc={P2.FIXED_MPC['Nc']}, "
                     f"Q_scale={P2.FIXED_MPC['Q_scale']})")
        for alg in P2.ALGS:
            b = selected[key][alg]
            lines.append(f"  {alg:4s} NP={b['NP']:3d}  maxIter(derived)={b['maxIter_derived']:4d}"
                         f"  held-out nRMSE={b['score']:.6e}"
                         f"  realized NFE/step={b['mean_nfe_per_step']:.1f}"
                         f"  time/step={b['mean_time_ms']:.2f} ms")
        lines.append("")
    lines.append(f"wall time: {time.perf_counter()-t0:.1f} s")
    with open(os.path.join(OUT, "tuning_report_optimizer_only.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
