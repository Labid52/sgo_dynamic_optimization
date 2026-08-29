#!/usr/bin/env python3
"""Phase-2C Task 4: bounded perturbed objective-sequence validation (reviewer R2.3).

Deliberately SMALL. Phase 2B already measured naturally occurring drift spanning
seven orders of magnitude; this does not build a second benchmark suite. Its only
purpose is to test whether DELIBERATELY INDUCED changes that increase measured
objective drift produce the same drift-difficulty relationship observed in the
nominal sequences.

Two contrasting systems:
    HVAC  -- nominal sequence is completely benign (0 of 1500 non-benign)
    UAV   -- substantial natural tail drift and the only clear optimizer separation

Four predeclared conditions, never combined:
    nominal
    process_disturbance    bounded additive disturbance on the plant update
    measurement_noise      noise on the state handed to the controller only
    reference_event        one abrupt reference change mid-trajectory

Magnitudes are physically interpretable, fixed in `perturbation_protocol.json`
BEFORE any optimizer result is seen, and are NOT tuned to induce failure.

Perturbations are applied ONLY in this script, through explicit hooks; the frozen
MPC formulation, objective definitions and optimizers are untouched.

Outputs:
    data/perturbation_protocol.json
    data/perturbation_drift.csv
    data/perturbation_raw.csv
    data/perturbation_summary.csv
    data/tables/perturbation.tex
    data/figures/perturbation_diagnostic.eps
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(PROJECT, "data")
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import (build_problem, make_cost, solve_one_step,      # noqa: E402
                        normalized_rmse)
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import osqp_reference, problem_linear_term      # noqa: E402

SYSTEMS = ["hvac", "uav"]
CONDITIONS = ["nominal", "process_disturbance", "measurement_noise", "reference_event"]
RUNS = 10
SEED_BASE = 16_000_000
BENIGN_CRITERION = 1e-4
NSIM_CAP = {"hvac": 400, "uav": 400}     # bounded length; see protocol note


def protocol():
    return {
        "purpose": "Test whether deliberately induced changes that increase measured objective "
                   "drift reproduce the drift-difficulty relationship found in the nominal "
                   "sequences. Not a second benchmark suite.",
        "systems": SYSTEMS,
        "system_choice_reason": {
            "hvac": "nominal sequence is completely benign (0 of 1500 transitions non-benign), "
                    "so any induced drift is unambiguous",
            "uav": "largest natural tail drift and the only system with clear optimizer "
                   "separation"},
        "conditions": CONDITIONS,
        "conditions_never_combined": True,
        "magnitudes": {
            "process_disturbance": {
                "hvac": "additive state disturbance, sigma = 0.05 K per step on the three room "
                        "temperature states (about 2.5% of the 2 K setpoint offset)",
                "uav": "additive state disturbance, sigma = 0.02 m per step on the three "
                       "position states (0.2% of the 10 m reference travel)"},
            "measurement_noise": {
                "hvac": "sigma = 0.05 K on the state passed to the controller only; the true "
                        "plant state is unperturbed",
                "uav": "sigma = 0.02 m on the position states passed to the controller only"},
            "reference_event": {
                "hvac": "room setpoints step by +1.0 K at 50% of the horizon",
                "uav": "position reference steps by +1.0 m in x at 50% of the horizon"}},
        "magnitude_rationale":
            "Each magnitude is a small fraction of the quantity the controller is regulating, "
            "chosen to be physically plausible and NOT tuned to induce optimizer failure. They "
            "were fixed before any optimizer result was observed.",
        "trajectory_length": NSIM_CAP,
        "trajectory_length_reason":
            "Bounded to 400 steps per system so the study stays small; the nominal condition is "
            "rerun at the same length so all comparisons are like-for-like.",
        "runs_per_optimizer_per_condition": RUNS,
        "mpc": "frozen Phase-2A fixed MPC", "fe_budget": P2.FE_BUDGET,
        "seed_base": SEED_BASE,
    }


def perturb_spec(key, condition, proto):
    """Numeric perturbation parameters for one (system, condition)."""
    if key == "hvac":
        idx, sig, ref_step = [10, 11, 12], 0.05, 1.0
    else:
        idx, sig, ref_step = [0, 1, 2], 0.02, 1.0
    return dict(state_idx=idx,
                proc_sigma=sig if condition == "process_disturbance" else 0.0,
                meas_sigma=sig if condition == "measurement_noise" else 0.0,
                ref_step=(ref_step if condition == "reference_event" else 0.0),
                ref_idx=(idx if key == "hvac" else [0]))


def run_closed_loop(key, sysdef, alg, condition, seed, proto, collect_drift=False):
    """Closed loop with explicit perturbation hooks; frozen MPC and optimizers."""
    mpc = P2.fixed_mpc_params(sysdef)
    problem = build_problem(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"])
    lb, ub = problem.lb_seq, problem.ub_seq
    spec = perturb_spec(key, condition, proto)
    Nsim = int(NSIM_CAP[key])
    ev = int(0.5 * Nsim)
    rng = np.random.default_rng(seed + 777)

    tuned = P2.load_tuned_np()["selected"]
    if alg == "REF":
        params = None
    else:
        params = dict(mpc)
        params.update(P2.strict_fe_params(int(tuned[key][alg]["NP"]), alg=alg))
    np.random.seed(seed)

    x = P2.load_initial_conditions()[key]["eval"].copy()
    x_hist = np.zeros((sysdef.nx, Nsim + 1)); x_hist[:, 0] = x
    u_hist = np.zeros((sysdef.nu, Nsim))
    nfes, times, drift_rows = [], [], []
    last = np.zeros(len(lb))
    prev_ustar = None

    for k in range(Nsim):
        # Reference event: shift the tracked setpoint from step ev onward.
        ref_shift = np.zeros(sysdef.nx)
        if spec["ref_step"] and k >= ev:
            for i in spec["ref_idx"]:
                ref_shift[i] = spec["ref_step"]

        x_ctrl = x.copy()
        if spec["meas_sigma"] > 0:
            x_ctrl[spec["state_idx"]] += rng.normal(0, spec["meas_sigma"],
                                                    len(spec["state_idx"]))
        # The reference event is expressed as an equal-and-opposite state offset,
        # which is exactly how a setpoint change enters this deviation-form cost.
        x_eff = x_ctrl - ref_shift

        if alg == "REF":
            f = problem_linear_term(problem, sysdef, x_eff, k)
            u_sol, status, _ = osqp_reference(problem, f)
            if u_sol is None:
                raise RuntimeError(f"OSQP failed {key} {condition} k={k}: {status}")
            sol = np.clip(u_sol, lb, ub); nfe, wall = 0, 0.0
        else:
            sol, _, _, _, wall, nfe = solve_one_step(problem, x_eff, k, alg, params, last)

        if collect_drift:
            f = problem_linear_term(problem, sysdef, x_eff, k)
            u_star, st, _ = osqp_reference(problem, f)
            if u_star is not None:
                u_star = np.clip(u_star, lb, ub)
                cost_k = make_cost(problem, x_eff, k)
                J_star = float(cost_k(u_star))
                warm_gap = np.nan
                if prev_ustar is not None:
                    warm_gap = (float(cost_k(prev_ustar)) - J_star) / max(1.0, abs(J_star))
                drift_rows.append(dict(system=key, condition=condition, k=k,
                                       J_star=J_star, warm_gap_norm=warm_gap))
                prev_ustar = u_star

        u = np.clip(sol[:sysdef.nu], sysdef.u_min, sysdef.u_max)
        x = sysdef.step(x, u)
        if spec["proc_sigma"] > 0:
            x[spec["state_idx"]] += rng.normal(0, spec["proc_sigma"], len(spec["state_idx"]))
        last = sol.copy()
        if problem.Nc > 1:
            last[:-sysdef.nu] = last[sysdef.nu:]
        x_hist[:, k + 1] = x; u_hist[:, k] = u
        nfes.append(nfe); times.append(wall * 1e3)

    return x_hist, u_hist, np.array(nfes), np.array(times), pd.DataFrame(drift_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--define", action="store_true")
    ap.add_argument("--runs", type=int, default=RUNS)
    args = ap.parse_args()

    proto = protocol()
    with open(os.path.join(OUT, "perturbation_protocol.json"), "w") as f:
        json.dump(proto, f, indent=2)
    if args.define:
        print(json.dumps(proto, indent=2))
        return 0

    t0 = time.perf_counter()
    systems = get_systems()
    ics = P2.load_initial_conditions()

    # ---- stage 1: measure the objective-sequence change itself --------------
    drift_frames = []
    for key in SYSTEMS:
        for cond in CONDITIONS:
            _, _, _, _, dfd = run_closed_loop(key, systems[key], "REF", cond,
                                              SEED_BASE + 1, proto, collect_drift=True)
            drift_frames.append(dfd)
            w = dfd.warm_gap_norm.dropna()
            print(f"  drift {key:5s} {cond:20s} median={w.median():.3e} "
                  f"p90={np.percentile(w,90):.3e} max={w.max():.3e} "
                  f"non-benign={100*(w>BENIGN_CRITERION).mean():.1f}%")
    drift = pd.concat(drift_frames, ignore_index=True)
    drift.to_csv(os.path.join(OUT, "perturbation_drift.csv"), index=False)

    # ---- stage 2: optimizer performance under each condition ----------------
    rows = []
    for key in SYSTEMS:
        sysdef = systems[key]
        x0 = ics[key]["eval"]
        for ci, cond in enumerate(CONDITIONS):
            xh_ref, _, _, _, _ = run_closed_loop(key, sysdef, "REF", cond,
                                                 SEED_BASE + 1, proto)
            ref_nrmse = float(normalized_rmse(xh_ref, sysdef, x0=x0))
            for alg in P2.ALGS:
                for t in range(args.runs):
                    seed = (SEED_BASE + 1_000_000 * SYSTEMS.index(key)
                            + 10_000 * ci + 1_000 * P2.ALG_INDEX[alg] + t)
                    xh, uh, nfe, tms, _ = run_closed_loop(key, sysdef, alg, cond,
                                                          seed, proto)
                    rows.append(dict(
                        system=key, condition=cond, alg=alg, trial=t, seed=seed,
                        nrmse=float(normalized_rmse(xh, sysdef, x0=x0)),
                        reference_nrmse=ref_nrmse,
                        realized_nfe_mean=float(np.mean(nfe)),
                        fe_exact=bool(np.all(nfe == P2.FE_BUDGET)),
                        time_ms_mean=float(np.mean(tms))))
            print(f"  {key:5s} {cond:20s} done ({time.perf_counter()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "perturbation_raw.csv"), index=False)

    dstat = (drift.dropna(subset=["warm_gap_norm"]).groupby(["system", "condition"])
             .agg(median_drift=("warm_gap_norm", "median"),
                  p90_drift=("warm_gap_norm", lambda s: float(np.percentile(s, 90))),
                  max_drift=("warm_gap_norm", "max"),
                  frac_non_benign=("warm_gap_norm",
                                   lambda s: float((s > BENIGN_CRITERION).mean()))).reset_index())
    summ = (df.groupby(["system", "condition", "alg"])
              .agg(n=("nrmse", "size"), nrmse_mean=("nrmse", "mean"),
                   nrmse_median=("nrmse", "median"), nrmse_std=("nrmse", "std"),
                   reference_nrmse=("reference_nrmse", "first"),
                   fe_exact=("fe_exact", "all")).reset_index())
    summ = summ.merge(dstat, on=["system", "condition"], how="left")
    summ["rank"] = summ.groupby(["system", "condition"]).nrmse_mean.rank(method="min").astype(int)
    summ["relative_excess_vs_reference"] = (summ.nrmse_mean - summ.reference_nrmse) / summ.reference_nrmse
    summ.to_csv(os.path.join(OUT, "perturbation_summary.csv"), index=False)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Bounded perturbed objective-sequence study. Induced objective drift "
             r"(median and fraction of non-benign transitions, measured on the deterministic "
             r"sequence) and the resulting closed-loop nRMSE. Conditions are never combined and "
             r"magnitudes were fixed before any optimizer result was observed.}",
             r"\label{tab:perturbation}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
             r"System & Condition & Median drift & Non-benign & SGO & GWO & PSO & WOA & "
             r"Det.\ ref. \\", r"\midrule"]
    for key in SYSTEMS:
        for cond in CONDITIONS:
            g = summ[(summ.system == key) & (summ.condition == cond)]
            if g.empty:
                continue
            cells = []
            for alg in P2.ALGS:
                r = g[g.alg == alg]
                v = float(r.nrmse_mean.iloc[0])
                cells.append((r"\textbf{" + f"{v:.4g}" + "}") if int(r["rank"].iloc[0]) == 1
                             else f"{v:.4g}")
            lines.append(f"{key} & {cond.replace('_',' ')} & "
                         f"{float(g.median_drift.iloc[0]):.2e} & "
                         f"{100*float(g.frac_non_benign.iloc[0]):.0f}\\% & " +
                         " & ".join(cells) +
                         f" & {float(g.reference_nrmse.iloc[0]):.4g} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "perturbation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for j, key in enumerate(SYSTEMS):
        ax = axes[2 * j]
        g = dstat[dstat.system == key]
        ax.bar(range(len(g)), np.maximum(g.median_drift, 1e-18))
        ax.set_yscale("log"); ax.set_xticks(range(len(g)))
        ax.set_xticklabels([c[:10] for c in g.condition], rotation=45, ha="right", fontsize=7)
        ax.axhline(BENIGN_CRITERION, color="crimson", ls="--", lw=.8)
        ax.set_title(f"{key}: induced drift"); ax.set_ylabel("median warm-gap")
        ax.grid(alpha=.3)
        ax = axes[2 * j + 1]
        gg = summ[summ.system == key]
        for alg in P2.ALGS:
            v = [float(gg[(gg.condition == c) & (gg.alg == alg)].nrmse_mean.iloc[0])
                 for c in CONDITIONS]
            ax.plot(range(len(CONDITIONS)), v, "o-", ms=4, label=alg)
        ref = [float(gg[gg.condition == c].reference_nrmse.iloc[0]) for c in CONDITIONS]
        ax.plot(range(len(CONDITIONS)), ref, "k--", lw=1, label="det. ref.")
        ax.set_yscale("log"); ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([c[:10] for c in CONDITIONS], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{key}: nRMSE"); ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "perturbation_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    meta = {"rows": int(len(df)), "runs": args.runs,
            "fe_exact_everywhere": bool(df.fe_exact.all()),
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "perturbation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.set_option("display.width", 250)
    print(dstat.to_string(index=False))
    print(summ[["system", "condition", "alg", "nrmse_mean", "reference_nrmse",
                "rank"]].to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
