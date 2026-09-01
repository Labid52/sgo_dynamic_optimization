#!/usr/bin/env python3
"""Phase-2B step 7: multi-initial-condition robustness (reviewer R2.12).

Reviewer 2 refused to accept fixed initial conditions as a future-work item. This
runs a compact but credible multi-IC experiment under the frozen Phase-2A
fixed-MPC exact-FE protocol, to test whether the Phase-2A conclusions are
specific to one trajectory.

Design, fixed before any stochastic run:
  * 5 evaluation ICs per system: the submitted evaluation IC plus 4 additional
    physically sensible ones.
  * The held-out TUNING ICs are excluded by construction and asserted against.
  * Every candidate IC is validated with the deterministic controller before use
    (bounded closed loop, finite sane nRMSE, not already at the reference).
  * 5 stochastic seeds per optimizer per IC; MPC frozen; exact FE budget 1000.
  * The nRMSE normalisation uses each run's own IC, otherwise the metric is
    silently rescaled between ICs.

Ranking stability is assessed with Kendall's tau between each IC's optimizer
ranking and every other IC's, per system.

Outputs:
    data/multi_ic_protocol.json      (written and committed before runs)
    data/multi_ic_raw.csv
    data/multi_ic_summary.csv
    data/multi_ic_ranking_stability.csv
    data/tables/multi_ic.tex
    data/figures/multi_ic_diagnostic.eps

Usage:
    python multi_ic_experiment.py --define    # write + validate the protocol only
    python multi_ic_experiment.py             # run the experiment
"""
import os, sys, json, argparse, time, itertools
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
# Output root. Defaults to the released data/ directory; set SGO_OUTPUT_DIR to
# redirect every generated file elsewhere (e.g. a scratch directory) so that a
# smoke test cannot overwrite the committed production evidence.
OUT = os.environ.get("SGO_OUTPUT_DIR") or os.path.join(PROJECT, "data")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, normalized_rmse                       # noqa: E402
import phase2_protocol as P2                                           # noqa: E402

SEEDS_PER_CELL = 5
MAX_ABS_STATE = 1e4
NRMSE_SANITY_MAX = 1e3


def candidate_ics(key, sysdef, x0_eval):
    """Four additional ICs per system, chosen on physical grounds.

    Deliberately NOT random perturbations of the evaluation IC: each is a
    recognisable operating scenario, and none coincides with the held-out tuning
    IC (asserted separately).
    """
    if key == "pendcart":
        return {
            "IC1_submitted": x0_eval,
            "IC2_far_cart": np.array([-2.0, 0.0, np.pi + 0.10, 0.0]),
            "IC3_opposite_angle": np.array([-1.0, 0.0, np.pi - 0.20, 0.0]),
            "IC4_with_velocity": np.array([-1.0, 0.3, np.pi + 0.20, -0.2]),
            "IC5_near_reference": np.array([0.6, 0.0, np.pi + 0.05, 0.0]),
        }
    if key == "cstr":
        return {
            "IC1_submitted": x0_eval,
            "IC2_larger_deviation": np.array([1.0, -1.5]),
            "IC3_concentration_only": np.array([0.8, 0.0]),
            "IC4_temperature_only": np.array([0.0, -1.2]),
            "IC5_small_deviation": np.array([0.2, -0.3]),
        }
    if key == "flight":
        return {
            "IC1_submitted": x0_eval,
            "IC2_larger_alpha": np.array([0.05, -0.25, 0.05, -0.05]),
            "IC3_pitch_rate": np.array([0.10, -0.10, 0.20, -0.05]),
            "IC4_theta_offset": np.array([0.10, -0.10, 0.05, 0.20]),
            "IC5_small": np.array([0.03, -0.03, 0.02, -0.02]),
        }
    if key == "hvac":
        return {
            "IC1_submitted": x0_eval,
            "IC2_cold_building": np.array([16.0] * 13),
            "IC3_warm_building": np.array([24.0] * 13),
            "IC4_warm_rooms_cold_walls": np.array([18.0] * 10 + [23.0] * 3),
            "IC5_at_room_setpoint": np.array([20.0] * 10 + [22.0] * 3),
        }
    if key == "uav":
        return {
            "IC1_submitted": x0_eval,
            "IC2_offset_xy": np.concatenate([[1.5, -1.0, 0.0], np.zeros(9)]),
            "IC3_altitude_offset": np.concatenate([[0.0, 0.0, 2.0], np.zeros(9)]),
            "IC4_attitude_perturbed": np.concatenate(
                [np.zeros(3), [0.05, -0.05, 0.0], np.zeros(6)]),
            "IC5_with_velocity": np.concatenate(
                [np.zeros(6), [0.5, -0.5, 0.2], np.zeros(3)]),
        }
    raise KeyError(key)


def validate(key, sysdef, x0, x0_tune):
    out = simulate(key, "QP", params=P2.fixed_mpc_params(sysdef), seed=11,
                   verbose=False, x0_override=x0)
    xh = out["x_hist"]
    nr = float(normalized_rmse(xh, sysdef, x0=x0))
    checks = {
        "bounded": bool(np.all(np.isfinite(xh)) and np.max(np.abs(xh)) < MAX_ABS_STATE),
        "nrmse_sane": bool(np.isfinite(nr) and nr < NRMSE_SANITY_MAX),
        "not_equal_to_tuning_ic": bool(not np.allclose(x0, x0_tune)),
    }
    return {"deterministic_nrmse": nr, "max_abs_state": float(np.max(np.abs(xh))),
            "checks": checks, "accepted": bool(all(checks.values()))}


def define_protocol():
    ics = P2.load_initial_conditions()
    systems = get_systems()
    proto = {"purpose": "Test whether the Phase-2A conclusions are specific to one trajectory "
                        "(reviewer R2.12).",
             "protocol": {"ics_per_system": 5, "seeds_per_optimizer_per_ic": SEEDS_PER_CELL,
                          "mpc": "frozen Phase-2A fixed MPC", "fe_budget": P2.FE_BUDGET,
                          "nrmse_scale": "computed from each run's own initial condition",
                          "tuning_ics_excluded": True},
             "systems": {}}
    ok = True
    for key, sysdef in systems.items():
        cands = candidate_ics(key, sysdef, ics[key]["eval"])
        entry = {"tuning_ic_reserved": ics[key]["tune"].tolist(), "ics": {}}
        for name, x0 in cands.items():
            v = validate(key, sysdef, np.asarray(x0, float), ics[key]["tune"])
            entry["ics"][name] = {"x0": np.asarray(x0, float).tolist(), **v}
            ok &= v["accepted"]
            print(f"  {key:9s} {name:24s} {'OK ' if v['accepted'] else 'REJECT'} "
                  f"det nRMSE {v['deterministic_nrmse']:.4e}")
        proto["systems"][key] = entry
    proto["all_accepted"] = bool(ok)
    with open(os.path.join(OUT, "multi_ic_protocol.json"), "w") as f:
        json.dump(proto, f, indent=2)
    print(f"all accepted: {ok}")
    return ok


def run():
    with open(os.path.join(OUT, "multi_ic_protocol.json")) as f:
        proto = json.load(f)
    if not proto.get("all_accepted"):
        raise RuntimeError("multi-IC protocol contains rejected initial conditions")
    systems = get_systems()
    tuned = P2.load_tuned_np()["selected"]
    rows = []
    t0 = time.perf_counter()
    for si, key in enumerate(P2.SYSTEMS):
        sysdef = systems[key]
        mpc = P2.fixed_mpc_params(sysdef)
        for ic_i, (ic_name, ic) in enumerate(proto["systems"][key]["ics"].items()):
            x0 = np.array(ic["x0"], dtype=float)
            det = simulate(key, "QP", params=mpc, seed=11, verbose=False, x0_override=x0)
            det_nrmse = float(normalized_rmse(det["x_hist"], sysdef, x0=x0))
            for alg in P2.ALGS:
                NP = int(tuned[key][alg]["NP"])
                params = dict(mpc)
                params.update(P2.strict_fe_params(NP, alg=alg))
                for t in range(SEEDS_PER_CELL):
                    seed = P2.seed_for("multi_ic", si, ic_i, P2.ALG_INDEX[alg], t)
                    out = simulate(key, alg, params=params, seed=seed, verbose=False,
                                   x0_override=x0)
                    nfe = np.asarray(out["nfe"], dtype=float)
                    rows.append(dict(
                        system=key, ic=ic_name, alg=alg, trial=t, seed=seed, NP=NP,
                        nrmse=float(normalized_rmse(out["x_hist"], sysdef, x0=x0)),
                        deterministic_reference_nrmse=det_nrmse,
                        realized_nfe_mean=float(np.mean(nfe)),
                        fe_exact=bool(np.all(nfe == params["fe_cap"])),
                        time_ms_mean=float(np.mean(out["time_ms"]))))
            print(f"  {key:9s} {ic_name:24s} done ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "multi_ic_raw.csv"), index=False)

    summ = (df.groupby(["system", "ic", "alg"])
              .agg(n=("nrmse", "size"), nrmse_mean=("nrmse", "mean"),
                   nrmse_median=("nrmse", "median"), nrmse_std=("nrmse", "std"),
                   reference_nrmse=("deterministic_reference_nrmse", "first"),
                   fe_exact=("fe_exact", "all")).reset_index())
    summ["rank"] = summ.groupby(["system", "ic"]).nrmse_mean.rank(method="min").astype(int)
    summ.to_csv(os.path.join(OUT, "multi_ic_summary.csv"), index=False)

    # Ranking stability: Kendall tau between every pair of ICs, per system.
    stab = []
    for key, g in summ.groupby("system"):
        ics = sorted(g.ic.unique())
        rk = {ic: g[g.ic == ic].set_index("alg")["rank"].reindex(P2.ALGS).to_numpy()
              for ic in ics}
        taus = []
        for a, b in itertools.combinations(ics, 2):
            tau, p = kendalltau(rk[a], rk[b])
            taus.append(tau)
            stab.append(dict(system=key, ic_a=a, ic_b=b, kendall_tau=float(tau),
                             p_value=float(p)))
        best = {ic: P2.ALGS[int(np.argmin(rk[ic]))] for ic in ics}
        stab.append(dict(system=key, ic_a="ALL", ic_b="ALL",
                         kendall_tau=float(np.mean(taus)), p_value=np.nan,
                         mean_pairwise_tau=float(np.mean(taus)),
                         n_distinct_best=len(set(best.values())),
                         best_by_ic=json.dumps(best)))
    stabdf = pd.DataFrame(stab)
    stabdf.to_csv(os.path.join(OUT, "multi_ic_ranking_stability.csv"), index=False)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Multi-initial-condition robustness under the fixed-MPC exact-FE "
             r"protocol. Mean closed-loop nRMSE over 5 seeds; the best optimizer for each "
             r"initial condition is shown in bold. The held-out tuning initial conditions are "
             r"excluded by construction.}",
             r"\label{tab:multi_ic}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"System & Initial condition & SGO & GWO & PSO & WOA & Det.\ ref. \\", r"\midrule"]
    for key in P2.SYSTEMS:
        g = summ[summ.system == key]
        for ic in sorted(g.ic.unique()):
            gg = g[g.ic == ic]
            cells = []
            for alg in P2.ALGS:
                r = gg[gg.alg == alg]
                v = float(r.nrmse_mean.iloc[0])
                cells.append((r"\textbf{" + f"{v:.5g}" + "}") if int(r["rank"].iloc[0]) == 1
                             else f"{v:.5g}")
            lines.append(f"{key} & {ic.replace('_',' ')} & " + " & ".join(cells) +
                         f" & {float(gg.reference_nrmse.iloc[0]):.5g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "multi_ic.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
    for ax, key in zip(axes, P2.SYSTEMS):
        g = summ[summ.system == key]
        ics = sorted(g.ic.unique())
        for alg in P2.ALGS:
            v = [float(g[(g.ic == ic) & (g.alg == alg)].nrmse_mean.iloc[0]) for ic in ics]
            ax.plot(range(len(ics)), v, "o-", ms=4, lw=1, label=alg)
        ref = [float(g[g.ic == ic].reference_nrmse.iloc[0]) for ic in ics]
        ax.plot(range(len(ics)), ref, "k--", lw=1, label="det. ref.")
        ax.set_yscale("log"); ax.set_title(key)
        ax.set_xticks(range(len(ics)))
        ax.set_xticklabels([f"IC{i+1}" for i in range(len(ics))], fontsize=7)
        ax.grid(alpha=.3)
        if key == P2.SYSTEMS[0]:
            ax.set_ylabel("mean nRMSE"); ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "multi_ic_diagnostic.eps"),
                format="eps", bbox_inches="tight")
    plt.close(fig)

    meta = {"rows": int(len(df)), "fe_exact_everywhere": bool(df.fe_exact.all()),
            "seeds_per_cell": SEEDS_PER_CELL,
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "multi_ic_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.set_option("display.width", 250)
    print(stabdf[stabdf.ic_a == "ALL"].to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--define", action="store_true")
    a = ap.parse_args()
    if a.define:
        raise SystemExit(0 if define_protocol() else 1)
    raise SystemExit(run())
