#!/usr/bin/env python3
"""Phase-2A step 5: define, validate and FREEZE the held-out initial conditions.

Reviewer 2 identified tuning-to-test overlap: the submitted study tuned and
evaluated on the same trajectory from the same initial condition, so the tuned
configuration was never tested on unseen data.

This script fixes that by defining, for every benchmark:

    x0_eval   the primary evaluation IC  (kept equal to the submitted x0, so the
              revised primary experiment stays directly comparable with the
              submitted results)
    x0_tune   a held-out tuning IC, distinct from x0_eval, used ONLY for
              optimizer parameter selection and never evaluated

Each candidate tuning IC is validated with the DETERMINISTIC controller before
being accepted:
  * the closed loop must remain bounded (no divergence);
  * the normalized tracking error must be finite and of a sane magnitude;
  * the IC must not already sit at the reference (otherwise tuning would be
    driven by a trivial regulation problem);
  * the IC must differ from x0_eval by a meaningful margin.

An IC failing validation is rejected and the reason recorded, BEFORE any
optimizer tuning is run.

The tuning ICs are also recorded as reserved, so that the Phase-2B multi-initial
-condition experiment can exclude them from its evaluation set.

Outputs:
    data/initial_condition_protocol.json

Usage:
    python initial_condition_protocol.py
"""
import os, sys, json, time
import numpy as np

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

# Held-out tuning initial conditions, chosen on physical grounds BEFORE any
# tuning is run. Rationale is recorded per system and is not performance-based.
TUNING_ICS = {
    "pendcart": dict(
        x0=lambda s: np.array([0.5, 0.0, np.pi - 0.15, 0.0]),
        rationale="Cart displaced to the opposite side of the reference (+0.5 m vs -1 m) "
                  "and the pendulum perturbed in the opposite angular direction "
                  "(pi-0.15 vs pi+0.20). Same physical regime, different transient."),
    "cstr": dict(
        x0=lambda s: np.array([-0.4, 0.8]),
        rationale="Deviation state with both components sign-flipped relative to the "
                  "evaluation IC [0.5, -1.0] and of comparable magnitude, so the "
                  "regulation task is equally demanding from the opposite quadrant."),
    "flight": dict(
        x0=lambda s: np.array([-0.08, 0.12, -0.04, 0.06]),
        rationale="Longitudinal deviation state sign-flipped and rescaled relative to "
                  "[0.1, -0.1, 0.05, -0.05]; remains well inside the small-perturbation "
                  "range where the linearised model is valid."),
    "hvac": dict(
        x0=lambda s: 18.0 * np.ones(13),
        rationale="Building starting from a uniformly colder state (18 C) instead of "
                  "20 C, giving a larger initial deviation from the 22 C room setpoints "
                  "and a -2 C deviation for the wall states. Physically realistic "
                  "cold-start scenario."),
    "uav": dict(
        x0=lambda s: np.concatenate([np.array([-1.0, 0.5, 0.5]), np.zeros(9)]),
        rationale="Quadrotor starting displaced from the beginning of the ramp reference "
                  "(1 m behind in x, 0.5 m off in y and z) with zero attitude and rates, "
                  "so the tracking task starts with a real initial offset instead of "
                  "exactly on the trajectory."),
}

# Validation thresholds, fixed in advance.
MAX_ABS_STATE = 1e4          # divergence guard
MIN_IC_SEPARATION = 1e-3     # tuning IC must differ meaningfully from evaluation IC
MIN_INITIAL_DEVIATION = 1e-6  # tuning IC must not already sit at the reference
NRMSE_SANITY_MAX = 1e3


def initial_deviation(sysdef, x0):
    if sysdef.cost_mode == "trajectory":
        ref0 = sysdef.ref_traj[:, 0]
    else:
        ref0 = sysdef.reference_at(0)
    return float(np.linalg.norm(np.asarray(x0, dtype=float) - ref0))


def validate(key, sysdef, x0_tune, x0_eval):
    """Deterministic feasibility check of a candidate tuning IC."""
    rec = {"separation_from_eval_ic": float(np.linalg.norm(x0_tune - x0_eval)),
           "initial_deviation_from_reference": initial_deviation(sysdef, x0_tune)}
    out = simulate(key, "QP", params={"Nc": 1, "Q_scale": 1.0}, seed=11,
                   verbose=False, x0_override=x0_tune)
    xh = out["x_hist"]
    rec["max_abs_state"] = float(np.max(np.abs(xh)))
    rec["all_finite"] = bool(np.all(np.isfinite(xh)))
    rec["deterministic_nrmse"] = float(normalized_rmse(xh, sysdef, x0=x0_tune))
    rec["final_state_norm"] = float(np.linalg.norm(xh[:, -1]))

    checks = {
        "bounded": rec["all_finite"] and rec["max_abs_state"] < MAX_ABS_STATE,
        "distinct_from_eval_ic": rec["separation_from_eval_ic"] > MIN_IC_SEPARATION,
        "not_at_reference": rec["initial_deviation_from_reference"] > MIN_INITIAL_DEVIATION,
        "nrmse_sane": np.isfinite(rec["deterministic_nrmse"])
        and rec["deterministic_nrmse"] < NRMSE_SANITY_MAX,
    }
    rec["checks"] = checks
    rec["accepted"] = bool(all(checks.values()))
    return rec


def main():
    t0 = time.perf_counter()
    protocol = {
        "purpose": "Separate optimizer tuning from evaluation (reviewer R2.5). "
                   "Optimizer parameters are selected using x0_tune only; all "
                   "primary results are produced from x0_eval.",
        "frozen_before_tuning": True,
        "validation_thresholds": {
            "max_abs_state": MAX_ABS_STATE,
            "min_ic_separation": MIN_IC_SEPARATION,
            "min_initial_deviation_from_reference": MIN_INITIAL_DEVIATION,
            "nrmse_sanity_max": NRMSE_SANITY_MAX,
        },
        "reserved_for_tuning_only": "The x0_tune values below must be EXCLUDED from the "
                                    "Phase-2B multi-initial-condition evaluation set.",
        "systems": {},
    }
    all_ok = True
    for key, sysdef in get_systems().items():
        x0_sub = np.asarray(sysdef.x0, dtype=float)
        x0_eval = x0_sub.copy()          # evaluation IC = submitted IC, for comparability
        x0_tune = np.asarray(TUNING_ICS[key]["x0"](sysdef), dtype=float)
        rec = validate(key, sysdef, x0_tune, x0_eval)
        entry = {
            "submitted_x0": x0_sub.tolist(),
            "evaluation_x0": x0_eval.tolist(),
            "tuning_x0": x0_tune.tolist(),
            "evaluation_equals_submitted": bool(np.array_equal(x0_eval, x0_sub)),
            "tuning_equals_evaluation": bool(np.array_equal(x0_tune, x0_eval)),
            "rationale": TUNING_ICS[key]["rationale"],
            "deterministic_validation": rec,
        }
        protocol["systems"][key] = entry
        all_ok &= rec["accepted"]
        status = "ACCEPTED" if rec["accepted"] else "REJECTED"
        print(f"  {key:9s} {status}: separation {rec['separation_from_eval_ic']:.3f}, "
              f"det nRMSE {rec['deterministic_nrmse']:.4e}, "
              f"max|x| {rec['max_abs_state']:.3e}")
        if not rec["accepted"]:
            print(f"      failed checks: {[k for k, v in rec['checks'].items() if not v]}")

    protocol["all_accepted"] = bool(all_ok)
    protocol["any_overlap_between_tuning_and_evaluation"] = bool(
        any(e["tuning_equals_evaluation"] for e in protocol["systems"].values()))
    protocol["wall_seconds"] = round(time.perf_counter() - t0, 1)
    path = os.path.join(OUT, "initial_condition_protocol.json")
    with open(path, "w") as fh:
        json.dump(protocol, fh, indent=2)
    print(f"\nall accepted: {all_ok}; overlap: "
          f"{protocol['any_overlap_between_tuning_and_evaluation']}")
    print(f"Wrote {path}")
    return 0 if all_ok and not protocol["any_overlap_between_tuning_and_evaluation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
