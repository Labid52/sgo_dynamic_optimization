"""Frozen Phase-2A experimental protocol.

Single source of truth for every design decision that must be fixed BEFORE
production results are observed. Importing this module is the only sanctioned way
for Phase-2 scripts to obtain the fixed MPC design, the evaluation budget, the
initial conditions and the seeding scheme.

Fixed here, and committed before the primary runs:
  * the fixed MPC formulation (identical for all optimizers and the reference)
  * the exact common function-evaluation budget B
  * the optimizer-only tuning search space
  * the seed scheme
  * the number of stochastic runs
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
RESULTS_V2 = os.path.join(PROJECT, "data")

ALGS = ["SGO", "GWO", "PSO", "WOA"]
ALG_INDEX = {a: i for i, a in enumerate(ALGS)}
SYSTEMS = ["pendcart", "cstr", "flight", "hvac", "uav"]
SYSTEM_INDEX = {s: i for i, s in enumerate(SYSTEMS)}

# ---------------------------------------------------------------------------
# Fixed MPC design (reviewer R2.5). The MPC formulation is frozen at the nominal
# per-system design and is IDENTICAL for SGO, GWO, PSO, WOA and the deterministic
# reference. Nothing in this dictionary may be tuned.
#   P        : from system_defs (per system)
#   Nc       : 1
#   Q, R     : from system_defs
#   Q_scale  : 1.0
# ---------------------------------------------------------------------------
FIXED_MPC = {"Nc": 1, "Q_scale": 1.0}      # P comes from system_defs per system


def fixed_mpc_params(sysdef):
    """The frozen MPC design for one system. P is the nominal system horizon."""
    return {"P": int(sysdef.P), "Nc": int(FIXED_MPC["Nc"]),
            "Q_scale": float(FIXED_MPC["Q_scale"])}


# ---------------------------------------------------------------------------
# Exact common function-evaluation budget.
#
# B = 1000 objective evaluations per MPC step, for every optimizer, in both
# regimes. Chosen BEFORE observing any Phase-2A performance result, on two
# grounds: (a) it matches the nominal budget of the submitted equal-FE study, so
# the revised numbers stay comparable with the submitted ones; (b) it leaves a
# meaningful search after initialisation at every population size in the tuning
# grid -- at NP=30 it allows 30 initial evaluations plus ~32 sweeps, at NP=8
# about 124 sweeps.
# ---------------------------------------------------------------------------
FE_BUDGET = 1000

# ---------------------------------------------------------------------------
# Optimizer-only tuning search space.
#
# Only the population size is free. Under an exact FE budget the iteration limit
# is NOT an independent degree of freedom: the run must consume exactly B
# evaluations, so the iteration cap is determined by B and NP. Leaving MaxIter
# free would let a configuration stop before exhausting B, which is precisely the
# unequal-budget effect the strict-FE regime exists to remove. MaxIter is
# therefore derived, not tuned, and is reported as such.
# ---------------------------------------------------------------------------
TUNING_NP_GRID = [8, 10, 14, 20, 30]
TUNING_SEEDS_PER_CANDIDATE = 3

# Regime B fixes a common population size for all optimizers.
COMMON_NP = 20

# Stochastic runs per (system, optimizer, regime) in the primary experiment.
PRIMARY_RUNS = 20

# Seed bases; see data/SEED_PROTOCOL.md
SEED_BASE = {"tuning": 8_000_000, "primary": 9_000_000,
             "frozen_v2": 11_000_000, "multi_ic": 12_000_000,
             "budget_v2": 13_000_000, "tail": 17_000_000}
REGIME_INDEX = {"A_tuned_NP": 0, "B_common_NP": 1}


def seed_for(experiment, instance_index, regime_index, optimizer_index, trial):
    """Collision-free seed; trial is the fastest-varying term."""
    assert 0 <= trial < 1000 and 0 <= optimizer_index < 10 and 0 <= regime_index < 100
    return (SEED_BASE[experiment] + 1_000_000 * instance_index
            + 10_000 * regime_index + 1_000 * optimizer_index + trial)


def strict_fe_params(NP, budget=FE_BUDGET, alg=None, iter_margin=5):
    """Optimizer parameters enforcing an EXACT objective-call budget.

    The iteration cap must be finite AND large enough that the evaluation budget
    is always the binding constraint, but it cannot simply be set huge:

      * PSO and WOA test their evaluation cap only inside the candidate loop, so
        once the cap is reached their outer loop keeps iterating -- updating
        velocities/positions and CONSUMING RANDOM DRAWS -- without evaluating.
        An oversized cap would therefore both waste time and, because the global
        RNG stream carries into the next MPC step, change their results. Their
        cap is kept tight at ceil(B/NP): they evaluate exactly NP points per
        iteration, so the budget binds during the last permitted iteration.

      * SGO evaluates a variable number of points per iteration: between NP/2
        (every fight resolved on the defensive branch, no bridge stage) and about
        3*NP/2. Sizing its cap from the NP-per-iteration assumption lets the
        ITERATION cap bind before the budget on unlucky branch statistics, which
        is exactly what happened in the first Phase-2A primary run: flight/SGO at
        NP=8 fell to 978 evaluations in the worst step. Its cap is therefore
        sized from the WORST CASE of NP/2 evaluations per iteration, i.e.
        ceil(2B/NP). SGO checks its budget at the top of the outer loop and
        breaks, so a larger cap costs it nothing and consumes no extra draws.

    Passing alg=None uses the conservative SGO-compatible sizing.
    """
    NP = int(NP); budget = int(budget)
    per_iter = NP if alg in ("GWO", "PSO", "WOA") else max(NP // 2, 1)
    return {"NP": NP,
            "maxIter": int(np.ceil(budget / per_iter)) + int(iter_margin),
            "max_evals": budget,
            "fe_cap": budget,
            "tol": 0.0,                 # disables stagnation stopping
            "stagnant_limit": 10 ** 9}  # disables stagnation stopping


def load_initial_conditions():
    """Held-out tuning and evaluation ICs, frozen before tuning."""
    path = os.path.join(RESULTS_V2, "initial_condition_protocol.json")
    with open(path) as fh:
        proto = json.load(fh)
    if proto.get("any_overlap_between_tuning_and_evaluation", True):
        raise RuntimeError("tuning and evaluation initial conditions overlap")
    if not proto.get("all_accepted", False):
        raise RuntimeError("not all tuning initial conditions passed validation")
    return {k: {"tune": np.array(v["tuning_x0"], dtype=float),
                "eval": np.array(v["evaluation_x0"], dtype=float)}
            for k, v in proto["systems"].items()}


def load_tuned_np():
    """Population sizes selected by optimizer-only tuning on the held-out IC."""
    path = os.path.join(RESULTS_V2, "tuned_params_optimizer_only.json")
    with open(path) as fh:
        return json.load(fh)
