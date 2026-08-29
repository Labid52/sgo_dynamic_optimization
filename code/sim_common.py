"""
Common MPC simulation engine for SGO/GWO/PSO/WOA and QP baseline.
"""
import os, json, time
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy.optimize import minimize

from system_defs import get_system, get_systems, SystemDef
from mpc_utils import build_prediction_matrices, build_qp_matrices, compute_f, normalize_cost
from sgo import sgo
from gwo import gwo
from pso import pso
from woa import woa

OPTIMIZERS = {"SGO": sgo, "GWO": gwo, "PSO": pso, "WOA": woa}
ALG_ORDER = ["SGO", "GWO", "PSO", "WOA", "QP"]
ROOT = os.path.dirname(os.path.abspath(__file__))
TUNED_PATH = os.path.join(ROOT, "tuned_params.json")

DEFAULT_PARAMS = {
    "SGO": {"NP": 20, "maxIter": 60, "P": None, "Nc": 1, "Q_scale": 1.0},
    "GWO": {"NP": 20, "maxIter": 60, "P": None, "Nc": 1, "Q_scale": 1.0},
    "PSO": {"NP": 20, "maxIter": 60, "P": None, "Nc": 1, "Q_scale": 1.0},
    "WOA": {"NP": 20, "maxIter": 60, "P": None, "Nc": 1, "Q_scale": 1.0},
}


def load_tuned_params(path: str = TUNED_PATH) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def get_params(system_key: str, alg: str, tuned: Optional[Dict] = None, mode: str = "tuned") -> Dict:
    if alg == "QP":
        return {"NP": 0, "maxIter": 0, "P": None, "Nc": 1, "Q_scale": 1.0}
    tuned = load_tuned_params() if tuned is None else tuned
    p = dict(DEFAULT_PARAMS[alg])
    if mode == "tuned" and system_key in tuned and alg in tuned[system_key]:
        p.update(tuned[system_key][alg])
    elif mode == "equal_fe":
        p.update(equal_fe_params(system_key)[alg])
    out = {"NP": int(p["NP"]), "maxIter": int(p["maxIter"]),
           "Nc": int(p.get("Nc",1)), "Q_scale": float(p.get("Q_scale",1.0))}
    # A tuned prediction horizon is optional. If it is absent/None, system_defs.P is used.
    if p.get("P", None) is not None:
        out["P"] = int(p["P"])
    # Budget/stopping fields must survive, otherwise equal-FE callers silently
    # lose their evaluation cap (this was a real defect: run_all.py --mode
    # equal_fe ran uncapped because max_evals was dropped here).
    for key in ("max_evals", "fe_cap", "tol", "stagnant_limit"):
        if key in p and p[key] is not None:
            out[key] = p[key]
    return out


def equal_fe_params(system_key: str, NP_equal: Optional[int] = None, budget: Optional[int] = None) -> Dict[str, Dict]:
    """Strict equal evaluation-budget parameters for all metaheuristics.

    Runtime can be controlled by environment variables:
      MPC_EQUAL_NP, MPC_EQUAL_BUDGET
    """
    NP_equal = int(os.getenv("MPC_EQUAL_NP", NP_equal or 20))
    if NP_equal % 2: NP_equal += 1
    budget = int(os.getenv("MPC_EQUAL_BUDGET", budget or 1000))
    max_iter = max(1, budget // max(NP_equal, 1))
    return {a: {"NP": NP_equal, "maxIter": max_iter, "Nc": 1, "Q_scale": 1.0,
                "max_evals": budget} for a in ["SGO","GWO","PSO","WOA"]}


@dataclass
class MPCProblem:
    sys: SystemDef
    P: int
    Nc: int
    Phi: np.ndarray
    Gamma: np.ndarray
    Q_block: np.ndarray
    H: np.ndarray
    lb_seq: np.ndarray
    ub_seq: np.ndarray


def build_problem(sys: SystemDef, Nc: int, Q_scale: float = 1.0, P: Optional[int] = None) -> MPCProblem:
    P = int(sys.P if P is None else P)
    Phi, Gamma = build_prediction_matrices(sys.Ad, sys.Bd, P, Nc)
    Q_block = np.kron(np.eye(P), Q_scale * sys.Q)
    R_block = np.kron(np.eye(Nc), sys.R)
    H = build_qp_matrices(Phi, Gamma, Q_block, R_block)
    lb_seq = sys.u_min * np.ones(Nc * sys.nu)
    ub_seq = sys.u_max * np.ones(Nc * sys.nu)
    return MPCProblem(sys, P, Nc, Phi, Gamma, Q_block, H, lb_seq, ub_seq)


def make_cost(problem: MPCProblem, x: np.ndarray, k: int):
    sys = problem.sys
    x_c = sys.cost_state(x)
    ref = sys.reference_horizon(k, problem.P)
    f = compute_f(problem.Phi, problem.Gamma, problem.Q_block, x_c, ref)

    def cost(U):
        U = np.asarray(U, dtype=float).ravel()
        J = 0.5 * (U @ problem.H @ U) + f @ U
        if sys.cart_constraint is not None:
            # Keep the same penalty idea as the original pendulum-cart script.
            S = np.kron(np.eye(problem.P), np.array([[1,0,0,0]], dtype=float))
            pred = problem.Phi @ x_c + problem.Gamma @ U
            cart = (S @ pred).ravel()
            J += sys.constraint_penalty * np.sum(np.maximum(np.abs(cart)-sys.cart_constraint, 0.0)**2)
        return float(J)
    return cost


class BudgetExhausted(Exception):
    """Raised by CountedCost when a hard evaluation cap would be exceeded."""
    pass


class CountedCost:
    """Objective wrapper that counts calls and can enforce a hard evaluation cap.

    fe_cap=None reproduces the original behaviour exactly (count only, no best
    tracking, no interruption), so every pre-existing call site is unaffected.

    When fe_cap is an integer, the cap is enforced at the OBJECTIVE-CALL level:
    every evaluation counts, including the initial population sweep, and call
    number fe_cap+1 raises BudgetExhausted instead of being evaluated. An
    optimizer that proposes more candidates than the remaining budget allows
    therefore has only the allowable candidates evaluated. The best point seen
    so far is retained so a run can still return a solution after interruption.
    This gives a strictly identical realized budget across optimizers, which the
    optimizers' own max_evals checks cannot guarantee because they are tested at
    loop boundaries rather than per evaluation.
    """
    def __init__(self, fn, fe_cap=None):
        self.fn = fn; self.count = 0
        self.fe_cap = None if fe_cap is None else int(fe_cap)
        self.best_x = None; self.best_f = float("inf")
    def __call__(self, x):
        if self.fe_cap is not None and self.count >= self.fe_cap:
            raise BudgetExhausted(f"evaluation cap {self.fe_cap} reached")
        self.count += 1
        val = self.fn(x)
        if self.fe_cap is not None:
            v = float(val)
            if v < self.best_f:
                self.best_f = v
                self.best_x = np.array(x, dtype=float, copy=True)
        return val


def _warm_norm(last_seq, lb_seq, ub_seq):
    return (last_seq - lb_seq)/(ub_seq-lb_seq)*2.0 - 1.0


def solve_one_step(problem: MPCProblem, x: np.ndarray, k: int, alg: str, params: Dict,
                   last_seq: Optional[np.ndarray] = None):
    raw_cost = make_cost(problem, x, k)
    if last_seq is None:
        last_seq = np.zeros_like(problem.lb_seq)
    last_seq = np.clip(last_seq, problem.lb_seq, problem.ub_seq)

    if alg == "QP":
        counted = CountedCost(raw_cost)
        t0 = time.perf_counter()
        res = minimize(counted, last_seq, method="L-BFGS-B",
                       bounds=list(zip(problem.lb_seq, problem.ub_seq)),
                       options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": int(os.getenv("MPC_QP_MAXITER", 200))})
        wall = time.perf_counter()-t0
        sol = np.clip(res.x, problem.lb_seq, problem.ub_seq)
        return sol, float(raw_cost(sol)), None, int(res.nit), wall, counted.count

    cost_norm, lb_n, ub_n, unscale = normalize_cost(raw_cost, problem.lb_seq, problem.ub_seq)
    fe_cap = params.get("fe_cap", None)
    counted = CountedCost(cost_norm, fe_cap=fe_cap)
    warm_n = _warm_norm(last_seq, problem.lb_seq, problem.ub_seq)
    opt = OPTIMIZERS[alg]
    kwargs = {}
    if "max_evals" in params:
        kwargs["max_evals"] = int(params["max_evals"])
    # Stopping-rule overrides. Absent keys leave each optimizer's own defaults
    # (tol=1e-10, stagnant_limit=15) untouched, so the default path is unchanged.
    if "tol" in params:
        kwargs["tol"] = float(params["tol"])
    if "stagnant_limit" in params:
        kwargs["stagnant_limit"] = int(params["stagnant_limit"])
    t_start = time.perf_counter()
    try:
        sol_n, best_cost, conv, n_iter, wall = opt(counted, int(params["NP"]), int(params["maxIter"]),
                                                   lb_n, ub_n, x0_warm=warm_n, **kwargs)
    except BudgetExhausted:
        # Hard cap hit mid-iteration: return the best point actually evaluated.
        sol_n = counted.best_x
        best_cost = counted.best_f
        conv = None
        n_iter = -1
        wall = time.perf_counter() - t_start
        if sol_n is None:      # cap exhausted before any evaluation completed
            sol_n = warm_n
    if fe_cap is not None and counted.best_x is not None:
        # Guarantee the returned point is the best evaluated under the cap.
        if float(counted.best_f) < float(cost_norm(sol_n)):
            sol_n = counted.best_x
    sol = np.clip(unscale(sol_n), problem.lb_seq, problem.ub_seq)
    # Re-evaluate raw cost for a consistent value across algorithms.
    return sol, float(raw_cost(sol)), conv, int(n_iter), wall, counted.count


def reference_solve(problem: MPCProblem, x: np.ndarray, k: int,
                    last_seq: Optional[np.ndarray] = None,
                    n_restarts: int = 20, seed: int = 0):
    """High-accuracy deterministic reference for FROZEN MPC subproblems.

    L-BFGS-B is run from the warm/zero start plus `n_restarts` random initial
    points sampled uniformly in the input bounds; the lowest objective value is
    kept. This matches the methodology statement in the paper. The MPC
    subproblems in this study are convex (quadratic cost plus a convex soft
    cart penalty), so all starts converge to the same value; the restarts are a
    verification safeguard, not a search mechanism.

    The closed-loop QP baseline (solve_one_step with alg='QP') intentionally
    remains a single warm-started L-BFGS-B run per step, which is the standard
    receding-horizon usage; this function is reserved for the convergence,
    statistics, and budget-sensitivity reference solutions.

    Returns: sol, J_star, n_starts_used, spread (max-min objective over starts)
    """
    raw_cost = make_cost(problem, x, k)
    if last_seq is None:
        last_seq = np.zeros_like(problem.lb_seq)
    last_seq = np.clip(last_seq, problem.lb_seq, problem.ub_seq)
    rng = np.random.default_rng(seed)
    starts = [last_seq] + [problem.lb_seq + rng.random(len(problem.lb_seq)) *
                           (problem.ub_seq - problem.lb_seq)
                           for _ in range(int(n_restarts))]
    best_sol, best_val, vals = None, np.inf, []
    bounds = list(zip(problem.lb_seq, problem.ub_seq))
    for s0 in starts:
        res = minimize(raw_cost, s0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "gtol": 1e-10,
                                "maxiter": int(os.getenv("MPC_QP_MAXITER", 200))})
        sol = np.clip(res.x, problem.lb_seq, problem.ub_seq)
        val = float(raw_cost(sol))
        vals.append(val)
        if val < best_val:
            best_val, best_sol = val, sol
    return best_sol, best_val, len(starts), float(max(vals) - min(vals))


def simulate(system_key: str, alg: str, params: Optional[Dict] = None, seed: int = 42,
             mode: str = "tuned", Nsim_override: Optional[int] = None, verbose: bool = False,
             x0_override: Optional[np.ndarray] = None):
    """Closed-loop MPC simulation.

    x0_override replaces the system's nominal initial condition and is used by
    the held-out-tuning and multi-initial-condition experiments. Leaving it as
    None reproduces the original behaviour exactly.
    """
    sys = get_system(system_key)
    if params is None:
        params = get_params(system_key, alg, mode=mode)
    if alg == "QP":
        params = {"Nc": int(params.get("Nc", 1)), "Q_scale": float(params.get("Q_scale",1.0))}
    np.random.seed(seed)
    Nsim = int(os.getenv(f"MPC_{system_key.upper()}_NSIM", Nsim_override or sys.Nsim))
    problem = build_problem(sys, int(params.get("Nc",1)), float(params.get("Q_scale",1.0)), params.get("P", None))
    x = sys.x0.copy() if x0_override is None else np.asarray(x0_override, dtype=float).copy()
    if x.shape != sys.x0.shape:
        raise ValueError(f"x0_override shape {x.shape} != system x0 shape {sys.x0.shape}")
    x_hist = np.zeros((sys.nx, Nsim+1)); u_hist = np.zeros((sys.nu, Nsim))
    x_hist[:,0] = x
    cost_log = np.zeros(Nsim); iter_log = np.zeros(Nsim, dtype=int)
    time_log = np.zeros(Nsim); nfe_log = np.zeros(Nsim, dtype=int)
    conv_last = None
    last_seq = np.zeros(problem.Nc*sys.nu)
    for k in range(Nsim):
        sol, cost, conv, nit, wall, nfe = solve_one_step(problem, x, k, alg, params, last_seq)
        u = np.clip(sol[:sys.nu], sys.u_min, sys.u_max)
        x = sys.step(x, u)
        last_seq = sol.copy()
        # Shift warm sequence for next step if Nc > 1.
        if problem.Nc > 1:
            last_seq[:-sys.nu] = last_seq[sys.nu:]
        x_hist[:,k+1] = x; u_hist[:,k] = u
        cost_log[k] = cost; iter_log[k] = nit; time_log[k] = wall*1e3; nfe_log[k] = nfe
        if conv is not None: conv_last = conv
        if verbose and ((k+1) % max(1, Nsim//5) == 0):
            print(f"    {alg} {system_key}: step {k+1}/{Nsim}, J={cost:.3e}, u0={u[0]:.3f}")
    return dict(system=sys, alg=alg, params=params, x_hist=x_hist, u_hist=u_hist,
                cost=cost_log, iters=iter_log, time_ms=time_log, nfe=nfe_log,
                conv=conv_last, Nsim=Nsim)


def trajectory_states(system_key: str, baseline_alg: str = "QP"):
    out = simulate(system_key, baseline_alg, seed=123, verbose=False)
    return out["x_hist"]


def rmse_for_indices(x_hist: np.ndarray, sys: SystemDef, indices: Optional[list] = None) -> Dict[str,float]:
    indices = sys.metric_indices if indices is None else indices
    vals = {}
    for i in indices:
        if sys.cost_mode == "trajectory":
            ref = sys.ref_traj[i, :x_hist.shape[1]]
        else:
            ref = np.ones(x_hist.shape[1]) * sys.reference_at(0)[i]
        vals[sys.state_names[i]] = float(np.sqrt(np.mean((x_hist[i] - ref)**2)))
    vals["mean_metric_rmse"] = float(np.mean(list(vals.values()))) if vals else float("nan")
    return vals


def normalized_rmse(x_hist: np.ndarray, sys: SystemDef, indices: Optional[list] = None,
                    x0: Optional[np.ndarray] = None) -> float:
    """Normalized closed-loop RMSE over the evaluated states.

    x0 defaults to the system's nominal initial condition. It MUST be passed
    explicitly whenever the run used x0_override (multi-initial-condition and
    held-out tuning experiments), because the per-state normalization scale is
    defined by the initial deviation from the reference; using the nominal x0
    for a run started elsewhere silently rescales the metric.
    """
    indices = sys.metric_indices if indices is None else indices
    x0 = sys.x0 if x0 is None else np.asarray(x0, dtype=float)
    scores=[]
    for i in indices:
        if sys.cost_mode == "trajectory":
            ref = sys.ref_traj[i, :x_hist.shape[1]]
            scale = max(np.ptp(ref), 1.0)
        else:
            r = sys.reference_at(0)[i]
            ref = np.ones(x_hist.shape[1])*r
            scale = max(abs(x0[i]-r), abs(r), 1.0)
        scores.append(np.sqrt(np.mean(((x_hist[i]-ref)/scale)**2)))
    return float(np.mean(scores)) if scores else float("nan")
