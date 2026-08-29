"""
Particle Swarm Optimization (PSO)
Kennedy & Eberhart (1995) with inertia weight (Shi & Eberhart 1998).

w  linearly decreases from w_max to w_min over iterations (common standard).
c1 = cognitive coefficient (particle best)
c2 = social coefficient   (global best)
"""

import numpy as np
import time


def pso(cost_fn, NP, max_iter, lb, ub,
        x0_warm=None, tol=1e-10, stagnant_limit=15,
        w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, max_evals=None):
    """
    Standard PSO for bound-constrained minimisation.

    Parameters
    ----------
    cost_fn       : callable  f(x: 1-D ndarray) -> float
    NP            : int       swarm size
    max_iter      : int       maximum iterations
    lb, ub        : array-like
    x0_warm       : 1-D ndarray  optional warm-start (placed in particle 0)
    w_max, w_min  : float  inertia weight range
    c1, c2        : float  acceleration coefficients

    Returns
    -------
    best_sol, best_cost, cost_hist, n_iters, wall_time
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    d  = len(lb)
    t0 = time.perf_counter()
    evals = 0
    def eval_cost(z):
        nonlocal evals
        evals += 1
        return float(cost_fn(z))

    # Initialise positions and velocities
    X = lb + np.random.rand(NP, d) * (ub - lb)
    if x0_warm is not None:
        X[0] = np.clip(x0_warm, lb, ub)

    v_max = (ub - lb) * 0.5
    V = -v_max + np.random.rand(NP, d) * 2 * v_max

    costs  = np.array([eval_cost(X[i]) for i in range(NP)])
    p_best = X.copy()          # personal best positions
    p_cost = costs.copy()      # personal best costs
    g_idx  = np.argmin(p_cost)
    g_best = p_best[g_idx].copy()
    best_cost = float(p_cost[g_idx])

    cost_hist = [best_cost]
    stagnant  = 0
    prev_best = best_cost

    for it in range(max_iter):
        w = w_max - (w_max - w_min) * it / max_iter   # linear decay

        r1 = np.random.rand(NP, d)
        r2 = np.random.rand(NP, d)

        V = (w * V
             + c1 * r1 * (p_best - X)
             + c2 * r2 * (g_best - X))

        # Clamp velocity
        V = np.clip(V, -v_max, v_max)
        X = X + V
        X = np.clip(X, lb, ub)

        # Evaluate
        for i in range(NP):
            if max_evals is not None and evals >= max_evals:
                break
            c = eval_cost(X[i])
            if c < p_cost[i]:
                p_cost[i] = c
                p_best[i] = X[i].copy()

        g_idx2 = np.argmin(p_cost)
        if p_cost[g_idx2] < best_cost:
            best_cost = p_cost[g_idx2]
            g_best = p_best[g_idx2].copy()

        cost_hist.append(best_cost)

        if abs(prev_best - best_cost) < tol:
            stagnant += 1
        else:
            stagnant = 0
        prev_best = best_cost
        if stagnant >= stagnant_limit:
            break

    return g_best, best_cost, np.array(cost_hist), len(cost_hist), time.perf_counter() - t0
