"""
Grey Wolf Optimizer (GWO)
Canonical implementation of Mirjalili et al. (2014), Advances in Engineering Software.

FIX (critical, vs. previous project version):
  The previous version reassigned alpha/beta/delta from the CURRENT population
  at the end of every iteration. Because standard GWO replaces positions
  unconditionally (no greedy per-wolf selection), the population best can get
  worse over iterations, so the returned "alpha" could be WORSE than the best
  solution GWO had already found. In frozen-MPC convergence tests this produced
  artificial optimality gaps (e.g., UAV mid-step median normalized gap 4.27e-2
  with 0% convergence; the canonical version reaches ~1e-9 with 100%).

  The canonical Mirjalili source updates the leaders GREEDILY:
    Alpha/Beta/Delta positions are only replaced when a newly evaluated wolf
    improves on them, i.e., they store the three best solutions found so far.
  This file implements that rule, so the returned solution is always the best
  solution evaluated during the run, and best_cost == cost(returned solution).

Pack structure:
  Alpha  - best solution found so far
  Beta   - second best found so far
  Delta  - third best found so far
  Omega  - all population members (updated toward alpha/beta/delta)

Position update (per dimension j):
  D_alpha = |C1 * alpha_j - X_i_j|;   X1_j = alpha_j - A1 * D_alpha
  D_beta  = |C2 * beta_j  - X_i_j|;   X2_j = beta_j  - A2 * D_beta
  D_delta = |C3 * delta_j - X_i_j|;   X3_j = delta_j - A3 * D_delta
  X_i_j   = (X1_j + X2_j + X3_j) / 3

  a = 2 - 2*iter/max_iter   (linearly decreases 2 -> 0)
  A = 2*a*r1 - a            r1, r2 ~ U[0,1]
  C = 2*r2
"""

import numpy as np
import time


def gwo(cost_fn, NP, max_iter, lb, ub, x0_warm=None, tol=1e-10,
        stagnant_limit=15, max_evals=None):
    """
    Canonical Grey Wolf Optimizer for bound-constrained minimisation.

    Parameters
    ----------
    cost_fn        : callable  f(x: 1-D ndarray) -> float
    NP             : int       population size (>= 3)
    max_iter       : int       maximum iterations
    lb, ub         : array-like  lower / upper bounds
    x0_warm        : 1-D ndarray or None  warm-start position (placed in slot 0)
    tol            : float     stagnation tolerance on best cost
    stagnant_limit : int       stop after this many non-improving iterations
    max_evals      : int or None  hard cap on objective evaluations
                     (for equal-function-evaluation studies)

    Returns
    -------
    best_sol  : 1-D ndarray   best solution found over the whole run (= alpha)
    best_cost : float         cost of best_sol (guaranteed consistent)
    cost_hist : 1-D ndarray   best-so-far cost per iteration (length <= max_iter)
    n_iters   : int
    wall_time : float seconds
    """
    if NP < 3:
        raise ValueError("GWO requires NP >= 3.")

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    d  = len(lb)

    t0 = time.perf_counter()
    evals = 0

    def eval_cost(z):
        nonlocal evals
        evals += 1
        return float(cost_fn(z))

    def budget_left():
        return max_evals is None or evals < max_evals

    # -- Initialise population -----------------------------------------------
    X = lb + np.random.rand(NP, d) * (ub - lb)
    if x0_warm is not None:
        X[0] = np.clip(x0_warm, lb, ub)

    costs = np.array([eval_cost(X[i]) for i in range(NP)])

    # -- Greedy leader archive (canonical GWO): three best solutions so far --
    alpha_score = beta_score = delta_score = np.inf
    alpha_pos = beta_pos = delta_pos = None

    def update_leaders(c, x):
        nonlocal alpha_score, beta_score, delta_score
        nonlocal alpha_pos, beta_pos, delta_pos
        if c < alpha_score:
            delta_score, delta_pos = beta_score, beta_pos
            beta_score,  beta_pos  = alpha_score, alpha_pos
            alpha_score, alpha_pos = c, x.copy()
        elif c < beta_score:
            delta_score, delta_pos = beta_score, beta_pos
            beta_score,  beta_pos  = c, x.copy()
        elif c < delta_score:
            delta_score, delta_pos = c, x.copy()

    for i in range(NP):
        update_leaders(costs[i], X[i])

    cost_hist = []
    stagnant  = 0
    prev_best = alpha_score
    n_iters   = 0

    # -- Main loop ------------------------------------------------------------
    for it in range(max_iter):
        if not budget_left():
            break
        a = 2.0 - 2.0 * it / max_iter      # linearly 2 -> 0

        for i in range(NP):
            if not budget_left():
                break
            X_new = np.empty(d)
            for j in range(d):
                # Alpha contribution
                r1, r2 = np.random.rand(2)
                A1, C1 = 2*a*r1 - a, 2*r2
                X1     = alpha_pos[j] - A1 * abs(C1*alpha_pos[j] - X[i, j])

                # Beta contribution
                r1, r2 = np.random.rand(2)
                A2, C2 = 2*a*r1 - a, 2*r2
                X2     = beta_pos[j]  - A2 * abs(C2*beta_pos[j]  - X[i, j])

                # Delta contribution
                r1, r2 = np.random.rand(2)
                A3, C3 = 2*a*r1 - a, 2*r2
                X3     = delta_pos[j] - A3 * abs(C3*delta_pos[j] - X[i, j])

                X_new[j] = (X1 + X2 + X3) / 3.0

            X[i] = np.clip(X_new, lb, ub)
            costs[i] = eval_cost(X[i])
            update_leaders(costs[i], X[i])   # greedy leader update (canonical)

        n_iters = it + 1
        cost_hist.append(alpha_score)

        # Stagnation on the monotone best-so-far cost
        if abs(prev_best - alpha_score) < tol:
            stagnant += 1
        else:
            stagnant = 0
        prev_best = alpha_score
        if stagnant >= stagnant_limit:
            break

    wall_time = time.perf_counter() - t0
    return alpha_pos, float(alpha_score), np.array(cost_hist), n_iters, wall_time
