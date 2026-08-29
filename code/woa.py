"""
Whale Optimization Algorithm (WOA)
Mirjalili & Lewis (2016), Advances in Engineering Software 95, 51-67.

Three behaviours mimic humpback whale hunting:
  Encircling prey       (exploitation)
  Bubble-net attacking  (exploitation, spiral update)
  Search for prey       (exploration, random whale)

p ~ U[0,1]: if p < 0.5 → encircling or spiral; else → random search
A  = 2a·r - a,  a linearly decreases 2 → 0
"""

import numpy as np
import time


def woa(cost_fn, NP, max_iter, lb, ub,
        x0_warm=None, tol=1e-10, stagnant_limit=15, b=1.0, max_evals=None):
    """
    Standard WOA for bound-constrained minimisation.

    Parameters
    ----------
    cost_fn  : callable
    NP       : int   population size
    max_iter : int
    lb, ub   : array-like
    x0_warm  : optional warm-start
    b        : float  spiral shape constant (default 1)

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

    X = lb + np.random.rand(NP, d) * (ub - lb)
    if x0_warm is not None:
        X[0] = np.clip(x0_warm, lb, ub)

    costs     = np.array([eval_cost(X[i]) for i in range(NP)])
    best_idx  = np.argmin(costs)
    X_star    = X[best_idx].copy()
    best_cost = float(costs[best_idx])

    cost_hist = [best_cost]
    stagnant  = 0
    prev_best = best_cost

    for it in range(max_iter):
        a  = 2.0 - 2.0 * it / max_iter   # decreases 2 → 0
        a2 = -1.0 - it / max_iter         # for spiral: -1 → -2

        for i in range(NP):
            if max_evals is not None and evals >= max_evals:
                break
            r1, r2 = np.random.rand(), np.random.rand()
            A  = 2 * a * r1 - a
            C  = 2 * r2
            p  = np.random.rand()
            l  = (a2 - 1) * np.random.rand() + 1   # l in [a2, 1]

            if p < 0.5:
                if abs(A) < 1:
                    # Encircling prey
                    D     = abs(C * X_star - X[i])
                    X_new = X_star - A * D
                else:
                    # Search for prey (random whale)
                    X_rand = X[np.random.randint(NP)]
                    D      = abs(C * X_rand - X[i])
                    X_new  = X_rand - A * D
            else:
                # Bubble-net: spiral update
                D_star = abs(X_star - X[i])
                X_new  = D_star * np.exp(b * l) * np.cos(2 * np.pi * l) + X_star

            X[i] = np.clip(X_new, lb, ub)
            costs[i] = eval_cost(X[i])

        best_idx2 = np.argmin(costs)
        if costs[best_idx2] < best_cost:
            best_cost = float(costs[best_idx2])
            X_star    = X[best_idx2].copy()

        cost_hist.append(best_cost)

        if abs(prev_best - best_cost) < tol:
            stagnant += 1
        else:
            stagnant = 0
        prev_best = best_cost
        if stagnant >= stagnant_limit:
            break

    return X_star, best_cost, np.array(cost_hist), len(cost_hist), time.perf_counter() - t0
