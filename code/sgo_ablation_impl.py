"""Ablation-capable copy of SGO for Phase-2C diagnostics.

`code/sgo.py` is left BYTE-IDENTICAL to the submitted version and remains the
implementation used by every production experiment. This module mirrors it line
for line and adds keyword flags whose defaults reproduce it exactly, including
the random-number consumption order. `sgo_ablation.py` asserts that equivalence
on a seed-matched test set before any ablation result is used.

Flags (defaults = frozen behaviour):

    greedy_accept=True   Trials from Eq. 9 / Eq. 12 / Eq. 13 replace the
                         incumbent only if strictly better (decision D3).
                         False = the literal paper reading, unconditional
                         replacement. This is D3's documented rejected
                         alternative, not an invented rule.

    bridge_stage=True    Run the Eq. 13 bridge-passing stage (decisions D4/D5).
                         False = skip that stage entirely. Under a fixed
                         evaluation budget the freed evaluations are spent on
                         further iterations, so the comparison stays equal-FE.

    grouping="random"    Fresh uniformly random permutation each iteration
                         (decision D11). "fitness" = best half offensive, worst
                         half defensive, which is D11's documented rejected
                         alternative.

                         NOTE: the Phase-2C brief described the frozen grouping
                         as "cost-based" and proposed random grouping as the
                         ablation. That premise is inverted: sgo.py:50 already
                         groups at random, independently of fitness. The
                         meaningful predeclared alternative is therefore the
                         fitness-based split, and that is what is implemented.

    warm_slot=True       Insert the supplied warm start into population slot 0.
                         False = leave the population fully random. Only
                         meaningful when a warm start is supplied.

Nothing else differs from sgo.py.
"""
import numpy as np
import time


def sgo_ablatable(cost_fn, NP, max_iter, lb, ub, x0_warm=None, tol=1e-10,
                  stagnant_limit=15, max_evals=None,
                  greedy_accept=True, bridge_stage=True, grouping="random",
                  warm_slot=True):
    if NP % 2 != 0:
        NP += 1
    lb = np.asarray(lb, dtype=float); ub = np.asarray(ub, dtype=float)
    d = len(lb); m = NP // 2
    t0 = time.perf_counter(); evals = 0

    def eval_cost(x):
        nonlocal evals
        evals += 1
        return float(cost_fn(np.asarray(x, dtype=float)))

    def budget_left():
        return max_evals is None or evals < max_evals

    X = lb + np.random.rand(NP, d) * (ub - lb)
    if x0_warm is not None and warm_slot:
        X[0] = np.clip(x0_warm, lb, ub)
    costs = np.empty(NP)
    for i in range(NP):
        costs[i] = eval_cost(X[i])
    bi = int(np.argmin(costs)); BS = X[bi].copy(); best_cost = float(costs[bi])
    cost_hist = []; stagnant = 0; prev_best = best_cost

    for it in range(max_iter):
        if not budget_left(): break
        if grouping == "fitness":
            order = np.argsort(costs, kind="stable")
            off_idx, def_idx = order[:m], order[m:]
        else:
            perm = np.random.permutation(NP); off_idx = perm[:m]; def_idx = perm[m:]
        X_off = X[off_idx].copy(); X_def = X[def_idx].copy()
        c_off = costs[off_idx].copy(); c_def = costs[def_idx].copy()
        DG = X_def.mean(axis=0); OG = X_off.mean(axis=0)
        sog_slots = []; sog_pos = []; sog_cost = []; sdg_pos = []

        for i in range(m):
            if not budget_left(): break
            r3 = np.random.randint(m)
            r1, r2 = np.random.rand(2)
            X_O1 = (X_off[i] + r1 * DG - r2 * X_def[r3]) / 2.0
            X_O1 = np.clip(X_O1, lb, ub)
            WS_off = c_off[i]; WS_def = c_def[r3]
            if WS_def <= WS_off:
                c_O1 = eval_cost(X_O1)
                X_best_local = X_O1.copy(); c_best_local = c_O1
                sog_temp = sog_pos + [X_O1.copy()]
                SOG_mean = np.mean(sog_temp, axis=0)
                if budget_left():
                    r1b, r2b = np.random.rand(2)
                    X_O2 = X_O1 + r1b * SOG_mean - r2b * BS
                    X_O2 = np.clip(X_O2, lb, ub)
                    c_O2 = eval_cost(X_O2)
                    if (c_O2 < c_best_local) or (not greedy_accept):
                        X_best_local = X_O2.copy(); c_best_local = c_O2
                X_off[i] = X_best_local; c_off[i] = c_best_local
                sog_slots.append(i); sog_pos.append(X_best_local.copy())
                sog_cost.append(c_best_local)
            else:
                sdg_pos.append(X_def[r3].copy())
                if budget_left():
                    r4 = np.random.randint(m); r1d, r2d = np.random.rand(2)
                    X_D1 = X_def[r3] + r1d * OG - r2d * X_off[r4]
                    X_D1 = np.clip(X_D1, lb, ub)
                    c_D1 = eval_cost(X_D1)
                    if (c_D1 < c_def[r3]) or (not greedy_accept):
                        X_def[r3] = X_D1; c_def[r3] = c_D1

        if bridge_stage and sog_pos and budget_left():
            for jj, slot in enumerate(sog_slots):
                if not budget_left(): break
                if sdg_pos:
                    X_scd = sdg_pos[np.random.randint(len(sdg_pos))]
                else:
                    X_scd = X_def[np.random.randint(m)]
                r1b, r2b = np.random.rand(2)
                X_O3 = sog_pos[jj] + r1b * BS - r2b * X_scd
                X_O3 = np.clip(X_O3, lb, ub)
                c_O3 = eval_cost(X_O3)
                if (c_O3 < sog_cost[jj]) or (not greedy_accept):
                    X_off[slot] = X_O3; c_off[slot] = c_O3

        X[off_idx] = X_off; X[def_idx] = X_def
        costs[off_idx] = c_off; costs[def_idx] = c_def
        bi2 = int(np.argmin(costs))
        if costs[bi2] < best_cost:
            best_cost = float(costs[bi2]); BS = X[bi2].copy()
        cost_hist.append(best_cost)
        if abs(prev_best - best_cost) < tol: stagnant += 1
        else: stagnant = 0
        prev_best = best_cost
        if stagnant >= stagnant_limit: break
    return BS, best_cost, np.asarray(cost_hist), len(cost_hist), time.perf_counter() - t0


VARIANTS = {
    "baseline":        dict(),
    "A_no_greedy":     dict(greedy_accept=False),
    "B_no_bridge":     dict(bridge_stage=False),
    "C_fitness_group": dict(grouping="fitness"),
    "D_no_warm_slot":  dict(warm_slot=False),
}
