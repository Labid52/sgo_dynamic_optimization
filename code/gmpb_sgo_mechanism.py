#!/usr/bin/env python3
"""Grouping-mechanism ablation of the Squid Game Optimizer on GMPB.

Diagnostic only.  This is not a new algorithm and is never named as one; it is
the *fitness-based grouping ablation* of the Squid Game Optimizer of Azizi,
Baghalzadeh Shishehgarkhaneh, Basiri and Moehler, Scientific Reports 13:5373
(2023).

The class below subclasses the validated dynamic driver
``code/gmpb_optimizers.py:SGO`` and overrides ``iteration`` with a verbatim copy
of the parent body in which **only the two grouping lines** are parameterised:

    grouping="random"    perm = rng.permutation(NP); best half is arbitrary
    grouping="fitness"   order = argsort(costs, kind="stable"); best half
                         offensive, worst half defensive

That is exactly the intervention already committed as ``C_fitness_group`` in
``code/sgo_ablation_impl.py`` and used for the flight ablation.  No other
operator, constant, bound or acceptance rule differs, and no parameter is added.

With ``grouping="random"`` the class must reproduce the validated driver bit for
bit; ``code/gmpb_mechanism_equivalence.py`` is the gate that proves it.  Note
that ``grouping="fitness"`` skips the ``rng.permutation`` draw, so the random
number streams necessarily diverge from the first iteration onwards.  Runs are
paired by the benchmark environment trajectory, not by the optimizer stream.

Neither ``code/sgo.py`` nor ``code/gmpb_optimizers.py`` is modified.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmpb_optimizers import SGO, INF   # noqa: E402


class SGOGrouping(SGO):
    """Validated dynamic SGO with the grouping rule exposed as an ablation switch."""

    name = "SGO"

    def __init__(self, problem, rng, NP, grouping="random"):
        if grouping not in ("random", "fitness"):
            raise ValueError(f"unknown grouping {grouping!r}")
        super().__init__(problem, rng, NP)
        self.grouping = grouping

    # ------------------------------------------------------------------
    def _groups(self):
        """The only difference between the baseline and the intervention."""
        m = self.m
        if self.grouping == "fitness":
            order = np.argsort(self.costs, kind="stable")
            return order[:m], order[m:]
        perm = self.rng.permutation(self.NP)
        return perm[:m], perm[m:]

    # ------------------------------------------------------------------
    def iteration(self):
        # Verbatim copy of gmpb_optimizers.SGO.iteration with the two grouping
        # lines replaced by self._groups().  Nothing else differs.
        p, rng, m = self.p, self.rng, self.m
        off_idx, def_idx = self._groups()
        X_off, X_def = self.X[off_idx].copy(), self.X[def_idx].copy()
        c_off, c_def = self.costs[off_idx].copy(), self.costs[def_idx].copy()
        DG, OG = X_def.mean(axis=0), X_off.mean(axis=0)
        sog_slots, sog_pos, sog_cost, sdg_pos = [], [], [], []

        for i in range(m):
            if p.recent_change or p.finished:
                break
            r3 = rng.integers(m)
            r1, r2 = rng.random(2)
            X_O1 = (X_off[i] + r1 * DG - r2 * X_def[r3]) / 2.0
            X_O1 = np.clip(X_O1, self.lb, self.ub)
            WS_off, WS_def = c_off[i], c_def[r3]
            if WS_def <= WS_off:
                c_O1 = float(self._cost(X_O1[None, :])[0])
                X_best_local, c_best_local = X_O1.copy(), c_O1
                sog_temp = sog_pos + [X_O1.copy()]
                SOG_mean = np.mean(sog_temp, axis=0)
                if not (p.recent_change or p.finished):
                    r1b, r2b = rng.random(2)
                    X_O2 = X_O1 + r1b * SOG_mean - r2b * self.BS
                    X_O2 = np.clip(X_O2, self.lb, self.ub)
                    c_O2 = float(self._cost(X_O2[None, :])[0])
                    if c_O2 < c_best_local:
                        X_best_local, c_best_local = X_O2.copy(), c_O2
                X_off[i], c_off[i] = X_best_local, c_best_local
                sog_slots.append(i)
                sog_pos.append(X_best_local.copy())
                sog_cost.append(c_best_local)
            else:
                sdg_pos.append(X_def[r3].copy())
                if not (p.recent_change or p.finished):
                    r4 = rng.integers(m)
                    r1d, r2d = rng.random(2)
                    X_D1 = X_def[r3] + r1d * OG - r2d * X_off[r4]
                    X_D1 = np.clip(X_D1, self.lb, self.ub)
                    c_D1 = float(self._cost(X_D1[None, :])[0])
                    if c_D1 < c_def[r3]:
                        X_def[r3], c_def[r3] = X_D1, c_D1

        # Eq. 13 bridge passing
        if sog_pos and not (self.p.recent_change or self.p.finished):
            for jj, slot in enumerate(sog_slots):
                if self.p.recent_change or self.p.finished:
                    break
                if sdg_pos:
                    X_scd = sdg_pos[rng.integers(len(sdg_pos))]
                else:
                    X_scd = X_def[rng.integers(m)]
                r1b, r2b = rng.random(2)
                X_O3 = sog_pos[jj] + r1b * self.BS - r2b * X_scd
                X_O3 = np.clip(X_O3, self.lb, self.ub)
                c_O3 = float(self._cost(X_O3[None, :])[0])
                if c_O3 < sog_cost[jj]:
                    X_off[slot], c_off[slot] = X_O3, c_O3

        self.X[off_idx], self.X[def_idx] = X_off, X_def
        self.costs[off_idx], self.costs[def_idx] = c_off, c_def
        bi2 = int(np.argmin(self.costs))
        if self.costs[bi2] < self.best_cost_env:
            self.best_cost_env = float(self.costs[bi2])
            self.BS = self.X[bi2].copy()
        self._note_best(self.X, self.costs)

    # ------------------------------------------------------------------
    def diversity(self):
        """Normalised positional diversity, same definition as the GMPB study."""
        span = self.p.ub - self.p.lb
        return float(np.mean(np.std(self.X, axis=0)) / span)
