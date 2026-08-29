#!/usr/bin/env python3
"""Dynamic-loop drivers for SGO, GWO, PSO and WOA on the GMPB benchmark.

The operator equations are transcribed unchanged from the project's static
implementations (``code/sgo.py``, ``gwo.py``, ``pso.py``, ``woa.py``), which are
the same code used for every result in the manuscript.  Nothing in the search
rules is altered for the dynamic benchmark.  Only two things are added, and both
are outside the operators:

  * an iteration is abandoned as soon as the benchmark reports an environmental
    change, because the official ``fitness.m`` refuses further evaluations until
    the algorithm acknowledges it;
  * the annealing schedules that the static versions index by ``it/max_iter``
    (``a`` in GWO and WOA, ``w`` in PSO) are indexed by progress through the
    current environment, since a dynamic run has no global iteration budget.
    The same rule is applied to all three, and SGO has no such schedule.

GMPB is a maximisation benchmark.  The operators are written for minimisation,
so the drivers minimise ``cost = -fitness``; this is an exact relabelling and
leaves every published equation untouched.

``code/gmpb_sgo_fidelity.py`` checks the SGO driver against ``code/sgo.py``
evaluation by evaluation on a static objective.
"""
import numpy as np

INF = float("inf")


class LegacyRNG:
    """Adapter exposing the Generator API on top of ``numpy.random.RandomState``.

    The static implementations in ``code/sgo.py`` and friends draw from the
    legacy ``numpy.random`` functions.  Using the same underlying stream here is
    what allows ``code/gmpb_sgo_fidelity.py`` to compare the dynamic driver with
    the published implementation evaluation by evaluation.
    """

    def __init__(self, seed):
        self._rs = np.random.RandomState(seed)

    def random(self, size=None):
        if size is None:
            return self._rs.rand()
        if isinstance(size, tuple):
            return self._rs.rand(*size)
        return self._rs.rand(size)

    def permutation(self, n):
        return self._rs.permutation(n)

    def integers(self, n):
        return self._rs.randint(n)


class _Base:
    """Common driver scaffolding."""

    name = "BASE"

    def __init__(self, problem, rng, NP):
        self.p = problem
        self.rng = rng
        self.NP = NP
        self.d = problem.d
        self.lb = np.full(problem.d, problem.lb, dtype=float)
        self.ub = np.full(problem.d, problem.ub, dtype=float)
        self.best_x = None
        self.best_cost = INF
        self.trace = []          # (FE, cost) after every accepted improvement

    # -- evaluation helpers -------------------------------------------------
    def _cost(self, X):
        """Evaluate rows of X, returning minimisation costs (NaN -> +inf)."""
        X = np.atleast_2d(X)
        f = self.p.evaluate(X)
        c = -f
        c[np.isnan(c)] = INF
        return c

    def _note_best(self, X, c):
        i = int(np.argmin(c))
        if c[i] < self.best_cost:
            self.best_cost = float(c[i])
            self.best_x = np.array(X[i], dtype=float, copy=True)

    @property
    def env_progress(self):
        """Fraction of the current environment's budget already consumed."""
        cf = self.p.change_frequency
        return (self.p.FE % cf) / cf

    # -- interface ----------------------------------------------------------
    def initialize(self, seed_x=None):
        raise NotImplementedError

    def iteration(self):
        raise NotImplementedError

    def reevaluate(self):
        """Informed-change reaction shared by all four optimizers.

        The retained population is re-evaluated under the new environment, which
        consumes budget exactly as the official EDOLAB change reactions do.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SGO -- Squid Game Optimizer, Azizi et al. (2023), Sci. Rep. 13:5373
# Transcribed from code/sgo.py without changing any operator.
# ---------------------------------------------------------------------------
class SGO(_Base):
    name = "SGO"

    def __init__(self, problem, rng, NP):
        if NP % 2 != 0:
            NP += 1
        super().__init__(problem, rng, NP)
        self.m = NP // 2

    def initialize(self, seed_x=None):
        self.X = self.lb + self.rng.random((self.NP, self.d)) * (self.ub - self.lb)
        if seed_x is not None:
            self.X[0] = np.clip(seed_x, self.lb, self.ub)
        self.costs = self._cost(self.X)
        bi = int(np.argmin(self.costs))
        self.BS = self.X[bi].copy()
        self.best_cost_env = float(self.costs[bi])
        self._note_best(self.X, self.costs)

    def reevaluate(self):
        self.costs = self._cost(self.X)
        finite = np.isfinite(self.costs)
        if finite.any():
            bi = int(np.argmin(np.where(finite, self.costs, INF)))
            self.BS = self.X[bi].copy()
            self.best_cost_env = float(self.costs[bi])
            self._note_best(self.X, self.costs)
        else:
            self.best_cost_env = INF

    def iteration(self):
        p, rng, m = self.p, self.rng, self.m
        perm = rng.permutation(self.NP)
        off_idx, def_idx = perm[:m], perm[m:]
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


# ---------------------------------------------------------------------------
# GWO -- Mirjalili et al. (2014).  Transcribed from code/gwo.py.
# ---------------------------------------------------------------------------
class GWO(_Base):
    name = "GWO"

    def initialize(self, seed_x=None):
        self.X = self.lb + self.rng.random((self.NP, self.d)) * (self.ub - self.lb)
        if seed_x is not None:
            self.X[0] = np.clip(seed_x, self.lb, self.ub)
        self.costs = self._cost(self.X)
        self._rank()
        self._note_best(self.X, self.costs)

    def _rank(self):
        order = np.argsort(self.costs)
        self.alpha_pos, self.alpha_score = self.X[order[0]].copy(), float(self.costs[order[0]])
        b = order[1] if self.NP > 1 else order[0]
        dl = order[2] if self.NP > 2 else b
        self.beta_pos, self.beta_score = self.X[b].copy(), float(self.costs[b])
        self.delta_pos, self.delta_score = self.X[dl].copy(), float(self.costs[dl])

    def reevaluate(self):
        self.costs = self._cost(self.X)
        self.costs[~np.isfinite(self.costs)] = INF
        self._rank()
        self._note_best(self.X, self.costs)

    def iteration(self):
        p, rng = self.p, self.rng
        a = 2.0 - 2.0 * self.env_progress          # 2 -> 0 within the environment
        Xn = np.empty_like(self.X)
        for i in range(self.NP):
            r1 = rng.random(self.d); r2 = rng.random(self.d)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = self.alpha_pos - A1 * np.abs(C1 * self.alpha_pos - self.X[i])
            r1 = rng.random(self.d); r2 = rng.random(self.d)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = self.beta_pos - A2 * np.abs(C2 * self.beta_pos - self.X[i])
            r1 = rng.random(self.d); r2 = rng.random(self.d)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = self.delta_pos - A3 * np.abs(C3 * self.delta_pos - self.X[i])
            Xn[i] = np.clip((X1 + X2 + X3) / 3.0, self.lb, self.ub)
        self.X = Xn
        self.costs = self._cost(self.X)
        self.costs[~np.isfinite(self.costs)] = INF
        for i in range(self.NP):
            c = self.costs[i]
            if c < self.alpha_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos
                self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos
                self.alpha_score, self.alpha_pos = float(c), self.X[i].copy()
            elif c < self.beta_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos
                self.beta_score, self.beta_pos = float(c), self.X[i].copy()
            elif c < self.delta_score:
                self.delta_score, self.delta_pos = float(c), self.X[i].copy()
        self._note_best(self.X, self.costs)


# ---------------------------------------------------------------------------
# PSO -- Kennedy & Eberhart (1995) with Shi & Eberhart (1998) inertia.
# Transcribed from code/pso.py.
# ---------------------------------------------------------------------------
class PSO(_Base):
    name = "PSO"
    W_MAX, W_MIN, C1, C2 = 0.9, 0.4, 2.0, 2.0

    def initialize(self, seed_x=None):
        self.X = self.lb + self.rng.random((self.NP, self.d)) * (self.ub - self.lb)
        if seed_x is not None:
            self.X[0] = np.clip(seed_x, self.lb, self.ub)
        self.vmax = (self.ub - self.lb) * 0.5
        self.V = -self.vmax + self.rng.random((self.NP, self.d)) * 2 * self.vmax
        self.costs = self._cost(self.X)
        self.costs[~np.isfinite(self.costs)] = INF
        self.P = self.X.copy()
        self.Pc = self.costs.copy()
        g = int(np.argmin(self.Pc))
        self.G, self.Gc = self.P[g].copy(), float(self.Pc[g])
        self._note_best(self.X, self.costs)

    def reevaluate(self):
        self.Pc = self._cost(self.P)
        self.Pc[~np.isfinite(self.Pc)] = INF
        g = int(np.argmin(self.Pc))
        self.G, self.Gc = self.P[g].copy(), float(self.Pc[g])
        self.costs = self.Pc.copy()
        self._note_best(self.P, self.Pc)

    def iteration(self):
        rng = self.rng
        w = self.W_MAX - (self.W_MAX - self.W_MIN) * self.env_progress
        r1 = rng.random((self.NP, self.d))
        r2 = rng.random((self.NP, self.d))
        self.V = (w * self.V
                  + self.C1 * r1 * (self.P - self.X)
                  + self.C2 * r2 * (self.G - self.X))
        self.V = np.clip(self.V, -self.vmax, self.vmax)
        self.X = np.clip(self.X + self.V, self.lb, self.ub)
        self.costs = self._cost(self.X)
        self.costs[~np.isfinite(self.costs)] = INF
        imp = self.costs < self.Pc
        self.P[imp] = self.X[imp]
        self.Pc[imp] = self.costs[imp]
        g = int(np.argmin(self.Pc))
        if self.Pc[g] < self.Gc:
            self.G, self.Gc = self.P[g].copy(), float(self.Pc[g])
        self._note_best(self.X, self.costs)


# ---------------------------------------------------------------------------
# WOA -- Mirjalili & Lewis (2016).  Transcribed from code/woa.py.
# ---------------------------------------------------------------------------
class WOA(_Base):
    name = "WOA"
    B = 1.0

    def initialize(self, seed_x=None):
        self.X = self.lb + self.rng.random((self.NP, self.d)) * (self.ub - self.lb)
        if seed_x is not None:
            self.X[0] = np.clip(seed_x, self.lb, self.ub)
        self.costs = self._cost(self.X)
        self.costs[~np.isfinite(self.costs)] = INF
        i = int(np.argmin(self.costs))
        self.Xstar, self.star_cost = self.X[i].copy(), float(self.costs[i])
        self._note_best(self.X, self.costs)

    def reevaluate(self):
        self.costs = self._cost(self.X)
        self.costs[~np.isfinite(self.costs)] = INF
        i = int(np.argmin(self.costs))
        self.Xstar, self.star_cost = self.X[i].copy(), float(self.costs[i])
        self._note_best(self.X, self.costs)

    def iteration(self):
        p, rng = self.p, self.rng
        prog = self.env_progress
        a = 2.0 - 2.0 * prog
        a2 = -1.0 - prog
        for i in range(self.NP):
            if p.recent_change or p.finished:
                break
            r1, r2 = rng.random(), rng.random()
            A, C = 2 * a * r1 - a, 2 * r2
            pr = rng.random()
            l = (a2 - 1) * rng.random() + 1
            if pr < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * self.Xstar - self.X[i])
                    Xnew = self.Xstar - A * D
                else:
                    Xr = self.X[rng.integers(self.NP)]
                    D = np.abs(C * Xr - self.X[i])
                    Xnew = Xr - A * D
            else:
                Ds = np.abs(self.Xstar - self.X[i])
                Xnew = Ds * np.exp(self.B * l) * np.cos(2 * np.pi * l) + self.Xstar
            self.X[i] = np.clip(Xnew, self.lb, self.ub)
            c = float(self._cost(self.X[i][None, :])[0])
            self.costs[i] = c if np.isfinite(c) else INF
        i2 = int(np.argmin(self.costs))
        if self.costs[i2] < self.star_cost:
            self.star_cost = float(self.costs[i2])
            self.Xstar = self.X[i2].copy()
        self._note_best(self.X, self.costs)


OPTIMIZERS = {"SGO": SGO, "GWO": GWO, "PSO": PSO, "WOA": WOA}
