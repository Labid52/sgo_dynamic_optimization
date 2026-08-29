#!/usr/bin/env python3
"""Python port of the official GMPB benchmark (EDOLAB) for the dynamic study.

The *landscape itself* is not regenerated here.  Every environment sequence is
produced by the unmodified official Octave/MATLAB generator
(``BenchmarkGenerator_GMPB.m``) and dumped to a ``.mat`` file; this module only
loads that state and reimplements

    * ``fitness_GMPB.m``  -- the peak evaluation, and
    * ``fitness.m``       -- the FE counter, environment change and error
                             bookkeeping,

both transcribed line for line from the official sources.  Equivalence against
official outputs is checked by ``code/gmpb_equivalence.py``; nothing in this
file may be used before that gate passes.

Official sources (EDOLAB-platform/EDOLAB-MATLAB, downloaded 2026-08-22):
    Benchmark/GMPB/BenchmarkGenerator_GMPB.m   sha256 11ffbfc2...
    Benchmark/GMPB/fitness_GMPB.m              sha256 1bbf2fd8...
    Benchmark/fitness.m                        sha256 0b7f479c...
"""
import os
import numpy as np
import scipy.io as sio


class BudgetExhausted(Exception):
    """Raised when the run has consumed its full MaxEvals budget."""


# Official competition instance table, verified against
# https://competition-hub.github.io/GMPB-Competition/  (IEEE CEC 2025).
CASES = {
    "F1":  dict(PeakNumber=5,   ChangeFrequency=5000, Dimension=5,  ShiftSeverity=1),
    "F2":  dict(PeakNumber=10,  ChangeFrequency=5000, Dimension=5,  ShiftSeverity=1),
    "F3":  dict(PeakNumber=25,  ChangeFrequency=5000, Dimension=5,  ShiftSeverity=1),
    "F4":  dict(PeakNumber=50,  ChangeFrequency=5000, Dimension=5,  ShiftSeverity=1),
    "F5":  dict(PeakNumber=100, ChangeFrequency=5000, Dimension=5,  ShiftSeverity=1),
    "F6":  dict(PeakNumber=10,  ChangeFrequency=2500, Dimension=5,  ShiftSeverity=1),
    "F7":  dict(PeakNumber=10,  ChangeFrequency=1000, Dimension=5,  ShiftSeverity=1),
    "F8":  dict(PeakNumber=10,  ChangeFrequency=500,  Dimension=5,  ShiftSeverity=1),
    "F9":  dict(PeakNumber=10,  ChangeFrequency=5000, Dimension=10, ShiftSeverity=1),
    "F10": dict(PeakNumber=10,  ChangeFrequency=5000, Dimension=20, ShiftSeverity=1),
    "F11": dict(PeakNumber=10,  ChangeFrequency=5000, Dimension=5,  ShiftSeverity=2),
    "F12": dict(PeakNumber=10,  ChangeFrequency=5000, Dimension=5,  ShiftSeverity=5),
}
CASE_LIST = [f"F{i}" for i in range(1, 13)]


def _transform(X, tau, eta):
    """Transcription of Transform() in fitness_GMPB.m.

    X   : (..., d) array
    tau : broadcastable to (..., 1)
    eta : (..., 4)
    """
    pos = X > 0
    neg = X < 0
    e1 = eta[..., 0:1]; e2 = eta[..., 1:2]
    e3 = eta[..., 2:3]; e4 = eta[..., 3:4]
    Y = X.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(np.abs(np.where(pos | neg, X, 1.0)))
        vp = np.exp(lg + tau * (np.sin(e1 * lg) + np.sin(e2 * lg)))
        vn = -np.exp(lg + tau * (np.sin(e3 * lg) + np.sin(e4 * lg)))
    np.copyto(Y, vp, where=pos)
    np.copyto(Y, vn, where=neg)
    return Y


class GMPB:
    """One official GMPB instance, loaded from an Octave-generated state file."""

    def __init__(self, state_path):
        raw = sio.loadmat(state_path, squeeze_me=True, struct_as_record=False)["state"]
        self.case = str(raw.case)
        self.seed = int(raw.seed)
        self.d = int(raw.Dimension)
        self.m = int(raw.PeakNumber)
        self.T = int(raw.EnvironmentNumber)
        self.change_frequency = int(raw.ChangeFrequency)
        self.shift_severity = float(raw.ShiftSeverity)
        self.lb = float(raw.MinCoordinate)
        self.ub = float(raw.MaxCoordinate)
        self.max_evals = int(raw.MaxEvals)
        self.peaks_position = np.atleast_3d(np.asarray(raw.PeaksPosition, float)).reshape(self.T, self.m, self.d)
        self.peaks_width = np.asarray(raw.PeaksWidth, float).reshape(self.T, self.m, self.d)
        self.peaks_height = np.asarray(raw.PeaksHeight, float).reshape(self.T, self.m)
        self.tau = np.asarray(raw.tau, float).reshape(self.T, self.m)
        self.eta = np.asarray(raw.eta, float).reshape(self.T, self.m, 4)
        self.rotation = np.asarray(raw.RotationMatrix, float).reshape(self.T, self.m, self.d, self.d)
        self.optimum_value = np.asarray(raw.OptimumValue, float).reshape(self.T)
        self.optimum_id = np.asarray(raw.OptimumID, float).reshape(self.T).astype(int)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Reset the run state (FE counter, environment, error trace)."""
        self.FE = 0
        self.env = 1                       # 1-based, as in the official code
        self.recent_change = False
        self.current_error = np.full(self.max_evals, np.nan)
        self.current_performance = np.full(self.max_evals, np.nan)
        self.ebbc = np.full(self.T, np.nan)

    # ------------------------------------------------------------------
    def _env_arrays(self, t):
        """Cache the per-environment landscape arrays (pure lookup, no maths)."""
        if getattr(self, "_cache_t", None) != t:
            self._cache_t = t
            self._cc = self.peaks_position[t]
            self._RR = self.rotation[t]
            self._WW = self.peaks_width[t]
            self._hh = self.peaks_height[t]
            self._tt = self.tau[t][None, :, None]
            self._ee = self.eta[t][None, :, :]
            self._Rt = np.ascontiguousarray(self._RR.transpose(0, 2, 1))
        return self._cc, self._RR, self._WW, self._hh, self._tt, self._ee

    def raw_fitness(self, X, env=None):
        """Transcription of fitness_GMPB.m, vectorised over rows of X.

        The official code forms

            a = Transform( (x-c)' R' ),   b = Transform( R (x-c) )

        Both arguments are the same vector, since element k of (x-c)' R' is
        sum_j (x-c)_j R_kj, which is element k of R (x-c).  So a == b and the
        official product  a * diag(W) * b  equals  sum_k W_k a_k^2.  The
        equivalence gate checks this against the official code directly.

        Consumes no budget and changes no state.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        t = (self.env if env is None else env) - 1
        c, R, W, h, tau, eta = self._env_arrays(t)
        diff = X[:, None, :] - c[None, :, :]             # (n, m, d)
        u = np.matmul(diff.transpose(1, 0, 2), self._Rt).transpose(1, 0, 2)
        a = _transform(u, tau, eta)
        inner = np.einsum("nmk,mk,nmk->nm", a, W, a)
        with np.errstate(invalid="ignore"):
            f = h[None, :] - np.sqrt(inner)
        return np.max(f, axis=1)

    # ------------------------------------------------------------------
    def evaluate(self, X):
        """Transcription of fitness.m: budgeted, change-aware evaluation.

        Returns an array of length ``len(X)``; entries after an environmental
        change (or after the budget is exhausted) are NaN, exactly as the
        official wrapper leaves them.  A chunk never spans an environment
        boundary, so the running-minimum bookkeeping below is equivalent to the
        official scalar loop.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        out = np.full(n, np.nan)
        if self.FE >= self.max_evals or self.recent_change:
            return out
        cf, me = self.change_frequency, self.max_evals
        done = 0
        while done < n:
            room = min(me - self.FE, cf - (self.FE % cf), n - done)
            if room <= 0:
                break
            sl = slice(done, done + room)
            vals = self.raw_fitness(X[sl])
            out[sl] = vals
            opt = self.optimum_value[self.env - 1]
            err = opt - vals
            i0 = self.FE                      # 0-based index of the first new FE
            run = np.minimum.accumulate(err)
            if i0 % cf != 0:                  # not the first FE of an environment
                run = np.minimum(run, self.current_error[i0 - 1])
            self.current_error[i0:i0 + room] = run
            # CurrentPerformance is the fitness of the best-so-far point, so it
            # is exactly opt - CurrentError inside the environment.
            self.current_performance[i0:i0 + room] = opt - run
            self.FE += room
            done += room
            # Ebbc is recorded at the FE with rem(FE, cf) == cf - 1
            tgt = (i0 // cf) * cf + (cf - 1)
            if i0 < tgt <= self.FE:
                self.ebbc[self.env - 1] = self.current_error[tgt - 1]
            if self.FE % cf == 0 and self.FE < me:
                self.env += 1
                self.recent_change = True
                break
            if self.FE >= me:
                break
        return out

    # ------------------------------------------------------------------
    def acknowledge_change(self):
        """Clear the change flag, as every EDOLAB algorithm does in its main loop."""
        self.recent_change = False

    @property
    def finished(self):
        return self.FE >= self.max_evals

    @property
    def offline_error(self):
        """E_o = mean(Problem.CurrentError) over all MaxEvals evaluations."""
        return float(np.mean(self.current_error))

    @property
    def best_error_before_change(self):
        """E_bbc = mean(Problem.Ebbc)."""
        return float(np.mean(self.ebbc))


def state_path(cache_dir, case, seed):
    return os.path.join(cache_dir, f"{case}_seed{seed}_state.mat")


def load(cache_dir, case, seed):
    return GMPB(state_path(cache_dir, case, seed))
