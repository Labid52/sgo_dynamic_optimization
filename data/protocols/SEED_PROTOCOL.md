# Seed protocol — Phase 2 onward

The submitted code used `np.random.seed(1000 + 17*trial + k)` in the frozen sweep, which collides:
`(trial=1, k=0)` and `(trial=0, k=17)` share a stream. Phase-2 experiments use a collision-free scheme
with the trial index as the fastest-varying term, so that comparable conditions see identical trial
streams and comparisons are paired.

## Scheme

```
seed = SEED_BASE[experiment]
     + 1_000_000 * instance_index      # system / subproblem instance
     +    10_000 * regime_index        # experimental regime (A, B, ...)
     +     1_000 * optimizer_index     # SGO=0, GWO=1, PSO=2, WOA=3
     +             trial_index         # fastest-varying
```

`trial_index < 1000`, `optimizer_index < 10`, `regime_index < 100`, so no field can overflow into the
next and every `(experiment, instance, regime, optimizer, trial)` tuple maps to a unique seed.

Because `trial_index` is the low-order term and every other field is held constant across the
optimizers being compared, optimizer *k*'s trial *t* and optimizer *j*'s trial *t* differ only by the
optimizer field — the comparison is paired at the trial level by construction.

## SEED_BASE registry

| Experiment | Script | SEED_BASE |
|---|---|---|
| Reference certification | `reference_verification.py` | 4 000 000 |
| SGO benchmark verification | `sgo_benchmark_verification.py` | 6 000 000 |
| Strict-FE budget verification | `strict_fe_verification.py` | 7 100 000 |
| Optimizer-only tuning | `tune_optimizer_only.py` | 8 000 000 |
| Primary fixed-MPC experiment | `primary_experiment.py` | 9 000 000 |

Phase-2B experiments must register new bases here rather than reusing an existing one.

## Legacy seeds

Two places deliberately keep the *submitted* seeding, because their whole purpose is to reproduce
submitted numbers, and reseeding would make a mismatch un-attributable:

- `regression_check.py` — replays `7000 + 101*trial` (closed loop) and `1000 + 17*trial + k` (frozen).
- `environment_regression.py` — same.

Everything that produces new scientific results uses the scheme above.

## Determinism

Every optimizer draws from the NumPy legacy global generator (`np.random.seed` / `np.random.rand`),
whose stream is guaranteed stable across NumPy versions. This was confirmed empirically in the
canonical-environment regression: closed-loop nRMSE reproduced to 1.1e-13 relative across a NumPy
1.26 → 2.4 change (`ENVIRONMENT_REGRESSION.md`).
