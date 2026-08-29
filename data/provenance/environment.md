# Execution environment (Phase 1)

Recorded 2026-08-15 before any code modification. All Phase 1 experiments were run on this machine.

## Hardware
- CPU: 12th Gen Intel(R) Core(TM) i7-12800H, 14 cores / 20 threads, max 4.80 GHz
- RAM: 15 GiB total
- OS: Ubuntu 22.04 (Linux 6.8.0-124-generic, x86_64)

Full `lscpu` output: `lscpu.txt`.

## Software
| Component | Version |
|---|---|
| Python | 3.10.12 |
| NumPy | 1.26.4 |
| SciPy | 1.8.0 |
| pandas | 2.3.3 |
| matplotlib | 3.5.1 |
| OSQP | 1.1.3 (installed during Phase 1, see note) |

Full package list: `pip_freeze.txt`. NumPy build config: `numpy_config.txt`
(OpenBLAS 0.3.23.dev, `MAX_THREADS=2`, `DYNAMIC_ARCH=1`).

**Known environment caveat (pre-existing, not introduced by this revision):** SciPy 1.8.0 declares a
requirement of NumPy `>=1.17.3,<1.25.0` while NumPy 1.26.4 is installed, so every SciPy import emits a
`UserWarning`. This pairing is what produced the *submitted* results, so it is retained deliberately for
Phase 1 in order to keep the regression test meaningful. Do not "fix" it mid-revision — that would
invalidate the comparison against the submitted numbers. If the pairing is changed later, the full
result set must be regenerated and the regression test rerun.

**OSQP note:** OSQP 1.1.3 was installed during Phase 1 solely as an independent cross-check for the
deterministic reference solutions (reviewer R2.18). It is not used by any simulation path. `pip_freeze.txt`
was captured *after* installation and therefore reflects the environment used for Phase 1 experiments.

## Threading policy
All Phase 1 experiment commands are executed with

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The problem sizes here are small (largest Hessian is 32x32; largest prediction matrix 260x3), so
multithreaded BLAS adds contention rather than speed, and pinning to one thread makes the timing
numbers reproducible. Timing-sensitive measurements are scheduled for Phase 2 step 3c and must run on
an otherwise idle machine.

## Random-seed protocol (Phase 1 onward)
```
seed = SEED_BASE[experiment] + 1_000_000 * instance_index
                             + 1_000     * condition_index
                             +             trial_index
```
`trial_index` is the fastest-varying term so that conditions (start condition, budget level, optimizer)
see identical trial streams and comparisons are paired. `SEED_BASE` values are recorded per script in
the script header and echoed into each output CSV.

Phase 1 additionally reuses the *legacy* seeding of the submitted code wherever it reproduces a
submitted number (regression test only), so that a mismatch is attributable to code changes rather than
to reseeding.
