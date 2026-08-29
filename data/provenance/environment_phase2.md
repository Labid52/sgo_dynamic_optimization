# Phase-2 canonical execution environment

All Phase-2 production work runs in the Conda environment **`mpc_sgo_gwo`** and nowhere else. Outputs
generated under any other environment must not be mixed into `results_v2/`.

## Activation

```bash
source /home/labid/miniconda3/etc/profile.d/conda.sh
conda activate mpc_sgo_gwo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Verified versions (checked at the start of Phase 2A)

| Component | Version |
|---|---|
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| pandas | 3.0.3 |
| matplotlib | 3.10.9 |
| OSQP | 1.1.3 |

Interpreter: `/home/labid/miniconda3/envs/mpc_sgo_gwo/bin/python`.
Full listings: `conda_list.txt`, `pip_freeze_phase2.txt`,
`provenance/numpy_config_phase2.txt`.

No packages were installed, upgraded or downgraded to create this environment during Phase 2A; it was
pre-existing and used as found.

## Hardware and OS

- CPU: 12th Gen Intel Core i7-12800H, 14 cores / 20 threads
- RAM: 15 GiB
- OS: Ubuntu 22.04, Linux 6.8.0-124-generic, x86_64
- Full: `provenance/lscpu_phase2.txt`, `provenance/uname_phase2.txt`

## Threading policy

Single-threaded BLAS/OpenMP for every Phase-2 run
(`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`). Problem sizes are small — the largest
Hessian in the primary experiment is 4×4 at N_c=1 — so multithreaded BLAS adds contention rather than
speed, and pinning makes timing reproducible.

**Timing runs additionally require an otherwise idle machine and no concurrent simulations.** This is
enforced procedurally, not by the code; the timing experiment is run alone.

## Relationship to the Phase-1 audit environment

Phase 1 ran under Python 3.10.12 / NumPy 1.26.4 / SciPy 1.8.0. That environment is **superseded** and
must not be used for Phase-2 outputs. The canonical environment was regression-tested against the
frozen submitted artifacts before any production run; see `ENVIRONMENT_REGRESSION.md` (verdict: PASS,
max relative difference 1.1e-13 on closed-loop nRMSE, all five optimizer rankings preserved).

Note that the canonical environment reproduces the submitted deterministic-reference values *better*
than the Phase-1 audit environment did — the archived results were evidently produced under a SciPy
close to 1.17, and SciPy 1.8.0 was the outlier.
