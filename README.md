# Dynamic optimization behavior of the Squid Game Optimizer — code and data

This repository contains the code and the raw repeated-run data supporting a
controlled study of the **Squid Game Optimizer (SGO)** under standardized and
MPC-generated dynamic optimization problems. It evaluates the *original* SGO of
Azizi et al. (2023) — it does not propose a modified algorithm and makes no claim
that SGO is superior. GWO, PSO and WOA are used as matched controls so that a
behavior can be tested for being SGO-specific rather than generic to
population-based search. The repository accompanies the manuscript but is designed
to stand alone: everything needed to regenerate the reported tables, statistics and
figures is here, and no manuscript file is required.

> **SGO here means the Squid Game Optimizer** (Azizi, Baghalzadeh Shishehgarkhaneh,
> Basiri & Moehler, *Scientific Reports* 13:5373, 2023; doi:10.1038/s41598-023-32465-z),
> **not** Social Group Optimization, which shares the acronym.

## The question this repository supports

SGO was introduced and validated mainly on static problems. This study asks how it
behaves when the objective changes during operation, and — more importantly — which of
its apparent dynamic weaknesses survive matched controls. Two settings are used:

1. **GMPB** — the 12 official Generalized Moving Peaks Benchmark competition instances,
   which give standardized, controlled environmental change.
2. **MPC-generated objective sequences** — five receding-horizon control problems, in
   which each sampling instant poses a new but related finite-horizon problem and the
   optimizer's own decision helps define the next one.

The two settings are complementary, not equivalent: GMPB directly manipulates how much
optimizer state survives a change, while the MPC warm-start analysis changes the source
and quality of the carried candidate sequence.

## Repository structure

```
.
├── README.md                    this file
├── requirements.txt             pinned versions of the environment that produced data/
├── .gitignore
├── code/                        all analysis and experiment code (flat, prefix-grouped)
│   ├── reproduce.py             single entry point (Level 1 / Level 2 / verification)
│   ├── verify_claims.py         independent recomputation of the headline numbers
│   ├── sgo.py gwo.py pso.py woa.py            optimizer implementations
│   ├── sgo_ablation_impl.py                   ablation-capable copy of SGO
│   ├── system_defs.py mpc_utils.py sim_common.py   MPC plants, cost construction, engine
│   ├── reference_solvers.py tail_prediction.py     deterministic reference, tail convention
│   ├── phase2_protocol.py initial_condition_protocol.py   frozen protocol and initial conditions
│   ├── primary_experiment.py  … frozen_sweep_v2.py  …     MPC experiments
│   ├── gmpb_*.py gmpb_*.sh                    GMPB benchmark, runners, gates, analysis
│   ├── sgo_benchmark_verification.py summarize_sgo_benchmark.py  original-SGO reproduction
│   └── phase2a_analysis.py phase3_artifact_prep.py phase3b_manuscript_artifacts.py  figures/tables
├── data/
│   ├── gmpb/                    primary GMPB study (raw runs, environments, statistics, Octave generator)
│   ├── gmpb_temporal_comparators/  matched temporal-transfer study, 4 optimizers
│   ├── gmpb_mechanism/          SGO grouping diagnostic and the 2x2 temporal interaction
│   ├── mpc_submitted_protocol/  earlier jointly tuned protocol (co-design comparison only)
│   ├── protocols/               frozen seed, drift-strata and tail-convention protocols
│   ├── provenance/              hardware, OS, interpreter and package records
│   ├── cache/                   regenerable optimizer warm-start histories
│   ├── figures/ tables/ figures_tables/   generated outputs
│   └── *.csv *.json             MPC raw and processed data (see docs/DATA_INDEX.md)
└── docs/
    ├── DATA_INDEX.md            every data file, raw vs processed, and what it contains
    └── THIRD_PARTY.md           third-party components, hashes and attribution
```

**A note on paths inside the frozen protocol files.** The `*_protocol.json` and
`*_PROTOCOL.md` records under `data/` are the pre-registration documents as they were
written before each study ran, and they are reproduced here unaltered. They refer to the
working directory `results_v2/`, which is this package's `data/`, and to `codes/`, which
is `code/`. Nothing else changed: the exported scripts differ from the working copies
only in those directory names (plus two shell launchers made portable). No optimization
mathematics or numerical procedure was modified when the package was assembled.

`code/` is intentionally flat. The modules import each other by plain module name
(`import system_defs`), so they must sit in one directory; the filename prefixes
(`gmpb_*`, `sim_*`/`system_*`, `sgo_*`) carry the grouping instead. Every file is listed
by role in the table further below.

## Installation

```bash
git clone <this repository>
cd SGO_Dynamic_Optimization_Reproducibility
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python **3.11** is expected (3.11.15 was used). The pinned `matplotlib` and `pandas`
versions matter: older combinations fail when plotting pandas Series directly.

GNU Octave is required **only** for a full GMPB rerun (Level 2), to regenerate the
benchmark state files. It is not needed for anything else.

## Environment that produced the data in `data/`

All results under `data/` were produced in one environment, recorded at the time in
`data/provenance/environment_phase2.md` and reproduced here verbatim:

| Component | Version |
|---|---|
| OS | Ubuntu 22.04, Linux 6.8.0-124-generic, x86_64 |
| CPU | 12th Gen Intel Core i7-12800H, 14 cores / 20 threads |
| RAM | 15 GiB |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| pandas | 3.0.3 |
| matplotlib | 3.10.9 |
| OSQP | 1.1.3 |

Full listings: `data/provenance/conda_list.txt`, `pip_freeze_phase2.txt`,
`numpy_config_phase2.txt`, `lscpu_phase2.txt`, `uname_phase2.txt`.

**Threading.** Every production run was executed single-threaded:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The problems are small (the largest Hessian in the primary MPC experiment is 4x4), so
multithreaded BLAS adds contention rather than speed, and pinning to one thread makes
timings reproducible. Use the same setting to reproduce timing numbers.

**Timing results are machine- and environment-dependent.** They come from an
unoptimized single-process Python implementation on one machine and support relative
comparison between the methods *as implemented here*. They are not a hardware-in-the-loop
result, not an embedded or real-time certification, and not a deployability claim.

An earlier audit environment (Python 3.10.12 / NumPy 1.26.4 / SciPy 1.8.0) is recorded in
`data/provenance/environment.md` for provenance only. It is superseded and must not be
used to regenerate anything.

## Two levels of reproduction

A full rerun of every optimizer experiment costs on the order of hundreds of millions of
objective evaluations. You do not need it to check the reported numbers.

### Level 1 — analysis reproduction (recommended; minutes)

Regenerates every summary table, reported statistic and figure from the committed raw
per-run data. Runs no optimizer, no MPC simulation and no benchmark environment, and
needs no GMPB state files.

```bash
python code/reproduce.py --from-data
```

Nine analysis stages run in dependency order. Outputs are written to
`data/figures_tables/{figures,tables}`, `data/{figures,tables}` and
`data/gmpb*/{figures,tables}`, overwriting the committed copies with identical content.

To check the headline numbers independently of the table generators:

```bash
python code/verify_claims.py     # or: python code/reproduce.py --verify
```

This recomputes 147 reported quantities directly from the raw CSVs and prints PASS/FAIL
for each — run counts, budget exactness, the GMPB rankings and pairwise outcomes, the
matched temporal-transfer counts, the grouping-diagnostic magnitudes and directions, the
MPC drift, convergence, budget, dimension, warm-start, tail and perturbation results, and
the original-SGO reproduction.

### Level 2 — full experiment rerun (expensive)

```bash
python code/reproduce.py --list-full-rerun
```

prints the ordered entry points. In outline:

**MPC** (hours, single machine): freeze the protocol and initial conditions, certify the
deterministic reference, verify the evaluation budget, tune optimizer-only parameters on
held-out initial conditions, then run the closed-loop primary experiment, the drift
measurement, the frozen sweep, and the budget / multi-IC / dimension / perturbation /
input-error / tail / ablation studies.

**GMPB** (days): first regenerate the benchmark states with Octave, then run the studies.

```bash
# 0. Generate the official GMPB generator states (Octave; 12 cases x 31 seeds).
#    Octave must be on PATH, or point OCTAVE_ACTIVATE at a script that puts it there.
bash code/gmpb_generate.sh ./gmpb_states 1 31 8

# 1. Equivalence gates first -- nothing downstream is valid unless these pass.
#    The Octave reference dumps go in their own directory:
#      cd data/gmpb/octave && octave --no-gui -q gmpb_refrun.m F2 1 20000 500 <refdir>
python code/gmpb_equivalence.py <refdir> data/gmpb          # Python port vs official Octave
python code/gmpb_sgo_fidelity.py data/gmpb                  # dynamic driver vs static SGO

# 2. Primary study (12 cases x 4 optimizers x 31 runs) and the SGO temporal modes.
bash code/gmpb_launch.sh ./gmpb_states data/gmpb 16
#   equivalently:
#     python code/gmpb_run.py primary  ./gmpb_states data/gmpb ALL 31 16
#     python code/gmpb_run.py temporal ./gmpb_states data/gmpb ALL 31 16
python code/gmpb_gates.py ./gmpb_states data/gmpb           # validation gates A-J

# 3. Matched comparator temporal transfer, then its validation gate.
#      <cache> <outdir> <cases> <seeds> <algs> <modes> [nproc]
python code/gmpb_temporal_comparators.py \
    ./gmpb_states data/gmpb_temporal_comparators ALL 31 ALL ALL 16
python code/gmpb_temporal_validation.py ./gmpb_states data/gmpb_temporal_comparators

# 4. Grouping diagnostic, after its equivalence gate.
#      <cache> <outdir> <cases> <seeds> <grouping> <mode> [nproc]
python code/gmpb_mechanism_equivalence.py ./gmpb_states data/gmpb_mechanism
python code/gmpb_mechanism_run.py ./gmpb_states data/gmpb_mechanism \
    ALL 31 random  population_persistence 16      # baseline reproduction
python code/gmpb_mechanism_run.py ./gmpb_states data/gmpb_mechanism \
    ALL 31 fitness population_persistence 16      # primary grouping diagnostic
python code/gmpb_mechanism_run.py ./gmpb_states data/gmpb_mechanism \
    F2,F8,F10,F12 31 fitness previous_best_seeded 16   # 2x2 temporal interaction
```

All the run scripts write incrementally and skip cells that already exist, so an
interrupted study can simply be relaunched with the same arguments.

**Original-SGO static reproduction** (hours):

```bash
python code/sgo_benchmark_verification.py
```

The GMPB state files are **not** committed: they are large binary generator states, and
they are regenerated deterministically from the seed by the unmodified official Octave
code. Level 1 does not need them.

## How to reproduce each reported result

| Result | Data | Regenerate with |
|---|---|---|
| GMPB primary table and figure, pairwise statistics | `data/gmpb/gmpb_primary_raw.csv` | `python code/gmpb_analysis.py` |
| Matched temporal-transfer comparison and behavioural diagnostics | `data/gmpb_temporal_comparators/gmpb_temporal_comparator_raw.csv` | `python code/gmpb_temporal_analysis.py` |
| SGO grouping diagnostic, diversity, 2x2 temporal interaction | `data/gmpb_mechanism/gmpb_*_raw.csv` | `python code/gmpb_mechanism_analysis.py` |
| MPC primary tables, budget accounting, timing, co-design comparison | `data/primary_closed_loop_raw.csv`, `data/mpc_submitted_protocol/` | `python code/phase2a_analysis.py` |
| MPC effect sizes and practical-significance labels | `data/primary_closed_loop_raw.csv` | `python code/effect_sizes_primary.py` |
| Drift strata, frozen-sweep difficulty, warm-start mechanism | `data/objective_drift_full_raw.csv`, `data/frozen_sweep_v2_raw.csv` | `python code/drift_performance_analysis.py` |
| SGO ablation and warm-start tables | `data/sgo_ablation_raw.csv` | `python code/phase3_artifact_prep.py` |
| Original-SGO 13-function reproduction | `data/sgo_benchmark_raw.csv` | `python code/summarize_sgo_benchmark.py` |
| All main figures (fig1–fig7) and tables | the files above | `python code/phase3b_manuscript_artifacts.py` |

### Original SGO implementation verification (Reviewer 2)

This is deliberately easy to find. The evidence that the SGO implementation reproduces
the behavior of the original paper lives in four places:

- `code/sgo.py` — the implementation used for every result in this study.
- `code/sgo_benchmark_verification.py` — reruns the original paper's own protocol:
  13 benchmark functions, 100 dimensions, 150 000 evaluations, 100 independent runs,
  population 50, under both readings of the ambiguous winning-condition branch.
- `data/sgo_benchmark_raw.csv` — all 2600 individual runs (13 functions x 2 conventions
  x 100 runs), not only the means.
- `data/sgo_benchmark_summary.csv`, `data/sgo_benchmark_comparison.txt`,
  `data/sgo_benchmark_verification.json` — per-function comparison against the published
  means and the pass/fail verdicts.

Under the as-implemented convention the reproduction agrees with the published mean on
**11 of the 13 functions**. Two fall outside the agreement band: Quintic, where this
implementation reaches a mean of 306.5 against a published 23.6, and Alpine 1, where this
implementation lands closer to the optimum than the published value. Both are reported
rather than adjusted.

For the dynamic setting, two further gates check that the same implementation is what
actually ran: `data/gmpb/sgo_fidelity.csv` (dynamic driver vs static implementation,
evaluation by evaluation) and `data/gmpb_mechanism/baseline_equivalence.csv` (308 checks
that the ablation driver with random grouping is bit-identical to the validated SGO).

## Raw vs processed data

`docs/DATA_INDEX.md` labels every file. In short: `*_raw.csv` files hold one row per
independent run, trial or transition and are the primary evidence; `*_summary.csv`,
`*_statistics.csv`, `*_association.csv`, `*_effects.csv` and everything in `tables/` and
`figures/` are recomputed from them by Level 1. Per-run data is retained everywhere,
including where a summary also exists — summaries are a convenience, not a replacement.

Running Level 1 rewrites every processed file. On the recorded environment this
reproduces all committed processed CSVs, JSONs and LaTeX tables byte-for-byte, except two
`wall_seconds` fields that record how long the analysis itself took.

## Seeds and randomness

The seed protocol is `data/protocols/SEED_PROTOCOL.md`. Every experiment records the
seeds it used in its own output CSV.

- **MPC.** `seed = SEED_BASE[experiment] + 1_000_000*instance_index + 1_000*condition_index
  + trial_index`. `trial_index` varies fastest, so different conditions (start condition,
  budget, optimizer) see identical trial streams and comparisons are paired.
- **GMPB.** The benchmark seed is the run index, so run *r* presents the same environment
  trajectory to every optimizer and every temporal mode. 31 independent runs per case and
  optimizer; comparisons are paired by benchmark seed. Optimizer seeds are recorded
  separately in the run records.

## Function-evaluation budget policy

Budgets are enforced at the **objective-call** level, not through iteration limits, because
one SGO iteration can evaluate the objective a variable number of times. Early stopping and
stagnation termination are disabled.

- **MPC.** Exactly `B = 1000` objective calls per MPC step unless a budget curve is shown
  explicitly. Enforcement was verified over 2000 cells spanning four budgets, five
  population sizes, five systems and all four optimizers; realized equalled requested in
  every cell (`data/strict_fe_budget_verification.*`). Every production run also records
  its own realized count.
- **GMPB.** Each optimizer receives exactly the benchmark-defined change frequency per
  environment and exactly 100 environments per run, giving the official per-case totals
  (50 000 to 500 000 evaluations). Initialization and post-change re-evaluations consume
  budget. All three temporal modes have identical budgets. Every run's `budget_exact`
  flag is `True` in the committed data.

## Optimizer configuration

- Population sizes, algorithm constants and the frozen MPC protocol: `code/phase2_protocol.py`.
- Optimizer-only tuned population sizes (regime A): `data/tuned_params_optimizer_only.json`,
  chosen on tuning initial conditions that are disjoint from the evaluation ones
  (`data/initial_condition_protocol.json`).
- The earlier jointly tuned configuration, kept only for the co-design comparison:
  `code/tuned_params.json` and `data/mpc_submitted_protocol/*_qp_tuned_configs.txt`.
- GMPB: population 20 for every optimizer on every case; PSO inertia 0.9→0.4 with
  `c1 = c2 = 2`; WOA `b = 1`; GWO and SGO have no parameters beyond population size.
  See `data/gmpb/gmpb_protocol.json`.

## Deterministic reference solver

The primary deterministic reference for the MPC subproblems is **OSQP**, cross-checked
against an independent multistart **L-BFGS-B** solve from the start point and 20 random
interior points (`code/reference_solvers.py`). A cell is accepted when the strong-convexity
suboptimality bound at the returned point is far below the 1e-4 convergence criterion *and*
the two solvers agree to better than 1e-4 relative.

22 of the 25 tested (system, `N_c`) cells are accepted. All five primary `N_c = 1` cells
pass. The three UAV cells at `N_c` in {3, 5, 8} fail the agreement rule and are excluded
from every optimality-gap computation. Per-instant certificates:
`data/reference_certificates.csv`, `data/reference_nc_sweep_summary.csv`.

## Expected outputs

| Output | Written to |
|---|---|
| Main-paper figures `fig1_objective_drift` … `fig7_tail_convention` (`.pdf` and `.eps`) | `data/figures_tables/figures/` |
| Main-paper and supplementary LaTeX tables (`tab2` … `tab8`, `tabS_*`) | `data/figures_tables/tables/` |
| GMPB primary and temporal-mode figures and tables | `data/gmpb/{figures,tables}/` |
| Matched temporal-transfer figures | `data/gmpb_temporal_comparators/figures/` |
| Grouping-diagnostic figures and table | `data/gmpb_mechanism/{figures,tables}/` |
| MPC diagnostic figures and per-experiment LaTeX tables | `data/figures/`, `data/tables/` |
| Recomputed summary and statistics CSVs | alongside the raw files they derive from |

## Data provenance

Every experiment writes, next to its results: run-level records, its machine-readable
protocol (`*_protocol.json`), run metadata (`*_meta.json`), the seeds it used and the
realized evaluation counts. Design decisions that had to be fixed before outcomes were
seen — the MPC protocol, the held-out initial conditions, the drift strata and frozen
instance list, the four diagnostic GMPB cases, the reference-acceptance criteria — are
committed as frozen protocol files under `data/protocols/`, `data/*_protocol.json` and
`data/gmpb*/`*`_protocol.json`.

The GMPB environments are not re-created: the official Octave generator produces each
complete environment sequence and the Python evaluator loads that state and reproduces the
released fitness and evaluation bookkeeping. The pinned upstream file hashes are in
`data/gmpb/gmpb_protocol.json` and `docs/THIRD_PARTY.md`.

## Third-party components

`docs/THIRD_PARTY.md` lists what is redistributed and what is only cited. In short: the
four official GMPB `.m` files under `data/gmpb/octave/official/` come from
[EDOLAB-MATLAB](https://github.com/EDOLAB-platform/EDOLAB-MATLAB) (author Danial Yazdani)
and are included unmodified with pinned SHA-256 hashes. **That upstream repository does
not carry an explicit licence file**, so if you plan to redistribute this package further,
fetch those four files from upstream rather than relying on the copies here; nothing else
depends on them. Everything else under `code/` — including the SGO, GWO, PSO and WOA
implementations — is authored in this project from the published algorithm descriptions.

## Known scope limitations relevant to reproduction

- The GMPB study uses the 12 released competition instances in the simplified
  single-sub-function configuration, with a common population size of 20 rather than
  per-case tuning.
- The matched temporal comparison covers four preselected diagnostic cases
  (F2, F8, F10, F12), not all twelve.
- The fitness-grouping change is applied to SGO only. It is a diagnostic ablation, not a
  proposed algorithm, and it is not compared with GWO, PSO or WOA as a method. Operator
  branch-activation frequencies were not instrumented, so the data do not identify the
  operator-level causal path.
- The accepted MPC subproblems are bound-constrained convex quadratic programs. Nonconvex
  MPC, moving hard state constraints and broader plant uncertainty are outside scope.
  Three UAV cells at larger control horizons are excluded because the two independent
  deterministic solvers do not meet the agreement tolerance.
- Most MPC analyses use the zero-tail prediction convention; hold-last is tested
  deterministically on all five systems and stochastically on two.
- Timing figures are environment-dependent and are not a real-time or embedded claim.
- GMPB benchmark state files are not shipped and must be regenerated with GNU Octave for a
  Level-2 GMPB rerun.

## AI-assisted development disclosure

Anthropic Claude Code was used as an AI-assisted **software-development tool** during this
project. Its role, as reflected in the repository history, covered planning and structuring
of the experimental code, creation of code scaffolding and backbone, implementation
assistance, debugging and troubleshooting, refactoring, and the preparation of analysis,
table and figure-generation scripts. ChatGPT and Claude were additionally used during
manuscript preparation for idea generation, language editing, organization and LaTeX
formatting, as disclosed in the manuscript itself.

AI-generated output was **not** treated as scientific evidence and no AI tool is cited as a
source for any optimization, benchmark or numerical claim. The experimental design, the
choice and freezing of protocols, the execution of the experiments, the numerical results,
their verification and interpretation, and the final conclusions were reviewed and are the
responsibility of the authors.

Verification that is actually implemented in this repository, and that a reader can rerun,
consists of: the equivalence and fidelity gates listed above, the exact function-evaluation
budget checks, the deterministic-reference certificates, and `code/verify_claims.py`, which
recomputes the reported numbers from the raw data. No claim of independent third-party code
review is made.

## Citation

If you use this code or data, please cite the manuscript this repository accompanies, and
cite the original algorithm and benchmark sources separately:

- **SGO** — M. Azizi, M. Baghalzadeh Shishehgarkhaneh, M. Basiri, R. C. Moehler, "Squid
  Game Optimizer (SGO): a novel metaheuristic algorithm," *Scientific Reports* 13:5373 (2023).
- **GMPB** — D. Yazdani et al., "Benchmarking Continuous Dynamic Optimization: Survey and
  Generalized Test Suite," *IEEE Transactions on Cybernetics* 52(5), 3380–3393; and
  D. Yazdani et al., arXiv:2106.06174.
- **GWO / PSO / WOA** — see `docs/THIRD_PARTY.md`.
