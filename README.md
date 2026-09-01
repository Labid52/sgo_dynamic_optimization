# SGO Dynamic Optimization

Code, data, protocols, and analysis scripts for the study of the **Squid Game Optimizer (SGO)** under standardized and MPC-generated dynamic optimization problems.

This repository contains:

- implementations of SGO, GWO, PSO, and WOA;
- dynamic MPC benchmark problems;
- Generalized Moving Peaks Benchmark (GMPB) experiments;
- frozen experimental protocols and random-seed rules;
- run-level production data;
- deterministic-reference evidence;
- scripts for regenerating analyses, tables, and figures;
- commands for generating fresh experimental data from the saved protocols.

> **SGO refers to the Squid Game Optimizer** of Azizi et al. (2023), not Social Group Optimization.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── code/
│   ├── reproduce.py
│   ├── verify_claims.py
│   ├── sgo.py
│   ├── gwo.py
│   ├── pso.py
│   ├── woa.py
│   ├── system_defs.py
│   ├── mpc_utils.py
│   ├── sim_common.py
│   ├── reference_solvers.py
│   ├── phase2_protocol.py
│   ├── primary_experiment.py
│   ├── frozen_sweep_v2.py
│   ├── objective_drift*.py
│   ├── budget_sensitivity_v2.py
│   ├── multi_ic_experiment.py
│   ├── dimension_sweep.py
│   ├── perturbation_experiment.py
│   ├── input_error_sensitivity.py
│   ├── sgo_ablation*.py
│   ├── tail_convention_*.py
│   ├── reference_nc_sweep.py
│   ├── strict_fe_verification.py
│   ├── gmpb_*.py
│   ├── gmpb_*.sh
│   └── sgo_benchmark_verification.py
├── data/
│   ├── gmpb/
│   ├── gmpb_temporal_comparators/
│   ├── gmpb_mechanism/
│   ├── protocols/
│   ├── provenance/
│   ├── figures/
│   ├── tables/
│   ├── figures_tables/
│   └── *.csv / *.json
└── docs/
    ├── DATA_INDEX.md
    └── THIRD_PARTY.md
```

Run-level `*_raw.csv` files are the primary numerical evidence. Summary files, statistical outputs, tables, and figures are derived from those records.

---

## Environment setup

The production data were generated with Python 3.11.15 and the package versions pinned in `requirements.txt`.

Create a Conda environment:

```bash
conda create -n sgo_repro python=3.11.15 pip -y
conda activate sgo_repro

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Recorded principal versions:

| Package | Version |
|---|---:|
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| pandas | 3.0.3 |
| matplotlib | 3.10.9 |
| OSQP | 1.1.3 |

Verify the environment:

```bash
python - <<'PY'
import sys, numpy, scipy, pandas, matplotlib, osqp

expected = {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "pandas": "3.0.3",
    "matplotlib": "3.10.9",
    "osqp": "1.1.3",
}

got = {
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "pandas": pandas.__version__,
    "matplotlib": matplotlib.__version__,
    "osqp": osqp.__version__,
}

print("Python:", sys.version.split()[0])
for name, version in expected.items():
    print(f"{'PASS' if got[name] == version else 'MISMATCH'}  "
          f"{name}: {got[name]} (expected {version})")
PY
```

Use single-threaded numerical libraries to match the recorded production configuration:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Detailed environment provenance is stored under `data/provenance/`.

---

## Experimental protocols and random seeds

Experimental design choices were frozen before the final production runs.

The seed protocol is documented in:

```text
data/protocols/SEED_PROTOCOL.md
```

The run-level data also retain the seeds used for individual experiments.

For MPC experiments, seeds are generated deterministically from the experiment, instance, condition, optimizer, and trial indices.

Relevant configuration files include:

```text
code/phase2_protocol.py
data/initial_condition_protocol.json
data/tuned_params_optimizer_only.json
data/protocols/
```

For GMPB, the benchmark seed is the run index. The same benchmark seed generates the same dynamic environment trajectory for all optimizers and temporal conditions. Optimizer seeds are stored separately in the run records.

---

# 1. Reproduce analyses from the supplied raw data

This mode starts from the committed raw run-level data and regenerates derived analyses, statistics, tables, and figures.

```bash
python code/reproduce.py --from-data
```

Expected final line:

```text
Level-1 analysis reproduction: 9/9 stages succeeded
```

The principal numerical claims can also be recomputed independently:

```bash
python code/verify_claims.py
```

Expected final line:

```text
147/147 checks passed
```

The claim checker reads the raw data directly and verifies quantities including:

- GMPB run counts, budgets, rankings, and pairwise comparisons;
- temporal-transfer effects;
- SGO grouping and interaction results;
- MPC drift and convergence results;
- deterministic-reference certification;
- budget, dimension, warm-start, tail, perturbation, and multi-initial-condition results;
- original-SGO benchmark reproduction.

`verify_claims.py` does not parse a manuscript `.tex` file.

### Optional isolated analysis run

To avoid rewriting derived files in the working tree, analysis can be run from a temporary copy:

```bash
PKG="$(pwd)"

rm -rf /tmp/sgo_analysis_check
cp -a "$PKG" /tmp/sgo_analysis_check
cd /tmp/sgo_analysis_check

python code/reproduce.py --from-data
python code/verify_claims.py
```

To compare regenerated machine-readable outputs with the repository copy:

```bash
diff -rq data "$PKG/data" \
  | grep -vE '\.(eps|pdf)( |$)' \
  | grep -v '_meta.json'
```

Expected: no output.

PDF/EPS binaries may differ because of serialization metadata. Runtime-only metadata such as `wall_seconds` may also differ.

---

# 2. Generate fresh MPC experimental data

The following validated test executes all four optimizers again using the stored protocol and deterministic seed rules.

Fresh results are written only under `scratch/`.

```bash
mkdir -p scratch/audit

python - <<'EOF' 2>&1 | grep -v "Polish"
import sys, numpy as np, pandas as pd
sys.path.insert(0, "code")

from system_defs import get_systems
from sim_common import simulate, build_problem, make_cost, solve_one_step
import phase2_protocol as P2
from reference_solvers import osqp_reference, problem_linear_term

ics = P2.load_initial_conditions()
systems = get_systems()
tuned = P2.load_tuned_np()["selected"]

key = "flight"
sysdef = systems[key]
mpc = P2.fixed_mpc_params(sysdef)

problem = build_problem(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"])
lb, ub = problem.lb_seq, problem.ub_seq

det = simulate(
    key,
    "QP",
    params=mpc,
    seed=11,
    verbose=False,
    x0_override=ics[key]["eval"],
)

k = 6
x = det["x_hist"][:, k]

f = problem_linear_term(problem, sysdef, x, k)
u, _, _ = osqp_reference(problem, f)
cost = make_cost(problem, x, k)
Jstar = float(cost(np.clip(u, lb, ub)))

production = pd.read_csv(
    "data/frozen_sweep_v2_raw.csv",
    float_precision="round_trip",
)

stored = float(
    production[
        (production.system == key) &
        (production.k == k)
    ].J_star.iloc[0]
)

print(
    f"fresh J*={Jstar:.12g}  stored={stored:.12g}  "
    f"bit-identical={Jstar == stored}"
)

rows = []

for alg in ["SGO", "GWO", "PSO", "WOA"]:
    NP = int(tuned[key][alg]["NP"])
    params = dict(mpc)
    params.update(P2.strict_fe_params(NP, alg=alg))

    for trial in range(20):
        seed = P2.seed_for(
            "frozen_v2",
            P2.SYSTEM_INDEX[key],
            0,
            P2.ALG_INDEX[alg],
            trial,
        )

        np.random.seed(seed)

        _, Jf, _, _, _, nfe = solve_one_step(
            problem,
            x,
            k,
            alg,
            params,
            np.zeros_like(lb),
        )

        rows.append({
            "alg": alg,
            "trial": trial,
            "nfe": int(nfe),
            "gap": abs(float(Jf) - Jstar) / max(1, abs(Jstar)),
        })

fresh = pd.DataFrame(rows)
fresh.to_csv("scratch/audit/fresh_mpc_smoke.csv", index=False)

prod_medians = (
    production[
        (production.system == key) &
        (production.k == k) &
        (production.start == "cold")
    ]
    .groupby("alg")
    .final_gap_norm
    .median()
)

comparison = (
    fresh.groupby("alg").gap.median()
    .to_frame("fresh")
    .join(prod_medians.to_frame("production"))
)

comparison["PASS"] = np.isclose(
    comparison.fresh,
    comparison.production,
    rtol=0,
    atol=0,
)

print(comparison.to_string())

print(
    "realized FE:",
    sorted(set(fresh.nfe)),
    "-> all exactly 1000:",
    sorted(set(fresh.nfe)) == [1000],
)
EOF
```

Expected:

```text
bit-identical=True

        fresh  production  PASS
GWO     ...       ...      True
PSO     ...       ...      True
SGO     ...       ...      True
WOA     ...       ...      True

realized FE: [1000] -> all exactly 1000: True
```

This checks the complete path:

```text
code + saved protocol + saved seeds
        ->
fresh optimizer execution
        ->
new run-level result
        ->
comparison with production data
```

---

# 3. Deterministic MPC reference evidence

The canonical deterministic-reference workflow is the Phase-2 OSQP-based `reference_nc_sweep.py` analysis.

Production evidence:

```text
data/reference_nc_sweep_raw.csv
data/reference_nc_sweep_summary.csv
data/reference_certificates.csv
```

The accepted production sweep contains 25 `(system, Nc)` configurations:

- 22 accepted;
- all five primary `Nc=1` configurations accepted;
- only UAV `Nc = 3, 5, 8` rejected from analyses requiring an optimality gap.

Inspect the stored certification:

```bash
python - <<'PY'
import pandas as pd

s = pd.read_csv(
    "data/reference_nc_sweep_summary.csv",
    float_precision="round_trip",
)

print("cells tested:", len(s))
print("accepted:", int(s.certified.sum()))

primary = s[s.Nc == 1]
print("all five primary Nc=1 accepted:", bool(primary.certified.all()))

rejected = sorted(zip(
    s[~s.certified].system,
    s[~s.certified].Nc,
))
print("rejected:", rejected)

print("\nPrimary Nc=1 reference quality:")
print(
    primary[
        [
            "system",
            "max_solver_rel_disagreement",
            "max_subopt_bound_norm",
            "cond_H",
        ]
    ].to_string(index=False)
)
PY
```

Expected:

```text
cells tested: 25
accepted: 22
all five primary Nc=1 accepted: True
rejected: [('uav', 3), ('uav', 5), ('uav', 8)]
```

`reference_verification.py` is retained as an earlier Phase-1 diagnostic and is not the canonical manuscript certification.

---

# 4. Generate fresh GMPB experimental data

GNU Octave is required to regenerate GMPB benchmark states from the official benchmark implementation.

## 4.1 Create an Octave Conda environment

```bash
conda create -n octave_gmpb -c conda-forge octave=10.3.0 -y
conda activate octave_gmpb

export OCTAVE_HOME="$CONDA_PREFIX"
export PATH="$OCTAVE_HOME/bin:$PATH"

which octave
octave --no-gui -q --eval "disp(fullfile('a','b')); disp(version)"
```

Expected:

```text
.../envs/octave_gmpb/bin/octave
a/b
10.3.0
```

With Conda Octave, `OCTAVE_HOME` should be set so that the standard Octave function path is resolved correctly.

## 4.2 Generate one fresh dynamic environment

From the repository root:

```bash
ROOT="$(pwd)"

mkdir -p \
  scratch/gmpb_states \
  scratch/gmpb_out \
  scratch/audit
```

Generate case `F2`, benchmark seed `1`:

```bash
cd "$ROOT/data/gmpb/octave"

octave --no-gui -q gmpb_dump.m \
  F2 \
  1 \
  "$ROOT/scratch/gmpb_states"
```

Expected:

```text
DUMP OK F2 seed 1  d=5 m=10 T=100 CF=5000 MaxEvals=500000 probes=2400
```

Run the official Octave reference evaluator:

```bash
octave --no-gui -q gmpb_refrun.m \
  F2 \
  1 \
  20000 \
  500 \
  "$ROOT/scratch/gmpb_states"
```

Expected:

```text
REFRUN OK F2 seed 1 batch 500  FE=20000 env=4
Eo=23.7523181571321 Ebbc=15.9498585853071
```

Return to the repository root:

```bash
cd "$ROOT"
```

## 4.3 Validate the Python GMPB implementation

Activate the Python environment again:

```bash
conda activate sgo_repro

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Python versus official Octave benchmark:

```bash
python code/gmpb_equivalence.py \
  scratch/gmpb_states \
  scratch/audit
```

Expected:

```text
9 checks, 0 failures
```

SGO dynamic-driver fidelity:

```bash
python code/gmpb_sgo_fidelity.py scratch/audit
```

Expected:

```text
12 checks, 0 failures
```

## 4.4 Run all four optimizers on the freshly generated GMPB state

Use a new output directory:

```bash
rm -rf scratch/gmpb_out_user
mkdir -p scratch/gmpb_out_user

python code/gmpb_run.py \
  primary \
  scratch/gmpb_states \
  scratch/gmpb_out_user \
  F2 \
  1 \
  4
```

A fresh execution should begin with:

```text
primary: 4 jobs, 0 already done, 4 to run
```

and finish with:

```text
4/4 runs
"completed_now": 4
```

## 4.5 Compare fresh GMPB results with production

```bash
python - <<'PY'
import pandas as pd

RT = dict(float_precision="round_trip")

fresh = pd.read_csv(
    "scratch/gmpb_out_user/gmpb_primary_raw.csv",
    **RT,
).set_index("alg")

production = pd.read_csv(
    "data/gmpb/gmpb_primary_raw.csv",
    **RT,
)

production = production[
    (production.case == "F2") &
    (production.seed == 1)
].set_index("alg")

columns = [
    "realized_FE",
    "environments_completed",
    "offline_error",
    "best_error_before_change",
    "mean_population_diversity",
]

all_pass = True

for alg in ["SGO", "GWO", "PSO", "WOA"]:
    print(f"\n{alg}")

    for col in columns:
        a = fresh.loc[alg, col]
        b = production.loc[alg, col]
        ok = a == b
        all_pass &= ok

        print(
            f"  {'PASS - identical' if ok else 'FAIL - difference found'}"
            f"  {col}: fresh={a}  production={b}"
        )

print("\nOVERALL:", "PASS" if all_pass else "FAIL")
PY
```

On the recorded environment, the validated F2/seed-1 test reproduces the stored production values exactly.

Expected:

```text
OVERALL: PASS
```

---

# 5. Full experimental rerun

The tests above regenerate selected MPC and GMPB production cells. The full experimental study can also be rerun from the frozen protocols.

List the ordered MPC production entry points:

```bash
python code/reproduce.py --list-full-rerun
```

The complete MPC workflow includes:

- protocol and initial-condition preparation;
- deterministic-reference certification;
- exact FE-budget verification;
- held-out optimizer tuning;
- primary closed-loop experiments;
- objective-drift calculation;
- frozen-subproblem experiments;
- budget sensitivity;
- multi-initial-condition experiments;
- control-horizon/dimension sweep;
- perturbation experiments;
- input-error sensitivity;
- prediction-tail sensitivity;
- SGO ablation.

Full regeneration is computationally expensive. New experiment output should be directed to a separate location rather than the committed production-data tree.

Where supported by the experiment scripts:

```bash
mkdir -p scratch/full_mpc
export SGO_OUTPUT_DIR="$PWD/scratch/full_mpc"
```

Then execute the commands printed by:

```bash
python code/reproduce.py --list-full-rerun
```

in the displayed order.

---

## Full GMPB state generation

Generate the complete set of benchmark states for seeds 1–31:

```bash
mkdir -p scratch/full_gmpb_states

bash code/gmpb_generate.sh \
  scratch/full_gmpb_states \
  1 \
  31 \
  8
```

The full primary GMPB study can then be written to a separate directory:

```bash
mkdir -p scratch/full_gmpb_primary

python code/gmpb_run.py \
  primary \
  scratch/full_gmpb_states \
  scratch/full_gmpb_primary \
  ALL \
  31 \
  16
```

Additional temporal-comparator and mechanism experiments are available through:

```text
code/gmpb_temporal_comparators.py
code/gmpb_mechanism_run.py
```

The complete GMPB study involves hundreds of millions of objective evaluations and may require substantial computation time.

---

# 6. Function-evaluation budget policy

Budgets are enforced at the objective-function call level.

The primary MPC protocol uses exactly 1000 objective evaluations per MPC step unless a budget-sensitivity experiment explicitly changes that value.

Retained evidence:

```text
data/strict_fe_budget_verification.csv
data/strict_fe_budget_verification.json
```

The validated fresh MPC test should report:

```text
realized FE: [1000] -> all exactly 1000: True
```

For GMPB, run-level records retain:

```text
realized_FE
environments_completed
```

The primary GMPB study uses the benchmark-defined budget and 100 environments per run.

---

# 7. Original-SGO benchmark reproduction

The repository also contains the static original-SGO implementation verification.

```bash
python code/sgo_benchmark_verification.py
```

Retained evidence:

```text
data/sgo_benchmark_raw.csv
data/sgo_benchmark_summary.csv
data/sgo_benchmark_comparison.txt
data/sgo_benchmark_verification.json
```

The as-implemented convention agrees with the published SGO mean on 11 of the 13 tested functions. Alpine 1 and Quintic are retained and reported as exceptions.

---

# 8. Raw and derived data

`docs/DATA_INDEX.md` provides a file-level index.

General convention:

- `*_raw.csv`: independent run/trial/transition records;
- `*_summary.csv`: derived summary statistics;
- `*_statistics.csv`: statistical comparisons;
- `*_association.csv` and `*_effects.csv`: derived analyses;
- `figures/` and `tables/`: generated presentation artifacts.

The run-level data are retained even when derived summaries are also supplied.

---

# 9. Output isolation

Fresh experimental output should be stored under `scratch/` or another user-selected output directory.

After running local checks:

```bash
git status --short
```

A clean source/data tree may show only untracked test output such as:

```text
?? scratch/
```

Tracked production data and source files should remain unchanged.

---

# 10. Third-party benchmark components

`docs/THIRD_PARTY.md` documents the GMPB source components, provenance, and hashes.

The official GNU Octave GMPB implementation is used to generate the dynamic benchmark state. The Python evaluator is checked against the official Octave implementation before dynamic optimizer runs.

---

# 11. Numerical reproducibility across systems

The recorded environment reproduced the validated MPC and GMPB test cases exactly.

Small floating-point differences may occur with materially different:

- CPUs;
- operating systems;
- BLAS implementations;
- SciPy/NumPy versions;
- OSQP builds.

For the closest reproduction, use the pinned environment and the documented seed and protocol files.

Timing results are machine-dependent and should not be interpreted as embedded or hardware-in-the-loop real-time certification.

---

# 12. Citation

Please cite the accompanying manuscript and the original algorithm/benchmark sources when using this repository.

Primary SGO reference:

M. Azizi, M. Baghalzadeh Shishehgarkhaneh, M. Basiri, and R. C. Moehler,
“Squid Game Optimizer (SGO): a novel metaheuristic algorithm,”
*Scientific Reports*, vol. 13, Art. no. 5373, 2023.

Additional GMPB, GWO, PSO, and WOA references are listed in `docs/THIRD_PARTY.md`.

---

## Reproduction summary

Two complementary paths are provided:

### Recompute results from supplied data

```bash
python code/reproduce.py --from-data
python code/verify_claims.py
```

This checks:

```text
raw data -> analyses/statistics -> reported numerical results
```

### Generate fresh experimental data

Use the MPC procedure in Section 2 and GMPB procedure in Section 4.

This checks:

```text
code + protocols + seeds -> fresh run-level experimental data
```

The full experiment-generation entry points are listed by:

```bash
python code/reproduce.py --list-full-rerun
```
