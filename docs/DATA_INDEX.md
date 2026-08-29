# Data index

Every file under `data/` is listed here with its role. **Raw** means one row per
independent run, trial or transition, written directly by an experiment script.
**Processed** means it is recomputed from raw files by an analysis script; running
`python code/reproduce.py --from-data` rewrites every processed file.

Reviewer 2 asked for raw repeated-run results rather than only means, standard
deviations and significance labels. Every raw file below is per-run or per-trial and
is kept even where a summary of it also exists.

## Standardized dynamic benchmark (GMPB)

### `data/gmpb/` — primary protocol, 12 official cases, common population persistence

| File | Kind | Contents |
|---|---|---|
| `gmpb_primary_raw.csv` | raw | 1488 runs = 12 cases x 4 optimizers x 31 seeds; offline error, `E_bbc`, realized FE, diversity |
| `gmpb_primary_environments.csv` | raw | per-environment records for every primary run |
| `gmpb_sgo_temporal_raw.csv` | raw | SGO-only temporal modes on F2/F8/F10/F12 |
| `gmpb_sgo_temporal_environments.csv` | raw | per-environment records for the above |
| `gmpb_primary_summary.csv` | processed | per case/optimizer mean, median, SD, quartiles, rank |
| `gmpb_primary_offline_error.csv`, `gmpb_factor_analysis.csv` | processed | offline-error matrix; factor-axis effects |
| `gmpb_statistics.csv` | processed | paired Wilcoxon, bootstrap CI, Cliff's delta, Holm-corrected p |
| `benchmark_equivalence.csv` | verification | Python port vs official Octave, per probe |
| `benchmark_environment_checks.csv` | verification | validation gates A–J |
| `sgo_fidelity.csv` | verification | dynamic SGO driver vs the static SGO implementation |
| `gmpb_protocol.json`, `GMPB_PROTOCOL.md` | protocol | frozen design, case table, upstream hashes |
| `octave/` | third-party + driver | official GMPB generator plus this project's dump/refrun drivers |
| `tables/`, `figures/` | processed | generated LaTeX tables and figures |

### `data/gmpb_temporal_comparators/` — matched temporal transfer, 4 optimizers

| File | Kind | Contents |
|---|---|---|
| `gmpb_temporal_comparator_raw.csv` | raw | 1488 runs = 4 optimizers x 4 cases x 3 modes x 31 seeds |
| `gmpb_temporal_comparator_environments.csv` | raw | per-environment records |
| `gmpb_temporal_comparator_diversity_trace.csv.gz` | raw | per-environment diversity traces (gzip; no analysis script reads it) |
| `gmpb_temporal_comparator_summary.csv` | processed | per cell descriptive statistics |
| `gmpb_temporal_effects.csv` | processed | E1/E2/E3 within-optimizer effects, log ratios, Holm |
| `gmpb_temporal_cross_optimizer_statistics.csv` | processed | cross-optimizer effect comparisons |
| `gmpb_temporal_pattern_classification.csv` | processed | 12 case x effect patterns labelled GENERAL / ALGORITHM-DEPENDENT |
| `gmpb_temporal_behavioral_diagnostics.csv` | processed | per-run diversity, recovery fraction, late-half share |
| `temporal_equivalence_checks.csv` | verification | 23 reuse/equivalence checks |
| `gmpb_temporal_comparator_protocol.json`, `GMPB_TEMPORAL_COMPARATOR_PROTOCOL.md` | protocol | frozen design |

### `data/gmpb_mechanism/` — SGO grouping diagnostic

| File | Kind | Contents |
|---|---|---|
| `gmpb_baseline_repro_raw.csv` | raw | random grouping, 12 cases x 31 seeds (reproduces the primary SGO runs exactly) |
| `gmpb_fitness_group_raw.csv` | raw | fitness-ordered grouping, 12 cases x 31 seeds |
| `gmpb_temporal_interaction_raw.csv` | raw | fitness grouping under previous-best seeding, 4 cases x 31 seeds |
| `*_environments.csv`, `*_diversity_trace.csv` | raw | per-environment records and diversity traces |
| `gmpb_mechanism_summary.csv`, `gmpb_temporal_interaction_summary.csv` | processed | descriptive statistics |
| `gmpb_mechanism_statistics.csv` | processed | paired tests for offline error, diversity, recovery, late-half share, and the 2x2 interaction |
| `gmpb_mechanism_case_classification.csv` | processed | per-case direction labels |
| `baseline_equivalence.csv` | verification | 308 checks that random grouping equals the validated SGO |
| `gmpb_mechanism_protocol.json`, `GMPB_MECHANISM_PROTOCOL.md` | protocol | frozen design |

## MPC-generated objective sequences (`data/`, top level)

| File | Kind | Contents |
|---|---|---|
| `primary_closed_loop_raw.csv` | raw | fixed-MPC closed loop, 5 systems x 2 regimes x 4 optimizers x 20 seeds |
| `primary_per_step.csv` | raw | per-MPC-step records for the above |
| `frozen_sweep_v2_raw.csv` | raw | 215 frozen subproblems x 4 optimizers x 3 start conditions x 20 trials |
| `budget_sensitivity_v2_raw.csv` | raw | 9 instances x 5 budgets x 4 optimizers x 30 trials |
| `multi_ic_raw.csv` | raw | 5 systems x 5 initial conditions x 4 optimizers x 5 seeds = 500 runs |
| `dimension_sweep_raw.csv` | raw | certified (system, `N_c`) cells x 4 optimizers |
| `perturbation_raw.csv`, `perturbation_drift.csv` | raw | nominal + 3 perturbed conditions on HVAC and UAV |
| `input_error_sensitivity_raw.csv` | raw | per-step first-input error vs closed-loop tracking |
| `sgo_ablation_raw.csv` | raw | SGO operator ablation on the MPC instances |
| `objective_drift_full_raw.csv` | raw | every one of the 2400 consecutive deterministic transitions |
| `objective_drift_raw.csv` | raw | earlier sampled drift; still an input to the reference `N_c` sweep |
| `tail_reference_raw.csv`, `tail_stochastic_raw.csv` | raw | zero-tail vs hold-last prediction convention |
| `optimizer_only_tuning_raw.csv` | raw | optimizer-only tuning on held-out initial conditions |
| `sgo_benchmark_raw.csv` | raw | original-SGO reproduction, 13 functions x 2 conventions x 100 runs |
| `strict_fe_budget_verification.csv` | verification | 2000 cells; realized vs requested objective calls |
| `reference_certificates.csv`, `reference_nc_sweep_raw.csv` | verification | per-instant deterministic-reference certificates |
| `pendcart_penalty_box_certificate.csv` | verification | exact box extrema over 2265 instances |
| `*_summary.csv`, `*_association.csv`, `effect_sizes_primary*.csv`, `warm_start_mechanism.csv`, `frozen_sweep_by_drift.csv`, `submitted_vs_revised_*` | processed | recomputed by `code/reproduce.py --from-data` |
| `*_meta.json`, `*_protocol.json` | protocol | machine-readable design and run metadata |
| `figures/`, `tables/`, `figures_tables/` | processed | generated figures and LaTeX tables |
| `cache/` | regenerable | 400 `.npy` optimizer warm-start histories used by the frozen sweep; deterministic from the seeds, kept so a Level-2 rerun reproduces the exact warm starts |
| `protocols/` | protocol | frozen seed, drift-strata and tail-convention protocols |
| `provenance/` | provenance | hardware, OS, interpreter, package listings, BLAS configuration |
| `mpc_submitted_protocol/` | raw | the earlier jointly tuned protocol, retained only so the controller/optimizer co-design comparison can be recomputed |

## What is deliberately not here

- The manuscript, its bibliography, the response letters and the reviewer reports.
- Internal phase reports, planning notes and claim-audit spreadsheets.
- Superseded pipelines, one-off patch and figure-fixing scripts, and the earlier
  submitted-protocol simulation code (its data is kept; its code is not needed to
  recompute anything reported).
- GMPB benchmark state files: they are large binary generator states, regenerated
  deterministically with GNU Octave (see the README, Level 2).
