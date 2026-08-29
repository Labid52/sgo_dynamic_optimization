# GMPB grouping-mechanism protocol (frozen before outcomes)

Tests one pre-existing, scientifically motivated intervention against the vanilla
**Squid Game Optimizer** (Azizi, Baghalzadeh Shishehgarkhaneh, Basiri, Moehler,
*Scientific Reports* **13**, 5373, 2023, doi:10.1038/s41598-023-32465-z) on the
Generalized Moving Peaks Benchmark. This is a diagnostic experiment. It is not an
attempt to improve SGO, and the intervention is never treated as a new algorithm.

Committed before the final runs. No revision after seeing outcomes.

## 1. Scientific question

Is the early finite-budget plateau of vanilla SGO on GMPB associated with its
repeated random regrouping, that is, with insufficient intensification?

Secondary: does any grouping effect survive when stale full-population carryover
is removed and SGO instead uses previous-best seeding? This separates a
grouping/intensification effect from an effect of carrying the old population
across a change.

## 2. Benchmark

Inherited unchanged from `results_v2/gmpb/GMPB_PROTOCOL.md` and
`gmpb_protocol.json`: official EDOLAB GMPB, `github.com/EDOLAB-platform/EDOLAB-MATLAB`,
`main`, downloaded 2026-08-22; generator sha256 `11ffbfc2...`, fitness
`1bbf2fd8...`, FE wrapper `0b7f479c...`. The 12 official CEC instances, 100
environments, official `ChangeFrequency` budgets. **The same 372 official
environment state files already generated for the completed study are reused**,
so the benchmark seed is again the run index and every new run is paired to its
baseline counterpart by an identical environment trajectory.

## 3. Baseline

Vanilla SGO exactly as run in the completed GMPB study: `codes/sgo.py` operators
through the validated dynamic driver `codes/gmpb_optimizers.py:SGO`, population
size 20, informed-change protocol, official FE accounting. **The committed
baseline results are reused; no baseline run is repeated.**

## 4. Intervention

The single change is the group-assignment rule, which is the pre-existing
`C_fitness_group` ablation from `codes/sgo_ablation_impl.py` used for the flight
analysis:

| | Rule |
|---|---|
| baseline `grouping="random"` | fresh uniformly random permutation each SGO iteration; half offensive, half defensive |
| intervention `grouping="fitness"` | stable sort by current evaluated cost; best half offensive, worst half defensive |

No other operator, constant, bound or acceptance rule differs. No parameter is
added. No alternative grouping strategy is searched. The intervention is not
tuned, and no other SGO variant is run on GMPB.

`codes/sgo.py` and `codes/gmpb_optimizers.py` are not modified. The ablation
lives in `codes/gmpb_sgo_mechanism.py`, which subclasses the validated driver and
overrides `iteration` with a verbatim copy of the parent body in which only the
two grouping lines are parameterised.

### Equivalence gate (mandatory, already passed)

`codes/gmpb_mechanism_equivalence.py` compares the ablation driver with
`grouping="random"` against the validated driver on 28 configurations spanning
F1, F2, F5, F7, F8, F9, F10, F12, three benchmark seeds, both temporal modes and
four to eight environment transitions, checking evaluated points, objective
values, FE counter, environment counter, final population and costs, best
solution and cost, the `CurrentError` trace, the `Ebbc` vector and the offline
error. **Result: 308 checks, 0 failures, every difference exactly zero.**

Because `grouping="fitness"` skips the `rng.permutation` draw, the optimizer
random streams necessarily diverge from the first iteration. No claim of
evaluation-by-evaluation pairing is made for the intervention; pairing is by
benchmark environment trajectory only.

## 5. Design

**Primary, all 12 cases.** Fitness grouping under population persistence, the
same primary protocol as the completed study. 12 cases x 31 runs = **372 runs**.
Paired against the committed vanilla-SGO population-persistence results.

**Secondary, 2 x 2 interaction.** On the four cases frozen before the original
temporal diagnostic, F2 baseline, F8 fastest change, F10 highest dimension, F12
highest shift severity:

| | population persistence | previous-best seeded |
|---|---|---|
| random grouping | already committed | already committed |
| fitness grouping | part of the primary 372 | **124 new runs** |

Total new runs: **372 + 124 = 496**. Cold restart is not added. No comparator run
is repeated: GWO, PSO and WOA appear only as existing committed reference values.

## 6. Settings

Population size 20 for every run, identical bounds, identical official FE
accounting, 100 environments, `ChangeFrequency` evaluations per environment, and
the same optimizer-seed recipe as the baseline study,
`20260900 + 1000*run + 10*1 + mode_offset`. Sorting and group assignment consume
no objective evaluations. No intervention receives extra evaluations.

## 7. Recorded quantities

Per run: case, seed, grouping, temporal mode, `ChangeFrequency`, environments,
realized FE, re-evaluation FE, budget-exact flag, official offline error,
`E_bbc`, mean diversity, wall time.

Per environment, from the official `CurrentError` trace: post-change error, error
at 5, 10, 25, 50 and 75 percent of the environment budget, end-of-environment
error, mean error, error reduction and its fraction, `recovery_evals_50` and
`recovery_evals_90` with the definitions already frozen for the GMPB study, and
the same normalised by `ChangeFrequency`.

Late-stage improvement, defined here before outcomes:

    late_half_reduction = error_at_50_percent - end_of_environment_error
    late_half_fraction  = late_half_reduction / (post_change_error - end_of_environment_error)

reported only where the denominator is positive, and used as a mechanism
diagnostic rather than a performance metric.

Diversity trajectory, using the same normalised definition as the GMPB study,
`mean(std(X, axis=0)) / (ub - lb)`, recorded at the first completed optimizer
iteration at or after each of 0 (immediately after initialisation or the change
reaction), 5, 10, 25, 50, 75 and 100 percent of the environment budget. The
actual evaluation fraction reached is stored alongside the target; population
states that never existed are never interpolated.

Distance to the true optimum is **not** recorded, because exposing it to the
optimizer process risks information leakage and the mechanism questions do not
need it.

## 8. Hypotheses, predeclared

* **H1** Fitness grouping lowers persistent SGO population diversity relative to
  random grouping.
* **H2** Fitness grouping increases the share of improvement obtained later in an
  environment, and/or increases the evaluation fraction needed to reach 90% of
  eventual improvement.
* **H3** Fitness grouping reduces official offline error.
* **H4** If H1 to H3 hold under population persistence but vanish under
  previous-best seeding, grouping and stale-population carryover interact
  strongly.
* **H5** If the effect survives under previous-best seeding, grouping has an
  effect beyond stale-population carryover.
* **H6** The effect need not appear on every case. Universal improvement is not
  required to support a mechanism.

## 9. Statistics

The independent unit is the **run**, that is, the benchmark seed. Environment
level data are repeated measurements inside a run and are aggregated to one value
per run before any inferential test. No test treats environments as independent.

Primary: paired comparison of fitness against random grouping under population
persistence, within each case, over the 31 shared benchmark seeds. Paired
Wilcoxon signed-rank, Holm correction across the 12 case comparisons, paired
median and mean differences with bootstrap 95% intervals (10\,000 resamples), and
Cliff's delta with a bootstrap interval.

Mechanism diagnostics: the same paired treatment applied to per-run means of
diversity, normalised `recovery_evals_90`, `late_half_fraction` and
end-of-environment error, with Holm correction inside each diagnostic family.

Interaction: paired fitness-minus-random effects computed separately within
population persistence and within previous-best seeding, followed by a
difference-of-differences summary with a bootstrap interval. No large exploratory
ANOVA.

## 10. Interpretation rules

Evidence is classified as **STRONG MECHANISM SUPPORT** only if the intervention
produces a coherent chain, fitness grouping to lower persistent diversity to more
continued within-environment refinement to lower end-of-environment or offline
error, across a meaningful set of cases; **MODERATE** if performance improves and
the diagnostics move as predicted but inconsistently by case; **PERFORMANCE
EFFECT WITHOUT MECHANISM SUPPORT** if error improves while the diagnostics do
not move as predicted; **NO SUPPORT** if neither changes materially; and
**CONTRADICTORY** if performance worsens while the predicted diagnostic changes
appear, or other incompatible evidence emerges.

The word "cause" is not used merely because offline error improves. High
diversity is not described as intrinsically harmful; the question is whether the
observed persistent spread is associated with insufficient refinement under a
finite budget. The intervention is not given a name, is not compared with the
controls as though it were the original algorithm, and does not become the
object of study. The paper's object of study remains vanilla SGO.

## 11. Stop conditions

If the equivalence gate fails, the mechanism experiment is not run and the
discrepancy is reported. If FE accounting, environment counts or budget-exactness
fail on any new run, interpretation stops until the technical cause is fixed.

## 12. Expected outputs

`baseline_equivalence.csv`, `gmpb_fitness_group_raw.csv`,
`gmpb_fitness_group_environments.csv`, `gmpb_fitness_group_diversity_trace.csv`,
`gmpb_temporal_interaction_raw.csv`, `gmpb_temporal_interaction_summary.csv`,
`gmpb_mechanism_summary.csv`, `gmpb_mechanism_statistics.csv`,
`gmpb_mechanism_case_classification.csv`, figures
`gmpb_grouping_offline_error`, `gmpb_grouping_diversity`,
`gmpb_grouping_recovery`, `gmpb_grouping_temporal_interaction`.

No file under `data/gmpb/` is overwritten.
