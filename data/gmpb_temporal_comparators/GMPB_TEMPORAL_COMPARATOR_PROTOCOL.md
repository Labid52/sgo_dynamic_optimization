# Comparator temporal-transfer protocol (frozen before outcomes)

Asks whether the temporal-information effects already measured for the **Squid
Game Optimizer** (Azizi, Baghalzadeh Shishehgarkhaneh, Basiri, Moehler,
*Scientific Reports* **13**, 5373, 2023, doi:10.1038/s41598-023-32465-z) are
peculiar to it or are a general property of population-based optimizers on GMPB.

HEAD at freeze: `3307a08`. Committed before the final outcome runs. Not revised
after seeing results.

## 1. Question

The completed work established, for SGO alone, that
previous-best seeding < cold restart < population persistence in offline error on
F2, F8, F10 and F12. A within-SGO effect cannot show that the effect belongs to
SGO. This study measures the same three modes for GWO, PSO and WOA on the same
cases, seeds and environment trajectories, and compares the *temporal effects*
rather than the raw errors.

## 2. Benchmark

Inherited unchanged from `results_v2/gmpb/GMPB_PROTOCOL.md`: official EDOLAB
GMPB, generator sha256 `11ffbfc2...`, fitness `1bbf2fd8...`, FE wrapper
`0b7f479c...`. The same 372 official environment state files are reused, so the
benchmark seed is the run index and every optimizer and mode sees an identical
environment trajectory for a given seed. All comparisons are paired by
construction.

Cases, frozen before the original SGO temporal diagnostic and not reselected:
**F2** baseline, **F8** fastest change, **F10** highest dimension, **F12**
highest shift severity. 31 runs each, 100 environments, official
`ChangeFrequency` budgets.

## 3. Optimizers

SGO, GWO, PSO and WOA, from the validated `codes/gmpb_optimizers.py`, unmodified,
with their already frozen parameters and population size 20. No tuning, no added
algorithms, no operator changes.

## 4. Temporal modes

**A. Cold restart.** At each change the optimizer rebuilds its population and all
algorithm state exactly as a brand-new run would, and evaluates it in the new
environment. Verified: after a cold restart each optimizer's state is identical
to a freshly constructed optimizer drawing from the same random stream. For PSO
this means fresh positions, fresh velocities and a personal-best archive derived
entirely from new-environment evaluations; no previous velocity or personal-best
value is carried.

**B. Previous-best seeded.** The same fresh rebuild, with exactly one individual
replaced by that optimizer's own best point from the immediately previous
environment, re-evaluated in the new environment before use. Verified: slot 0
holds the carried point and every other slot is identical to an unseeded
initialisation from the same random stream. The carried solution is information
only; its old objective value is never reused.

**C. Native population and state persistence.** The persistence path already used
by the committed primary GMPB study, unchanged and not redesigned. What persists
is deliberately algorithm-native rather than artificially matched:

| Optimizer | What survives a change | What is re-evaluated |
|---|---|---|
| SGO | population `X`, best-so-far `BS`, global best | the whole population `X` |
| GWO | population `X`; leaders alpha, beta, delta are recomputed from the refreshed costs | the whole population `X` |
| PSO | positions `X`, velocities `V`, personal-best positions `P` | the personal-best archive `P`, which is the standard PSO change reaction and matches the official EDOLAB `ChangeReaction_RPSO` |
| WOA | population `X`; the leader `X*` is reselected from the refreshed costs | the whole population `X` |

## 5. Data reuse and new runs

Reused, not re-run for evidence: SGO under all three modes on the four cases
(372 runs, `results_v2/gmpb/gmpb_sgo_temporal_raw.csv`) and GWO, PSO and WOA
under persistence on the four cases (372 runs, drawn from
`results_v2/gmpb/gmpb_primary_raw.csv`). 744 runs reused.

**New scientific runs: 3 comparators x 2 new modes x 4 cases x 31 seeds = 744.**

Reproduction runs are additionally executed for every reusable cell so that the
secondary diagnostics of section 8, which the earlier studies did not record, are
available symmetrically for all twelve optimizer-by-mode cells. These are not new
evidence: each must reproduce its committed offline error exactly, and the study
stops if any does not.

## 6. Validation, already passed

`codes/gmpb_temporal_validation.py`, 23 checks, 0 failures
(`temporal_equivalence_checks.csv`): persistence reproduces the committed primary
results over 96 cells; SGO's three modes reproduce the committed temporal results
over 72 cells; fresh-state initialisation matches a brand-new optimizer for all
four methods; previous-best seeding alters exactly one population slot; FE counts
are exact with 100 environments and 99 changes in all twelve cells; no objective
value from before a change survives it, checked against a re-evaluation of
whichever array each optimizer refreshes; and all optimizers load the same
environment state file for a paired seed.

Note recorded during validation: the committed CSVs must be read with
`float_precision="round_trip"`. The pandas default parser is off by one unit in
the last place on 296 of 1488 rows, which produces apparent differences of up to
6e-14 where the stored values are in fact identical.

## 7. Evaluation budget

Official GMPB accounting throughout. Every optimizer, mode and case receives
exactly `ChangeFrequency` evaluations per environment and `ChangeFrequency` x 100
per run. Initialisation and post-change re-evaluation both consume budget and are
recorded separately. No mode grants extra evaluations: all twelve cells spend the
same 20 evaluations on initialisation and 1980 on post-change reactions.

## 8. Outcomes

**Primary:** official GMPB offline error, per optimizer, case, mode and run, with
mean, median, SD, IQR and bootstrap 95% intervals.

**The quantity of interest is not which optimizer has the lowest error but how
much each optimizer changes when the temporal mode changes.** Per optimizer and
case, on the 31 paired seeds:

    E1 = previous-best seeded  -  cold restart
    E2 = population persistence -  cold restart
    E3 = population persistence -  previous-best seeded

Lower error is better, so negative E1 means seeding helps, positive E2 and E3
mean persistence hurts. Because absolute error scales differ by an order of
magnitude between optimizers and cases, the cross-optimizer comparison uses the
paired log ratio `log(error_A / error_B)`, which is defined since all errors are
positive.

**Secondary diagnostics**, aggregated to one value per run before any test:
normalised population diversity; normalised evaluation fraction to 50% and 90% of
each environment's own recovery; late-half improvement fraction; and
end-of-environment error. Their purpose is only to test whether SGO's previously
observed high persistent spread and early plateau are shared by the controls.
Lower diversity is not assumed to be better; the preceding mechanism experiment
already refuted that reading.

## 9. Cross-optimizer test

A significant within-SGO effect is not sufficient. For each effect and case the
difference of paired effects is formed on the shared seeds, for example

    D(SGO, GWO) = E3_SGO - E3_GWO

in log-ratio form, with a paired bootstrap 95% interval and a paired
nonparametric test. Holm correction is applied within the predefined family of
three comparisons (SGO against GWO, PSO and WOA) for each effect and case.
Magnitude is reported alongside significance and is not subordinate to it.

## 10. Statistics

The independent unit is the run, that is, the benchmark seed. The 100
environments inside a run are repeated measurements and are aggregated within the
run before any inferential test. Paired Wilcoxon signed-rank, Cliff's delta with
a bootstrap interval, and bootstrap 95% intervals for paired median differences,
10\,000 resamples.

## 11. Interpretation categories

Each temporal-transfer pattern is classified as **GENERAL** (same direction
across all or most methods, SGO not materially different in magnitude),
**ALGORITHM-DEPENDENT** (different optimizers prefer different strategies with no
uniquely SGO pattern), **SGO-AMPLIFIED** (same direction generally, SGO's effect
significantly larger), **SGO-DISTINCTIVE** (SGO's direction or pattern differs
materially and statistically from the controls), or **INCONCLUSIVE**.

Nothing is called SGO-specific unless the cross-optimizer comparison supports it.
A result is not highlighted merely because a p-value is small, because a harder
case is harder, or because every method degrades under faster change.

## 12. Stop conditions

If any validation check fails, no scientific interpretation follows until the
cause is found and fixed. If FE accounting, environment counts or budget
exactness fail on any new run, interpretation stops.

## 13. Outputs

`temporal_equivalence_checks.csv`, `gmpb_temporal_comparator_raw.csv`,
`gmpb_temporal_comparator_summary.csv`, `gmpb_temporal_effects.csv`,
`gmpb_temporal_cross_optimizer_statistics.csv`,
`gmpb_temporal_behavioral_diagnostics.csv`,
`gmpb_temporal_pattern_classification.csv`, figures
`temporal_modes_by_optimizer`, `temporal_transfer_effect_sizes`,
`temporal_effect_comparison`, `temporal_behavior_profiles`.

No existing GMPB or mechanism file is overwritten.
