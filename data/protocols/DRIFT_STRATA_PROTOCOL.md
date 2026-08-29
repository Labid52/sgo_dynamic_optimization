# Drift-strata protocol — frozen before any frozen-sweep optimizer trial

Machine-readable definition and the complete instance list with the drift value that caused each
assignment: `results_v2/drift_strata_definition.json`. Generator: `codes/drift_strata.py`.

**Selection used drift metrics only.** No optimizer result was consulted, and this protocol is committed
before the frozen sweep runs.

## Stratification variable

The **shifted-reference warm-start normalized suboptimality**

```
warm_gap_norm(k) = [ J_{k+1}(shift(U*_k)) - J*_{k+1} ] / max(1, |J*_{k+1}|)
```

It directly measures the optimization consequence of moving from one receding-horizon problem to the
next, on the same normalized scale as the reported optimality gaps.

## Why absolute cut points and not quantiles

Quantile strata were considered first and rejected on the evidence of the drift distribution alone:
**93 % of all 2 400 transitions fall below the 1e-4 criterion**, and per-system medians span 1e-15 to
1e-6. A 50th/75th/90th-percentile split would therefore place the Low/Moderate boundary inside numerical
noise and would leave the severe tail undistinguished — exactly the failure mode the Phase-2B brief
anticipated ("if the distribution has a large point mass near zero, adapt the strata transparently").

Absolute cut points on the optimality-gap scale:

| Stratum | Range of `warm_gap_norm` | Count (all systems) |
|---|---|---|
| Benign | ≤ 1e-4 | 2 229 |
| Moderate | 1e-4 – 1e-2 | 106 |
| High | 1e-2 – 1e-1 | 46 |
| Severe | > 1e-1 | 19 |

Per system:

| System | Benign | Moderate | High | Severe |
|---|---|---|---|---|
| Pendulum-cart | 98 | 40 | 12 | 0 |
| CSTR | 85 | 4 | 9 | 2 |
| Flight control | 107 | 37 | 5 | 1 |
| HVAC | 1 500 | 0 | 0 | 0 |
| UAV | 439 | 25 | 20 | 16 |

## Known imbalance — recorded, not engineered away

Two facts are reported rather than fixed by redrawing boundaries:

1. **The Severe stratum is UAV-dominated** (16 of 19). Any pooled Severe-stratum statistic is largely a
   UAV statistic.
2. **HVAC contributes no non-benign transition at all.** Its maximum warm gap over 1 500 transitions is
   6.3e-5. HVAC therefore cannot appear in any stratum above Benign, and no "HVAC under high drift"
   statement can be made from these data.

Mitigation, decided before running: the drift-performance analysis reports **both** per-stratum
statistics **and** a continuous Spearman association between `warm_gap_norm` and optimizer error, the
latter being independent of stratum balance and of how many instances each system contributes. Strata
were not redrawn to balance systems, and no boundary was chosen with reference to optimizer outcomes.

## Frozen-instance selection

Per system:

- **Uniform coverage:** 25 evenly spaced steps over the trajectory.
- **Stratified coverage:** up to 8 instances per stratum, taken at evenly spaced *ranks within that
  stratum's warm-gap ordering* — deterministic and drift-only.
- Duplicates between the two sets are recorded explicitly in the JSON, not silently dropped.

| System | Uniform | Stratified | Duplicates | Total selected |
|---|---|---|---|---|
| Pendulum-cart | 25 | 24 | 4 | 45 |
| CSTR | 25 | 22 | 6 | 41 |
| Flight control | 25 | 22 | 6 | 41 |
| HVAC | 25 | 8 | 1 | 32 |
| UAV | 25 | 32 | 1 | 56 |
| **Total** | | | | **215** |

Every Severe and High instance available in the data is represented, so the tail is covered despite
being rare.

## Fixed experimental settings for the sweep

| Setting | Value |
|---|---|
| MPC formulation | fixed-MPC Phase-2A protocol (P per system, N_c=1, Q_scale=1) |
| FE budget | 1000 objective evaluations, exact, enforced per objective call |
| Early/stagnation stopping | disabled |
| Optimizers | SGO, GWO, PSO, WOA |
| Start conditions | cold, deterministic-reference warm, optimizer warm |
| Trials | 20 per (system, instance, optimizer, start condition) |
| Seeds | `SEED_BASE["frozen_v2"] = 11_000_000`, collision-free, trial fastest-varying |
| Deterministic reference | OSQP (certified at N_c=1 for all systems) |

None of these may change after optimizer results are seen. If a genuine implementation error forces a
change, the affected cells are invalidated, rerun, and the correction committed separately.
