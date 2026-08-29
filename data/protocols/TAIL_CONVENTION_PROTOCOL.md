# Zero-tail vs hold-last sensitivity — protocol (frozen before results)

Machine-readable form: `results_v2/tail_convention_protocol.json`.
Implementation: `codes/tail_prediction.py` (standalone), `codes/tail_convention_check.py`.
Production code is not modified.

Purpose: answer reviewer 1 comment 2 — why the prediction sets the input contribution to zero beyond
the control horizon instead of holding the last optimized input — with evidence rather than assertion.
This is **not** an attempt to decide which convention is universally better.

## 1. The two conventions, as implemented

Decision-vector semantics read from the code, not assumed:
`U = [u_0; …; u_{N_c-1}]` are **absolute physical inputs**, bounded elementwise by `[u_min, u_max]`;
`X_pred = Φ x + Γ U` with `Γ ∈ R^{P n_x × N_c n_u}`.

**Zero-tail (submitted and revised implementation).** `mpc_utils.build_prediction_matrices` fills
`Γ[i,j] = A^{i-j} B` only for `j < min(i+1, N_c)`, so the prediction sum truncates at `j = N_c-1`.
That is exactly `u_j = 0 for j ≥ N_c`.

**Hold-last (this experiment).** `u_j = u_{N_c-1} for j ≥ N_c`. Substituting into
`x_{k+i+1} = A^{i+1}x + Σ_{j≤i} A^{i-j} B u_j` leaves every earlier column block unchanged and makes the
last block accumulate the tail:

```
Γ_hold[i, N_c-1] = Σ_{p=0}^{i-(N_c-1)} A^p B        for i ≥ N_c-1
```

**Consequence at N_c = 1** (the Phase-2A primary configuration): the last control index is also the
first, so hold-last applies `u_0` over the *entire* prediction horizon and
`Γ[i,0] = Σ_{p=0}^{i} A^p B` rather than `A^i B`. This is a substantially different prediction model,
not a marginal tail correction. The experiment is therefore not expected a priori to be negligible.

**Not to be confused with warm starting.** The prediction-tail convention governs the model *inside* one
optimisation. The shift-and-hold rule used between MPC steps constructs an initial guess for the *next*
optimisation. They are independent, and only the first is varied here.

## 2. Held fixed

Prediction horizon P, control horizon N_c, Q, R, Q_scale = 1, plant model, reference, evaluation initial
condition, input bounds, objective definition, sampling time, optimizer parameters from Phase-2A tuning.
Only the tail convention changes.

MPC configuration: the frozen Phase-2A fixed-MPC primary configuration (P per system from
`system_defs.py`, **N_c = 1**, Q_scale = 1). The old jointly tuned settings are not used.

## 3. Stages

1. **Equivalence gate.** The standalone `tail="zero"` path must reproduce production
   `mpc_utils.build_prediction_matrices` to numerical precision, on random feasible `U`, on the
   deterministic optimum, across several trajectory instants and all five systems. If it does not,
   STOP before any hold-last evaluation.
2. **Deterministic comparison, all five systems**, at 25 uniformly spaced instants per system: solve
   both conventions with OSQP, cross-check with multistart L-BFGS-B, and compare `J*`, `‖U*‖`, and —
   reported separately — the **first applied input**, since only that reaches the plant.
3. **Deterministic closed loop, all five systems**, from the Phase-2A evaluation IC under each
   convention.
4. **Stochastic sensitivity, flight and UAV**: SGO/GWO/PSO/WOA, exact 1000 objective evaluations per
   MPC step, early stopping disabled, Phase-2A optimizer parameters, **10 runs** per
   optimizer × system × convention, paired seeds. No retuning under hold-last.

## 4. Convexity and reference validity

Hold-last is a linear re-mapping of the same decision vector, so the quadratic form should stay convex —
**this is verified, not assumed**: λ_min(H) is computed for both conventions, and OSQP is cross-checked
against multistart L-BFGS-B.

For the pendulum-cart the soft cart penalty was certified globally inactive under zero-tail. Hold-last
produces larger predicted excursions, so the exact box-extrema certificate is **recomputed** under
hold-last. If the penalty can activate, that instance is not a pure QP and OSQP is not used for it; the
fact is reported rather than worked around.

## 5. Metrics, kept separate

Three families, never conflated:

- **finite-horizon optimization accuracy** — normalized gap against the deterministic reference *of the
  same convention*. A hold-last stochastic solution is never compared with a zero-tail optimum.
- **first-applied-control accuracy** — `‖u_0 − u_0*‖`, the only quantity that reaches the plant.
- **closed-loop tracking** — nRMSE.

Lower closed-loop nRMSE is never treated as evidence of better finite-horizon optimization.

## 6. Material-change criteria (declared in advance)

Classification of the tail-convention effect:

| Label | Criteria (all must hold) |
|---|---|
| **negligible** | mean first-applied-input change < 1e-6 of the input range; relative closed-loop nRMSE change < 0.1 %; no ordering change in either metric family |
| **small** | mean first-applied-input change < 1 % of the input range; relative closed-loop nRMSE change < 1 %; no ordering change in either metric family |
| **material** | anything else — in particular **any** ordering change in either metric family, or a relative nRMSE change ≥ 1 % |

Ordering is assessed separately for optimization accuracy and for closed-loop nRMSE, and a disagreement
between the two is reported explicitly.

## 7. Seeds

`SEED_BASE["tail"] = 17_000_000`, collision-free, trial fastest-varying, and **paired across
conventions**: the same trial index uses the same seed under zero-tail and hold-last, so the comparison
is within-trial.

## 8. Stop conditions

Stop and report before enlarging the experiment if: the equivalence gate fails; hold-last loses convexity
and the reference cannot be certified; exact FE accounting fails; a production-code defect is found; or
the tail convention materially changes an established Phase-2A/2B/2C conclusion. In the last case,
preserve all artifacts, perform only the minimum analysis needed to identify what changed, and return
the evidence — do not rerun the completed pipeline.
