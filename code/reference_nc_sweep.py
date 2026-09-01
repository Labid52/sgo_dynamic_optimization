#!/usr/bin/env python3
"""Phase-2B step 2: certify the deterministic reference at larger control horizons.

Phase 1 found that multistart L-BFGS-B alone did NOT certify the UAV at Nc = 2, 3
(suboptimality bound 4.3e-3, i.e. 43x looser than the 1e-4 criterion at which the
study declares convergence). Phase 2A showed OSQP reduces the Nc = 1 UAV bound
from 2.7e-6 to 4.9e-26. This script decides, per (system, Nc), whether the
deterministic reference is trustworthy enough to serve as the denominator of an
optimality gap at that dimension.

ACCEPTANCE CRITERIA — pre-registered, fixed before the results are seen:

  C1  OSQP reports "solved" at every evaluated instant.
  C2  The strong-convexity suboptimality certificate of the accepted reference,
      normalized as the optimality gaps are, satisfies
          max_k  subopt_bound_norm  <=  1e-6
      i.e. at least two orders of magnitude below the 1e-4 convergence criterion.
  C3  The two independent solvers agree on the objective:
          max_k  |J_OSQP - J_LBFGSB| / max(1,|J|)  <=  1e-6.
  C4  The Hessian is positive definite (lambda_min > 0), so the certificate in
      C2 is valid at all.

A (system, Nc) is CERTIFIED only if C1-C4 all hold. Anything else is reported as
NOT certified and remains blocked for optimizer gap studies at that dimension.

Instants sampled per system (five roles, so the certificate is not driven by one
easy point): k=0, early transient, mid-trajectory, the highest-drift instant
known from the Phase-1 sampled drift scan, and a near-settled late instant.

Nc grid: 1, 2, 3, 5, and 8 where structurally valid (Nc <= P).

Outputs:
    data/reference_nc_sweep_raw.csv
    data/reference_nc_sweep_summary.csv
    data/tables/reference_nc_certification.tex
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
# Output root. Defaults to the released data/ directory; set SGO_OUTPUT_DIR to
# redirect every generated file elsewhere (e.g. a scratch directory) so that a
# smoke test cannot overwrite the committed production evidence.
OUT = os.environ.get("SGO_OUTPUT_DIR") or os.path.join(PROJECT, "data")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, build_problem                         # noqa: E402
import phase2_protocol as P2                                           # noqa: E402
from reference_solvers import deterministic_reference                  # noqa: E402

NC_GRID = [1, 2, 3, 5, 8]
# Pre-registered acceptance criteria
TOL_SUBOPT_NORM = 1e-6
TOL_SOLVER_AGREEMENT = 1e-6
CONV_CRITERION = 1e-4          # the scale the study uses to declare convergence


def instants_for(key, Nsim):
    """Five roles, so the certificate is not driven by a single easy point."""
    drift = pd.read_csv(os.path.join(OUT, "objective_drift_raw.csv"))
    d = drift[(drift.system == key) & (drift.Nc == 1)]
    high = int(d.loc[d.warm_gap_norm.idxmax(), "k"]) if len(d) else Nsim // 3
    cand = {
        "k0": 0,
        "early_transient": max(1, int(0.05 * Nsim)),
        "mid_trajectory": int(0.50 * Nsim),
        "high_drift_phase1": min(max(high, 0), Nsim - 1),
        "near_settled": int(0.90 * Nsim),
    }
    # de-duplicate while keeping role labels
    seen, out = set(), {}
    for role, k in cand.items():
        if k not in seen:
            out[role] = k
            seen.add(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", type=int, nargs="*", default=NC_GRID)
    args = ap.parse_args()
    t0 = time.perf_counter()

    ics = P2.load_initial_conditions()
    rows = []
    for key, sysdef in get_systems().items():
        P = int(sysdef.P)
        x0 = ics[key]["eval"]
        base = simulate(key, "QP", params=P2.fixed_mpc_params(sysdef), seed=11,
                        verbose=False, x0_override=x0)
        xh, Nsim = base["x_hist"], base["Nsim"]
        roles = instants_for(key, Nsim)
        for Nc in args.nc:
            if Nc > P:
                print(f"  {key}: skipping Nc={Nc} (exceeds prediction horizon P={P})")
                continue
            problem = build_problem(sysdef, Nc, 1.0, P)
            for role, k in roles.items():
                _, J_ref, rec = deterministic_reference(
                    problem, sysdef, xh[:, k], int(k), allow_osqp=True,
                    n_restarts=20, seed=900 + int(k))
                rec.update(system=key, Nc=Nc, P=P, dim=int(Nc * sysdef.nu),
                           step=int(k), role=role)
                rows.append(rec)
            print(f"  {key:9s} Nc={Nc}: max subopt bound "
                  f"{max(r['subopt_bound_norm'] for r in rows if r['system']==key and r['Nc']==Nc):.3e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "reference_nc_sweep_raw.csv"), index=False)

    recs = []
    for (key, Nc), g in df.groupby(["system", "Nc"]):
        c1 = bool((g.osqp_status.astype(str) == "solved").all())
        max_bound = float(g.subopt_bound_norm.max())
        max_agree = float(g.obj_rel_diff.max(skipna=True)) if g.obj_rel_diff.notna().any() else np.nan
        c2 = bool(max_bound <= TOL_SUBOPT_NORM)
        c3 = bool(np.isfinite(max_agree) and max_agree <= TOL_SOLVER_AGREEMENT)
        c4 = bool((g.lambda_min > 0).all())
        recs.append(dict(system=key, Nc=int(Nc), dim=int(g.dim.iloc[0]),
                         instants=int(len(g)),
                         C1_osqp_solved=c1, C2_subopt_bound=c2, C3_agreement=c3,
                         C4_pos_def=c4,
                         max_subopt_bound_norm=max_bound,
                         max_solver_rel_disagreement=max_agree,
                         max_solution_inf_diff=float(g.sol_inf_diff.max(skipna=True)),
                         max_lbfgsb_multistart_spread=float(g.lbfgsb_multistart_spread.max()),
                         lambda_min=float(g.lambda_min.min()),
                         cond_H=float(g.cond_H.max()),
                         margin_vs_conv_criterion=float(CONV_CRITERION / max(max_bound, 1e-300)),
                         certified=bool(c1 and c2 and c3 and c4)))
    summ = pd.DataFrame(recs).sort_values(["system", "Nc"])
    summ.to_csv(os.path.join(OUT, "reference_nc_sweep_summary.csv"), index=False)

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Deterministic-reference certification as a function of the control "
             r"horizon. OSQP is the primary reference and multistart L-BFGS-B the independent "
             r"cross-check. A case is certified when OSQP solves at every instant, the Hessian "
             r"is positive definite, the two solvers agree to $10^{-6}$ relative, and the "
             r"strong-convexity suboptimality certificate is at most $10^{-6}$ -- two orders of "
             r"magnitude below the $10^{-4}$ criterion used to declare convergence.}",
             r"\label{tab:reference_nc_certification}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"System & $N_c$ & dim & Max subopt.\ bound & Max solver disagreement & "
             r"$\kappa(H)$ & Certified \\", r"\midrule"]
    yes_cell, no_cell = "yes", r"\textbf{no}"
    for _, r in summ.iterrows():
        cert = yes_cell if r.certified else no_cell
        lines.append(f"{r.system} & {r.Nc} & {r.dim} & {r.max_subopt_bound_norm:.2e} & "
                     f"{r.max_solver_rel_disagreement:.2e} & {r.cond_H:.2e} & "
                     f"{cert} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(OUT, "tables", "reference_nc_certification.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    meta = {"criteria": {"C1": "OSQP status == solved at every instant",
                         "C2": f"max normalized suboptimality bound <= {TOL_SUBOPT_NORM}",
                         "C3": f"max relative solver disagreement <= {TOL_SOLVER_AGREEMENT}",
                         "C4": "lambda_min(H) > 0"},
            "convergence_criterion_scale": CONV_CRITERION,
            "certified_cases": summ[summ.certified].apply(
                lambda r: f"{r.system}/Nc{r.Nc}", axis=1).tolist(),
            "not_certified_cases": summ[~summ.certified].apply(
                lambda r: f"{r.system}/Nc{r.Nc}", axis=1).tolist(),
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(OUT, "reference_nc_sweep_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
