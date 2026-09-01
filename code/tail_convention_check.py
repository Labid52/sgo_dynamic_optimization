#!/usr/bin/env python3
"""Phase-2C addendum: zero-tail vs hold-last prediction sensitivity (reviewer R1.2).

Standalone. Production code is not modified; this script builds its own MPC
problems via `tail_prediction.build_prediction_matrices_tail`.

Stages (each a separate --stage):
    equivalence    standalone tail="zero" must reproduce production exactly.
                   Gate: nothing else runs until this passes.
    deterministic  all five systems, 25 instants, both conventions, OSQP +
                   multistart L-BFGS-B cross-check, convexity verified, and the
                   pendulum-cart penalty box certificate recomputed.
    closedloop     deterministic closed loop under both conventions, 5 systems.
    stochastic     flight and UAV, 4 optimizers, exact 1000 FE, 10 paired runs.

Outputs are written under data/ with a `tail_` prefix.
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
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import (MPCProblem, normalized_rmse, CountedCost,      # noqa: E402
                        BudgetExhausted, OPTIMIZERS)
from mpc_utils import (build_prediction_matrices, build_qp_matrices,   # noqa: E402
                       compute_f, normalize_cost)
import phase2_protocol as P2                                           # noqa: E402
from tail_prediction import build_prediction_matrices_tail, cart_box_extrema  # noqa: E402
from reference_solvers import osqp_reference, lbfgsb_crosscheck        # noqa: E402

TAILS = ["zero", "hold_last"]
N_INSTANTS = 25
RUNS = 10
CONV_TOL = 1e-4
SEED_BASE = P2.SEED_BASE["tail"]


# --------------------------------------------------------------------------
def build_problem_tail(sysdef, Nc, Q_scale, P, tail):
    """MPCProblem under a chosen tail convention (mirrors sim_common.build_problem)."""
    Phi, Gamma = build_prediction_matrices_tail(sysdef.Ad, sysdef.Bd, P, Nc, tail=tail)
    Q_block = np.kron(np.eye(P), Q_scale * sysdef.Q)
    R_block = np.kron(np.eye(Nc), sysdef.R)
    H = build_qp_matrices(Phi, Gamma, Q_block, R_block)
    lb = sysdef.u_min * np.ones(Nc * sysdef.nu)
    ub = sysdef.u_max * np.ones(Nc * sysdef.nu)
    return MPCProblem(sysdef, P, Nc, Phi, Gamma, Q_block, H, lb, ub)


def make_cost_tail(problem, sysdef, x, k):
    """Objective for a tail-specific problem; identical in form to make_cost."""
    x_c = sysdef.cost_state(x)
    ref = sysdef.reference_horizon(k, problem.P)
    f = compute_f(problem.Phi, problem.Gamma, problem.Q_block, x_c, ref)

    def cost(U):
        U = np.asarray(U, dtype=float).ravel()
        J = 0.5 * (U @ problem.H @ U) + f @ U
        if sysdef.cart_constraint is not None:
            S = np.kron(np.eye(problem.P), np.array([[1, 0, 0, 0]], dtype=float))
            pred = problem.Phi @ x_c + problem.Gamma @ U
            cart = (S @ pred).ravel()
            J += sysdef.constraint_penalty * np.sum(
                np.maximum(np.abs(cart) - sysdef.cart_constraint, 0.0) ** 2)
        return float(J)
    return cost, f


def det_traj(key, sysdef, tail, x0, Nsim=None):
    """Deterministic closed loop under a tail convention (OSQP each step)."""
    mpc = P2.fixed_mpc_params(sysdef)
    prob = build_problem_tail(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"], tail)
    N = int(sysdef.Nsim if Nsim is None else Nsim)
    x = np.asarray(x0, float).copy()
    xh = np.zeros((sysdef.nx, N + 1)); xh[:, 0] = x
    uh = np.zeros((sysdef.nu, N))
    for k in range(N):
        cost, f = make_cost_tail(prob, sysdef, x, k)
        u, status, _ = osqp_reference(prob, f)
        if u is None:
            raise RuntimeError(f"OSQP failed {key}/{tail} k={k}: {status}")
        u = np.clip(u, prob.lb_seq, prob.ub_seq)
        applied = np.clip(u[:sysdef.nu], sysdef.u_min, sysdef.u_max)
        x = sysdef.step(x, applied)
        xh[:, k + 1] = x; uh[:, k] = applied
    return prob, xh, uh


# ------------------------------------------------------------- stage: equivalence
def stage_equivalence():
    ics = P2.load_initial_conditions()
    rng = np.random.default_rng(11)
    rows = []
    for key, sysdef in get_systems().items():
        mpc = P2.fixed_mpc_params(sysdef)
        Pp, Nc = mpc["P"], mpc["Nc"]
        Phi_p, G_p = build_prediction_matrices(sysdef.Ad, sysdef.Bd, Pp, Nc)
        Phi_s, G_s = build_prediction_matrices_tail(sysdef.Ad, sysdef.Bd, Pp, Nc, "zero")
        mat_phi = float(np.max(np.abs(Phi_p - Phi_s)))
        mat_g = float(np.max(np.abs(G_p - G_s)))

        prob_p = build_problem_tail(sysdef, Nc, mpc["Q_scale"], Pp, "zero")
        # production problem via the production builder, for an end-to-end check
        from sim_common import build_problem as prod_build
        prod = prod_build(sysdef, Nc, mpc["Q_scale"], Pp)
        from sim_common import make_cost as prod_make_cost

        _, xh, _ = det_traj(key, sysdef, "zero", ics[key]["eval"],
                            Nsim=min(sysdef.Nsim, 60))
        steps = sorted(set(np.linspace(0, xh.shape[1] - 2, 6).astype(int).tolist()))
        for k in steps:
            x = xh[:, k]
            c_std, f_std = make_cost_tail(prob_p, sysdef, x, int(k))
            c_prd = prod_make_cost(prod, x, int(k))
            worst_obj = 0.0
            for _ in range(40):
                U = prod.lb_seq + rng.random(len(prod.lb_seq)) * (prod.ub_seq - prod.lb_seq)
                worst_obj = max(worst_obj, abs(c_std(U) - c_prd(U)) /
                                max(1.0, abs(c_prd(U))))
            u_s, st_s, _ = osqp_reference(prob_p, f_std)
            f_prd = compute_f(prod.Phi, prod.Gamma, prod.Q_block,
                              sysdef.cost_state(x), sysdef.reference_horizon(int(k), prod.P))
            u_p, st_p, _ = osqp_reference(prod, f_prd)
            u_s = np.clip(u_s, prod.lb_seq, prod.ub_seq)
            u_p = np.clip(u_p, prod.lb_seq, prod.ub_seq)
            rows.append(dict(system=key, k=int(k), Phi_max_diff=mat_phi,
                             Gamma_max_diff=mat_g,
                             objective_max_rel_diff=worst_obj,
                             optimum_obj_rel_diff=abs(c_std(u_s) - c_prd(u_p)) /
                             max(1.0, abs(c_prd(u_p))),
                             solution_inf_diff=float(np.max(np.abs(u_s - u_p))),
                             first_control_diff=float(np.linalg.norm(
                                 u_s[:sysdef.nu] - u_p[:sysdef.nu])),
                             osqp_status_standalone=st_s, osqp_status_production=st_p))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tail_zero_equivalence.csv"), index=False)
    tol = 1e-12
    ok = bool(df.Phi_max_diff.max() == 0 and df.Gamma_max_diff.max() == 0
              and df.objective_max_rel_diff.max() <= tol
              and df.optimum_obj_rel_diff.max() <= tol
              and df.first_control_diff.max() <= 1e-9)
    print(f"  Phi max diff            {df.Phi_max_diff.max():.3e}")
    print(f"  Gamma max diff          {df.Gamma_max_diff.max():.3e}")
    print(f"  objective max rel diff  {df.objective_max_rel_diff.max():.3e}")
    print(f"  optimum obj rel diff    {df.optimum_obj_rel_diff.max():.3e}")
    print(f"  solution inf diff       {df.solution_inf_diff.max():.3e}")
    print(f"  first-control diff      {df.first_control_diff.max():.3e}")
    print(f"EQUIVALENCE: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------- stage: deterministic
def stage_deterministic():
    ics = P2.load_initial_conditions()
    rows, cert = [], []
    for key, sysdef in get_systems().items():
        mpc = P2.fixed_mpc_params(sysdef)
        x0 = ics[key]["eval"]
        _, xh, _ = det_traj(key, sysdef, "zero", x0)
        Nsim = xh.shape[1] - 1
        steps = sorted(set(np.linspace(0, Nsim - 1, N_INSTANTS).astype(int).tolist()))
        probs = {t: build_problem_tail(sysdef, mpc["Nc"], mpc["Q_scale"], mpc["P"], t)
                 for t in TAILS}
        for t in TAILS:
            eig = np.linalg.eigvalsh(0.5 * (probs[t].H + probs[t].H.T))
            cert.append(dict(system=key, tail=t, lambda_min=float(eig.min()),
                             lambda_max=float(eig.max()),
                             cond_H=float(eig.max() / eig.min()) if eig.min() > 0 else np.nan,
                             convex=bool(eig.min() > 0)))
        for k in steps:
            k = int(k); x = xh[:, k]
            res = {}
            for t in TAILS:
                prob = probs[t]
                cost, f = make_cost_tail(prob, sysdef, x, k)
                u, status, _ = osqp_reference(prob, f)
                u = np.clip(u, prob.lb_seq, prob.ub_seq)
                J = float(cost(u))
                u_l, J_l, info = lbfgsb_crosscheck(prob, x, k, n_restarts=20, seed=900 + k) \
                    if False else (None, np.nan, {"multistart_spread": np.nan})
                # multistart cross-check needs a tail-aware objective, so do it directly
                from scipy.optimize import minimize
                bnds = list(zip(prob.lb_seq, prob.ub_seq))
                rng = np.random.default_rng(900 + k)
                vals, best = [], (np.inf, None)
                for s0 in [np.zeros_like(prob.lb_seq)] + [
                        prob.lb_seq + rng.random(len(prob.lb_seq)) *
                        (prob.ub_seq - prob.lb_seq) for _ in range(20)]:
                    r = minimize(cost, s0, method="L-BFGS-B", bounds=bnds,
                                 options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 200})
                    v = float(cost(np.clip(r.x, prob.lb_seq, prob.ub_seq)))
                    vals.append(v)
                    if v < best[0]:
                        best = (v, np.clip(r.x, prob.lb_seq, prob.ub_seq))
                J_l, spread = best[0], float(max(vals) - min(vals))
                pen = np.nan
                if sysdef.cart_constraint is not None:
                    hi, lo = cart_box_extrema(prob.Phi, prob.Gamma, prob.P,
                                              sysdef.cost_state(x),
                                              prob.lb_seq, prob.ub_seq)
                    pen = float(max(np.max(np.abs(hi)), np.max(np.abs(lo))))
                res[t] = dict(J=J, u=u, status=status, J_lbfgsb=J_l, spread=spread,
                              worst_cart=pen,
                              n_active=int(np.sum((u <= prob.lb_seq + 1e-9) |
                                                  (u >= prob.ub_seq - 1e-9))))
            z, h = res["zero"], res["hold_last"]
            nu = sysdef.nu
            urange = float(np.max(probs["zero"].ub_seq - probs["zero"].lb_seq))
            rows.append(dict(
                system=key, k=k, Nc=mpc["Nc"], P=mpc["P"],
                J_zero=z["J"], J_hold=h["J"],
                J_abs_diff=abs(h["J"] - z["J"]),
                J_rel_diff=abs(h["J"] - z["J"]) / max(1.0, abs(z["J"])),
                U_inf_diff=float(np.max(np.abs(h["u"] - z["u"]))),
                U_norm_diff=float(np.linalg.norm(h["u"] - z["u"])),
                first_control_diff=float(np.linalg.norm(h["u"][:nu] - z["u"][:nu])),
                first_control_diff_rel_range=float(
                    np.linalg.norm(h["u"][:nu] - z["u"][:nu]) / urange),
                osqp_status_zero=z["status"], osqp_status_hold=h["status"],
                lbfgsb_rel_zero=abs(z["J_lbfgsb"] - z["J"]) / max(1.0, abs(z["J"])),
                lbfgsb_rel_hold=abs(h["J_lbfgsb"] - h["J"]) / max(1.0, abs(h["J"])),
                spread_zero=z["spread"], spread_hold=h["spread"],
                worst_cart_zero=z["worst_cart"], worst_cart_hold=h["worst_cart"],
                n_active_zero=z["n_active"], n_active_hold=h["n_active"]))
        print(f"  {key}: {len(steps)} instants done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tail_reference_raw.csv"), index=False)
    pd.DataFrame(cert).to_csv(os.path.join(OUT, "tail_convexity_certificate.csv"), index=False)
    summ = (df.groupby("system")
              .agg(instants=("k", "size"),
                   J_rel_diff_median=("J_rel_diff", "median"),
                   J_rel_diff_max=("J_rel_diff", "max"),
                   U_norm_diff_median=("U_norm_diff", "median"),
                   first_ctrl_diff_median=("first_control_diff", "median"),
                   first_ctrl_diff_max=("first_control_diff", "max"),
                   first_ctrl_rel_range_median=("first_control_diff_rel_range", "median"),
                   first_ctrl_rel_range_max=("first_control_diff_rel_range", "max"),
                   lbfgsb_rel_zero_max=("lbfgsb_rel_zero", "max"),
                   lbfgsb_rel_hold_max=("lbfgsb_rel_hold", "max"),
                   worst_cart_hold=("worst_cart_hold", "max")).reset_index())
    summ.to_csv(os.path.join(OUT, "tail_reference_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(pd.DataFrame(cert).to_string(index=False))
    return 0


# ------------------------------------------------------------- stage: closed loop
def stage_closedloop():
    ics = P2.load_initial_conditions()
    rows = []
    for key, sysdef in get_systems().items():
        x0 = ics[key]["eval"]
        out = {}
        for t in TAILS:
            _, xh, uh = det_traj(key, sysdef, t, x0)
            out[t] = (xh, uh)
        (xz, uz), (xh_, uh_) = out["zero"], out["hold_last"]
        du = np.linalg.norm(uh_ - uz, axis=0)
        nz = float(normalized_rmse(xz, sysdef, x0=x0))
        nh = float(normalized_rmse(xh_, sysdef, x0=x0))
        rows.append(dict(system=key,
                         nrmse_zero=nz, nrmse_hold=nh,
                         nrmse_abs_diff=abs(nh - nz),
                         nrmse_rel_diff=abs(nh - nz) / max(abs(nz), 1e-300),
                         max_state_dev_zero=float(np.max(np.abs(xz))),
                         max_state_dev_hold=float(np.max(np.abs(xh_))),
                         max_input_zero=float(np.max(np.abs(uz))),
                         max_input_hold=float(np.max(np.abs(uh_))),
                         mean_first_control_diff=float(np.mean(du)),
                         max_first_control_diff=float(np.max(du)),
                         input_range=float(sysdef.u_max - sysdef.u_min)))
        print(f"  {key}: nRMSE zero={nz:.6g} hold={nh:.6g} "
              f"rel={abs(nh-nz)/max(abs(nz),1e-300):.3e} mean|du0|={np.mean(du):.3e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tail_deterministic_closed_loop.csv"), index=False)
    df.to_csv(os.path.join(OUT, "tail_deterministic_summary.csv"), index=False)
    return 0


# -------------------------------------------------------------- stage: stochastic
def solve_step_tail(prob, sysdef, x, k, alg, params, warm):
    cost, _ = make_cost_tail(prob, sysdef, x, k)
    cn, lbn, ubn, unscale = normalize_cost(cost, prob.lb_seq, prob.ub_seq)
    counted = CountedCost(cn, fe_cap=int(params["fe_cap"]))
    wn = None
    if warm is not None:
        wn = (np.clip(warm, prob.lb_seq, prob.ub_seq) - prob.lb_seq) / \
             (prob.ub_seq - prob.lb_seq) * 2.0 - 1.0
    t0 = time.perf_counter()
    try:
        sol_n, _, _, _, _ = OPTIMIZERS[alg](
            counted, int(params["NP"]), int(params["maxIter"]), lbn, ubn,
            x0_warm=wn, tol=params["tol"], stagnant_limit=params["stagnant_limit"],
            max_evals=int(params["max_evals"]))
    except BudgetExhausted:
        sol_n = counted.best_x
    wall = time.perf_counter() - t0
    if counted.best_x is not None and float(counted.best_f) < float(cn(sol_n)):
        sol_n = counted.best_x
    sol = np.clip(unscale(sol_n), prob.lb_seq, prob.ub_seq)
    return sol, float(cost(sol)), counted.count, wall


def stage_stochastic(systems_sel, runs):
    ics = P2.load_initial_conditions()
    tuned = P2.load_tuned_np()["selected"]
    allsys = get_systems()
    rows = []
    t0 = time.perf_counter()
    for key in systems_sel:
        sysdef = allsys[key]
        mpc = P2.fixed_mpc_params(sysdef)
        x0 = ics[key]["eval"]
        for ti, tail in enumerate(TAILS):
            prob, xh_ref, uh_ref = det_traj(key, sysdef, tail, x0)
            ref_nrmse = float(normalized_rmse(xh_ref, sysdef, x0=x0))
            Nsim = xh_ref.shape[1] - 1
            for alg in P2.ALGS:
                NP = int(tuned[key][alg]["NP"])
                params = P2.strict_fe_params(NP, alg=alg)
                for t in range(runs):
                    # paired across conventions: seed does NOT depend on tail
                    seed = (SEED_BASE + 1_000_000 * P2.SYSTEM_INDEX[key]
                            + 1_000 * P2.ALG_INDEX[alg] + t)
                    np.random.seed(seed)
                    x = np.asarray(x0, float).copy()
                    xs = np.zeros((sysdef.nx, Nsim + 1)); xs[:, 0] = x
                    gaps, e_u, tms, nfes = [], [], [], []
                    last = np.zeros(len(prob.lb_seq))
                    for k in range(Nsim):
                        cost, f = make_cost_tail(prob, sysdef, x, k)
                        # reference of the SAME convention, at the state the
                        # optimizer itself reached
                        u_star, st, _ = osqp_reference(prob, f)
                        u_star = np.clip(u_star, prob.lb_seq, prob.ub_seq)
                        J_star = float(cost(u_star))
                        sol, J, nfe, wall = solve_step_tail(prob, sysdef, x, k, alg,
                                                            params, last)
                        gaps.append(abs(J - J_star) / max(1.0, abs(J_star)))
                        e_u.append(float(np.linalg.norm(sol[:sysdef.nu] -
                                                        u_star[:sysdef.nu])))
                        tms.append(wall * 1e3); nfes.append(nfe)
                        applied = np.clip(sol[:sysdef.nu], sysdef.u_min, sysdef.u_max)
                        x = sysdef.step(x, applied)
                        xs[:, k + 1] = x
                        last = sol.copy()
                        if prob.Nc > 1:
                            last[:-sysdef.nu] = last[sysdef.nu:]
                    rows.append(dict(
                        system=key, tail=tail, alg=alg, trial=t, seed=seed, NP=NP,
                        nrmse=float(normalized_rmse(xs, sysdef, x0=x0)),
                        reference_nrmse=ref_nrmse,
                        mean_norm_gap=float(np.mean(gaps)),
                        median_norm_gap=float(np.median(gaps)),
                        mean_first_input_error=float(np.mean(e_u)),
                        realized_fe_mean=float(np.mean(nfes)),
                        fe_exact=bool(all(n == P2.FE_BUDGET for n in nfes)),
                        time_ms_mean=float(np.mean(tms))))
                print(f"  {key:7s} {tail:10s} {alg:4s} done ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tail_stochastic_raw.csv"), index=False)
    summ = (df.groupby(["system", "tail", "alg"])
              .agg(n=("nrmse", "size"), nrmse_mean=("nrmse", "mean"),
                   nrmse_std=("nrmse", "std"),
                   reference_nrmse=("reference_nrmse", "first"),
                   mean_gap=("mean_norm_gap", "mean"),
                   median_gap=("median_norm_gap", "median"),
                   mean_first_input_error=("mean_first_input_error", "mean"),
                   fe_exact=("fe_exact", "all"),
                   time_ms=("time_ms_mean", "mean")).reset_index())
    summ["rank_nrmse"] = summ.groupby(["system", "tail"]).nrmse_mean.rank(method="min").astype(int)
    summ["rank_gap"] = summ.groupby(["system", "tail"]).mean_gap.rank(method="min").astype(int)
    summ.to_csv(os.path.join(OUT, "tail_stochastic_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["equivalence", "deterministic", "closedloop", "stochastic"])
    ap.add_argument("--systems", nargs="*", default=["flight", "uav"])
    ap.add_argument("--runs", type=int, default=RUNS)
    a = ap.parse_args()
    if a.stage == "equivalence":
        raise SystemExit(0 if stage_equivalence() else 1)
    if a.stage == "deterministic":
        raise SystemExit(stage_deterministic())
    if a.stage == "closedloop":
        raise SystemExit(stage_closedloop())
    raise SystemExit(stage_stochastic(a.systems, a.runs))


if __name__ == "__main__":
    main()
