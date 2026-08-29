#!/usr/bin/env python3
"""Phase-2A step 4: exact certificate for the pendulum-cart soft penalty.

The pendulum-cart MPC cost adds

    1e5 * sum_i max(|x_cart,i| - 10, 0)^2

over the prediction horizon. Phase 1 observed that random input samples never
activated it, but sampling cannot prove inactivity. This script proves (or
disproves) it exactly.

The predicted cart position at prediction step i is AFFINE in the decision
vector U:

    x_cart,i(U) = a_i + c_i' U,      a_i = (S_i Phi x_c),  c_i' = (S_i Gamma)

so over the box lb <= U <= ub the exact extrema are attained at a vertex and are
available in closed form:

    max_i = a_i + sum_j max(c_ij*lb_j, c_ij*ub_j)
    min_i = a_i + sum_j min(c_ij*lb_j, c_ij*ub_j)

No sampling, no optimisation. The certificate answers, for each subproblem
instance, whether |x_cart| can reach the 10 m limit ANYWHERE in the admissible
input box.

States are taken from every step of the deterministic closed-loop trajectory AND
from every step of the closed-loop trajectories actually produced by SGO, GWO,
PSO and WOA, so the certificate covers the states the experiments really visit,
not only the nominal ones.

Outputs:
    data/pendcart_penalty_box_certificate.csv
    data/pendcart_penalty_certificate.json

Usage:
    python pendcart_penalty_certificate.py [--nc 1 3]
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
OUT = os.path.join(PROJECT, "data")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

from system_defs import get_systems                                    # noqa: E402
from sim_common import simulate, build_problem, get_params, load_tuned_params  # noqa: E402

ALGS = ["SGO", "GWO", "PSO", "WOA"]


def box_extrema(problem, x_c):
    """Exact per-prediction-step extrema of the predicted cart position."""
    P = problem.P
    S = np.kron(np.eye(P), np.array([[1, 0, 0, 0]], dtype=float))
    a = (S @ (problem.Phi @ x_c)).ravel()          # (P,)
    C = S @ problem.Gamma                          # (P, Nc*nu)
    lo = np.asarray(problem.lb_seq, dtype=float)
    hi = np.asarray(problem.ub_seq, dtype=float)
    # Vertex-attained extrema of an affine function over a box.
    pos = np.maximum(C * lo, C * hi).sum(axis=1)
    neg = np.minimum(C * lo, C * hi).sum(axis=1)
    return a + pos, a + neg


def certify(nc_list):
    sysdef = get_systems()["pendcart"]
    limit = float(sysdef.cart_constraint)
    tuned = load_tuned_params()

    # Every state visited by the deterministic controller and by each optimizer.
    trajectories = {}
    det = simulate("pendcart", "QP", params={"Nc": 1, "Q_scale": 1.0}, seed=11, verbose=False)
    trajectories["deterministic"] = det["x_hist"]
    for alg in ALGS:
        p = get_params("pendcart", alg, tuned=tuned, mode="tuned")
        out = simulate("pendcart", alg, params=p, seed=42, verbose=False)
        trajectories[alg] = out["x_hist"]

    rows = []
    for nc in nc_list:
        problem = build_problem(sysdef, nc, 1.0, sysdef.P)
        for src, xh in trajectories.items():
            for k in range(xh.shape[1]):
                x_c = sysdef.cost_state(xh[:, k])
                hi_i, lo_i = box_extrema(problem, x_c)
                worst = float(max(np.max(np.abs(hi_i)), np.max(np.abs(lo_i))))
                rows.append(dict(system="pendcart", Nc=nc, source=src, k=int(k),
                                 max_cart_over_box=float(np.max(hi_i)),
                                 min_cart_over_box=float(np.min(lo_i)),
                                 worst_abs_cart_over_box=worst,
                                 limit=limit,
                                 penalty_can_activate=bool(worst > limit),
                                 margin=limit - worst))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", type=int, nargs="*", default=[1, 2, 3])
    args = ap.parse_args()
    t0 = time.perf_counter()
    df = certify(args.nc)
    df.to_csv(os.path.join(OUT, "pendcart_penalty_box_certificate.csv"), index=False)

    verdict = {
        "limit_m": float(df.limit.iloc[0]),
        "instances_checked": int(len(df)),
        "Nc_values": sorted(df.Nc.unique().tolist()),
        "state_sources": sorted(df.source.unique().tolist()),
        "method": "exact vertex extrema of an affine function over the input box "
                  "(no sampling, no optimisation)",
        "worst_abs_cart_over_box": float(df.worst_abs_cart_over_box.max()),
        "smallest_margin_m": float(df.margin.min()),
        "any_instance_can_activate_penalty": bool(df.penalty_can_activate.any()),
        "n_instances_that_can_activate": int(df.penalty_can_activate.sum()),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    verdict["verdict"] = ("PENALTY GLOBALLY INACTIVE over the admissible input box for all "
                          "checked instances; the pendulum-cart subproblems are exactly "
                          "bound-constrained QPs"
                          if not verdict["any_instance_can_activate_penalty"] else
                          "PENALTY CAN ACTIVATE for at least one instance; pendulum-cart is "
                          "NOT a pure QP there and OSQP must not be used for those instances")
    with open(os.path.join(OUT, "pendcart_penalty_certificate.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(df.groupby(["Nc", "source"]).agg(
        worst_abs=("worst_abs_cart_over_box", "max"),
        min_margin=("margin", "min"),
        can_activate=("penalty_can_activate", "any")).to_string())
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
