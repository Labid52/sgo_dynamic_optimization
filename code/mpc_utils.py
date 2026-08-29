"""
MPC utility functions shared by all five system simulations.

Provides:
  build_prediction_matrices  – compute Phi, Gamma for linear MPC
  make_mpc_cost              – returns scalar QP cost function
  compute_metrics            – rise/settle/overshoot/SSE/RMSE
  save_csv                   – write state + input history to CSV
  plot_states                – states vs time (EPS)
  plot_inputs                – inputs vs time (EPS)
  plot_opt_results           – optimiser convergence figures (EPS)
  save_metrics_txt           – write metrics table to .txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# MPC core
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_matrices(Ad, Bd, P, Nc):
    """
    Build MPC prediction matrices Phi (P*nx × nx) and Gamma (P*nx × Nc*nu).

    X_pred = Phi * x + Gamma * U
    where U = [u0; u1; ...; u_{Nc-1}].
    Important: this matches the MATLAB archive and assumes future inputs
    beyond Nc are zero in the prediction model, not held constant.
    """
    nx = Ad.shape[0]
    nu = Bd.shape[1]

    Phi   = np.zeros((P * nx, nx))
    Gamma = np.zeros((P * nx, Nc * nu))

    Ad_pow = np.eye(nx)
    for i in range(P):
        Ad_pow = Ad_pow @ Ad
        Phi[i*nx:(i+1)*nx, :] = Ad_pow

        for j in range(min(i + 1, Nc)):
            col_start = j * nu
            col_end   = col_start + nu
            row_start = i * nx
            row_end   = row_start + nx
            # Gamma[i, j] = Ad^(i-j) * Bd
            power = i - j
            AdP   = np.linalg.matrix_power(Ad, power)
            Gamma[row_start:row_end, col_start:col_end] = AdP @ Bd

    return Phi, Gamma


def build_qp_matrices(Phi, Gamma, Q_block, R_block):
    """
    H = 2*(Gamma'*Q_block*Gamma + R_block)
    (constant across MPC steps for fixed-point regulation)
    """
    H = 2.0 * (Gamma.T @ Q_block @ Gamma + R_block)
    H = 0.5 * (H + H.T)          # numerical symmetry
    return H


def mpc_cost_fn(H, f):
    """
    Return a scalar cost callable: J(U) = 0.5*U'*H*U + f'*U
    """
    def cost(U):
        U = np.asarray(U, dtype=float).ravel()
        return 0.5 * (U @ H @ U) + f @ U
    return cost


def compute_f(Phi, Gamma, Q_block, x, x_ref_horizon):
    """
    f = 2 * Gamma' * Q_block * (Phi*x - x_ref_horizon)
    """
    return 2.0 * Gamma.T @ Q_block @ (Phi @ x - x_ref_horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(x_hist, x_ref_val, time_vec, state_idx=0):
    """
    Compute standard step-response metrics for one state.

    Parameters
    ----------
    x_hist    : (nx, Nsim+1) array – full state history
    x_ref_val : float – scalar reference for this state
    time_vec  : (Nsim+1,) array
    state_idx : which state to analyse

    Returns
    -------
    dict with: rise_time, settling_time, overshoot_pct, ss_error, rmse
    """
    y    = x_hist[state_idx]
    r    = x_ref_val
    y0   = y[0]
    span = abs(r - y0)

    # Rise time: 10 % → 90 %
    if span < 1e-12:
        rise_time = 0.0
    else:
        lvl10 = y0 + 0.10 * (r - y0)
        lvl90 = y0 + 0.90 * (r - y0)
        if r >= y0:
            i10 = np.where(y >= lvl10)[0]
            i90 = np.where(y >= lvl90)[0]
        else:
            i10 = np.where(y <= lvl10)[0]
            i90 = np.where(y <= lvl90)[0]
        rise_time = (time_vec[i90[0]] - time_vec[i10[0]]
                     if (len(i10) and len(i90)) else float("nan"))

    # Settling time: last time outside ±2 % band
    tol = 0.02 * span if span > 1e-12 else 0.02 * abs(r)
    outside = np.where(np.abs(y - r) > tol)[0]
    settling_time = (time_vec[outside[-1] + 1]
                     if len(outside) and outside[-1] + 1 < len(time_vec)
                     else time_vec[-1])

    # Overshoot
    if r > y0:
        peak = y.max()
        os_pct = max(0.0, (peak - r) / span * 100) if span > 1e-12 else 0.0
    elif r < y0:
        peak = y.min()
        os_pct = max(0.0, (r - peak) / span * 100) if span > 1e-12 else 0.0
    else:
        os_pct = 0.0

    ss_error = abs(y[-1] - r)
    rmse     = float(np.sqrt(np.mean((y - r) ** 2)))

    return dict(rise_time=rise_time, settling_time=settling_time,
                overshoot_pct=os_pct, ss_error=ss_error, rmse=rmse)


# ─────────────────────────────────────────────────────────────────────────────
# File / figure output
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(out_dir, system_name, time_vec,
             x_hist_sgo, u_hist_sgo,
             x_hist_gwo, u_hist_gwo,
             state_names, input_names):
    """Save combined SGO+GWO state and input histories to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{system_name}_data.csv")

    cols = {"time": time_vec}
    Nsim = len(time_vec) - 1

    for si, sn in enumerate(state_names):
        cols[f"sgo_{sn}"] = x_hist_sgo[si]
        cols[f"gwo_{sn}"] = x_hist_gwo[si]
    for ui, un in enumerate(input_names):
        cols[f"sgo_{un}"] = np.append(u_hist_sgo[ui], np.nan)
        cols[f"gwo_{un}"] = np.append(u_hist_gwo[ui], np.nan)

    pd.DataFrame(cols).to_csv(path, index=False, float_format="%.6f")
    print(f"  [CSV] saved → {path}")


def _apply_style():
    """Consistent matplotlib style."""
    plt.rcParams.update({
        "font.family":  "DejaVu Sans",
        "font.size":    10,
        "axes.grid":    True,
        "grid.alpha":   0.35,
        "lines.linewidth": 1.6,
        "figure.dpi":  150,
    })


def plot_states(out_dir, system_name, time_vec,
                x_hist_sgo, x_hist_gwo,
                state_names, x_refs=None, time_unit="s"):
    """States vs time – SGO and GWO overlaid, one subplot per state."""
    _apply_style()
    nx   = len(state_names)
    fig, axes = plt.subplots(nx, 1, figsize=(8, 2.2 * nx), sharex=True)
    if nx == 1:
        axes = [axes]

    colors = {"SGO": "#1f77b4", "GWO": "#d62728"}

    for si, (ax, sn) in enumerate(zip(axes, state_names)):
        ax.plot(time_vec, x_hist_sgo[si], color=colors["SGO"],
                label="SGO", linewidth=1.8)
        ax.plot(time_vec, x_hist_gwo[si], color=colors["GWO"],
                label="GWO", linewidth=1.8, linestyle="--")
        if x_refs is not None and x_refs[si] is not None:
            ax.axhline(x_refs[si], color="k", linestyle=":", linewidth=1.2,
                       label="ref")
        ax.set_ylabel(sn, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=10)
    fig.suptitle(f"{system_name} – States vs Time", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "states_time.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  [EPS] saved → {path}")


def plot_inputs(out_dir, system_name, time_vec,
                u_hist_sgo, u_hist_gwo,
                input_names, time_unit="s"):
    """Control inputs vs time."""
    _apply_style()
    nu   = len(input_names)
    fig, axes = plt.subplots(nu, 1, figsize=(8, 2.0 * nu), sharex=True)
    if nu == 1:
        axes = [axes]

    colors = {"SGO": "#1f77b4", "GWO": "#d62728"}
    t_u   = time_vec[:-1]

    for ui, (ax, un) in enumerate(zip(axes, input_names)):
        ax.step(t_u, u_hist_sgo[ui], where="post",
                color=colors["SGO"], label="SGO")
        ax.step(t_u, u_hist_gwo[ui], where="post",
                color=colors["GWO"], label="GWO", linestyle="--")
        ax.set_ylabel(un, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=10)
    fig.suptitle(f"{system_name} – Control Inputs", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "input_time.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  [EPS] saved → {path}")


def plot_opt_results(out_dir, system_name,
                     sgo_cost_per_step, gwo_cost_per_step,
                     sgo_iters_per_step, gwo_iters_per_step,
                     sgo_last_conv, gwo_last_conv):
    """
    Three optimisation-result figures:
      1. Final optimiser cost per MPC step
      2. Iterations used per MPC step
      3. Convergence curve from the last MPC step
    """
    _apply_style()
    colors = {"SGO": "#1f77b4", "GWO": "#d62728"}
    steps  = np.arange(1, len(sgo_cost_per_step) + 1)

    # ── Fig 1: cost per step ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(steps, sgo_cost_per_step, color=colors["SGO"], label="SGO")
    ax.plot(steps, gwo_cost_per_step, color=colors["GWO"],
            label="GWO", linestyle="--")
    ax.set_xlabel("MPC Step")
    ax.set_ylabel("Final Optimiser Cost")
    ax.set_title(f"{system_name} – Optimiser Cost per MPC Step")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "opt_cost_per_step.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  [EPS] saved → {path}")

    # ── Fig 2: iterations per step ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(steps, sgo_iters_per_step, color=colors["SGO"], label="SGO", alpha=0.8)
    ax.plot(steps, gwo_iters_per_step, color=colors["GWO"],
            label="GWO", linestyle="--", alpha=0.8)
    ax.set_xlabel("MPC Step")
    ax.set_ylabel("Iterations Used")
    ax.set_title(f"{system_name} – Iterations per MPC Step")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "opt_iters_per_step.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  [EPS] saved → {path}")

    # ── Fig 3: last-step convergence curve ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(sgo_last_conv, color=colors["SGO"], label="SGO")
    ax.plot(gwo_last_conv, color=colors["GWO"], label="GWO", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost")
    ax.set_title(f"{system_name} – Convergence (Last MPC Step)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "opt_convergence_last.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  [EPS] saved → {path}")


def save_metrics_txt(out_dir, system_name,
                     metrics_sgo, metrics_gwo,
                     timing_sgo, timing_gwo,
                     extra_sgo=None, extra_gwo=None):
    """
    Write a formatted comparative metrics table to metrics.txt.

    metrics_sgo / metrics_gwo : list of dicts (one per tracked state)
    timing_sgo / timing_gwo   : dict with avg_time_ms, avg_iters, total_sim_s
    extra_sgo / extra_gwo     : optional dict of extra scalar metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.txt")

    def pct_improve(sgo_val, gwo_val, lower_better=True):
        if gwo_val == 0:
            return "N/A"
        if lower_better:
            pct = (gwo_val - sgo_val) / abs(gwo_val) * 100
        else:
            pct = (sgo_val - gwo_val) / abs(gwo_val) * 100
        sign = "+" if pct > 0 else ""
        tag  = "SGO better" if pct > 0 else "GWO better" if pct < 0 else "equal"
        return f"{sign}{pct:.1f}% ({tag})"

    lines = []
    lines.append("=" * 72)
    lines.append(f"  Comparative Metrics – {system_name}")
    lines.append("=" * 72)

    for k, (ms, mg) in enumerate(zip(metrics_sgo, metrics_gwo)):
        state_id = f"State {k}"
        lines.append(f"\n  ── {state_id} ──")
        rows = [
            ("Rise Time",      ms["rise_time"],       mg["rise_time"],
             True,  "s/h"),
            ("Settling Time",  ms["settling_time"],   mg["settling_time"],
             True,  "s/h"),
            ("Overshoot",      ms["overshoot_pct"],   mg["overshoot_pct"],
             True,  "%"),
            ("SS Error",       ms["ss_error"],        mg["ss_error"],
             True,  "units"),
            ("RMSE",           ms["rmse"],            mg["rmse"],
             True,  "units"),
        ]
        lines.append(f"  {'Metric':<20} {'SGO':>14} {'GWO':>14}   {'Δ (SGO vs GWO)':}")
        lines.append("  " + "-" * 68)
        for row in rows:
            label, sv, gv, lb2, unit = row
            sv_s = f"{sv:.5f}" if sv is not None and not (isinstance(sv, float) and np.isnan(sv)) else "NaN"
            gv_s = f"{gv:.5f}" if gv is not None and not (isinstance(gv, float) and np.isnan(gv)) else "NaN"
            pi   = pct_improve(sv, gv, lb2) if (sv_s != "NaN" and gv_s != "NaN") else "N/A"
            lines.append(f"  {label:<20} {sv_s:>14} {gv_s:>14}   {pi}")

    lines.append("\n  ── Optimiser Performance ──")
    lines.append(f"  {'Metric':<28} {'SGO':>12} {'GWO':>12}   {'Δ (SGO vs GWO)':}")
    lines.append("  " + "-" * 68)
    tm_rows = [
        ("Avg opt time (ms)", timing_sgo["avg_time_ms"], timing_gwo["avg_time_ms"],   True),
        ("Avg iterations",    timing_sgo["avg_iters"],   timing_gwo["avg_iters"],     True),
        ("Total sim time (s)",timing_sgo["total_sim_s"], timing_gwo["total_sim_s"],   True),
    ]
    for row in tm_rows:
        label, sv, gv, lb2 = row
        lines.append(f"  {label:<28} {sv:>12.4f} {gv:>12.4f}   "
                     f"{pct_improve(sv, gv, lb2)}")

    if extra_sgo and extra_gwo:
        lines.append("\n  ── Additional Metrics ──")
        for key in extra_sgo:
            sv = extra_sgo[key]
            gv = extra_gwo.get(key, float("nan"))
            lines.append(f"  {key:<28} {sv!s:>12} {gv!s:>12}")

    lines.append("\n" + "=" * 72)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [TXT] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Input normalisation (added for convergence improvement)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_cost(cost_fn, lb, ub):
    """
    Wrap cost_fn so the optimizer searches in the unit box [-1, 1]^d
    instead of the original [lb, ub]^d.

    Normalisation: u_norm = (u - centre) / half_range  =>  u_norm in [-1, 1]
    Inverse:       u      = centre + half_range * u_norm

    This improves conditioning of H (whose eigenvalues are affected by the
    input scale), which directly helps population-based optimisers converge.

    Returns
    -------
    cost_norm   : callable on [-1,1]^d
    lb_norm     : -1 * ones(d)
    ub_norm     : +1 * ones(d)
    unscale     : callable  u_norm -> u  (for recovering the real solution)
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    centre     = (lb + ub) / 2.0
    half_range = (ub - lb) / 2.0

    def cost_norm(u_norm):
        u = centre + half_range * np.asarray(u_norm, dtype=float)
        return cost_fn(u)

    def unscale(u_norm):
        return centre + half_range * np.asarray(u_norm, dtype=float)

    lb_norm = np.full_like(lb, -1.0)
    ub_norm = np.full_like(ub,  1.0)
    return cost_norm, lb_norm, ub_norm, unscale


# ── Multi-algorithm plotting (SGO / GWO / PSO / WOA) ─────────────────────────
_ALG_STYLES = {
    "SGO": dict(color="#1f77b4", ls="-",    lw=1.8),
    "GWO": dict(color="#d62728", ls="--",   lw=1.8),
    "PSO": dict(color="#2ca02c", ls=":",    lw=2.0),
    "WOA": dict(color="#ff7f0e", ls="-.",   lw=1.8),
    "QP":  dict(color="#000000", ls="-",    lw=2.0),
}

def plot_states_multi(out_dir, system_name, time_vec,
                      results,        # dict {alg_name: x_hist (nx × Nsim+1)}
                      state_names, x_refs=None, time_unit="s"):
    """
    States vs time for an arbitrary number of algorithms.
    results: dict  alg_name → (nx, Nsim+1) array
    """
    _apply_style()
    nx   = len(state_names)
    fig, axes = plt.subplots(nx, 1, figsize=(8, 2.2*nx), sharex=True)
    if nx == 1: axes = [axes]
    for si, (ax, sn) in enumerate(zip(axes, state_names)):
        for alg, xh in results.items():
            st = _ALG_STYLES.get(alg, dict(color="k", ls="-", lw=1.5))
            ax.plot(time_vec, xh[si], label=alg, **st)
        if x_refs is not None and x_refs[si] is not None:
            ax.axhline(x_refs[si], color="k", ls=":", lw=1.2, label="ref")
        ax.set_ylabel(sn, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=10)
    fig.suptitle(f"{system_name} – States vs Time", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "states_time.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  [EPS] saved → {path}")


def plot_inputs_multi(out_dir, system_name, time_vec,
                      results,        # dict {alg_name: u_hist (nu × Nsim)}
                      input_names, time_unit="s"):
    """Control inputs vs time for multiple algorithms."""
    _apply_style()
    nu   = len(input_names)
    fig, axes = plt.subplots(nu, 1, figsize=(8, 2.0*nu), sharex=True)
    if nu == 1: axes = [axes]
    t_u = time_vec[:-1]
    for ui, (ax, un) in enumerate(zip(axes, input_names)):
        for alg, uh in results.items():
            st = _ALG_STYLES.get(alg, dict(color="k", ls="-", lw=1.5))
            ax.step(t_u, uh[ui], where="post", label=alg, **st)
        ax.set_ylabel(un, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=10)
    fig.suptitle(f"{system_name} – Control Inputs", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "input_time.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  [EPS] saved → {path}")


def plot_opt_results_multi(out_dir, system_name,
                           cost_results,  # dict {alg: cost_per_step array}
                           iter_results,  # dict {alg: iters_per_step array}
                           conv_results): # dict {alg: last convergence curve}
    """Optimisation result figures for multiple algorithms."""
    _apply_style()
    steps = np.arange(1, len(list(cost_results.values())[0]) + 1)

    for data, ylabel, title, fname in [
        (cost_results, "Final Optimiser Cost", "Optimiser Cost per MPC Step",
         "opt_cost_per_step.eps"),
        (iter_results, "Iterations Used",      "Iterations per MPC Step",
         "opt_iters_per_step.eps"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for alg, vals in data.items():
            st = _ALG_STYLES.get(alg, dict(color="k", ls="-", lw=1.5))
            ax.plot(steps, vals, label=alg, **st)
        ax.set_xlabel("MPC Step"); ax.set_ylabel(ylabel)
        ax.set_title(f"{system_name} – {title}"); ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, format="eps", bbox_inches="tight")
        plt.close(fig)
        print(f"  [EPS] saved → {path}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for alg, conv in conv_results.items():
        if conv is None: continue
        st = _ALG_STYLES.get(alg, dict(color="k", ls="-", lw=1.5))
        ax.plot(conv, label=alg, **st)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best Cost")
    ax.set_title(f"{system_name} – Convergence (Last MPC Step)"); ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "opt_convergence_last.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  [EPS] saved → {path}")


def save_csv_multi(out_dir, system_name, time_vec,
                   x_results, u_results, state_names, input_names):
    """Save state + input histories for all algorithms to one CSV."""
    os.makedirs(out_dir, exist_ok=True)
    cols = {"time": time_vec}
    for alg, xh in x_results.items():
        for si, sn in enumerate(state_names):
            cols[f"{alg.lower()}_{sn}"] = xh[si]
    for alg, uh in u_results.items():
        for ui, un in enumerate(input_names):
            cols[f"{alg.lower()}_{un}"] = np.append(uh[ui], np.nan)
    pd.DataFrame(cols).to_csv(
        os.path.join(out_dir, f"{system_name}_data.csv"),
        index=False, float_format="%.6f")
    print(f"  [CSV] saved → {os.path.join(out_dir, system_name + '_data.csv')}")


def save_metrics_txt_multi(out_dir, system_name, metrics_dict, timing_dict):
    """
    metrics_dict: {alg: list of metric dicts (one per state)}
    timing_dict:  {alg: timing dict}
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.txt")
    lines = ["=" * 72, f"  Comparative Metrics – {system_name}", "=" * 72]
    algs = list(metrics_dict.keys())

    # Per-state metrics
    for si in range(len(list(metrics_dict.values())[0])):
        lines.append(f"\n  ── State {si} ──")
        hdr = f"  {'Metric':<22}" + "".join(f"{a:>14}" for a in algs)
        lines.append(hdr); lines.append("  " + "-" * (22 + 14*len(algs)))
        for key, label in [("rise_time","Rise Time"),
                            ("settling_time","Settling Time"),
                            ("overshoot_pct","Overshoot (%)"),
                            ("ss_error","SS Error"),
                            ("rmse","RMSE")]:
            row = f"  {label:<22}"
            for a in algs:
                v = metrics_dict[a][si][key]
                row += f"{v:>14.5f}" if not (isinstance(v,float) and np.isnan(v)) else f"{'NaN':>14}"
            lines.append(row)

    # Timing
    lines.append("\n  ── Optimiser Performance ──")
    hdr2 = f"  {'Metric':<28}" + "".join(f"{a:>12}" for a in algs)
    lines.append(hdr2); lines.append("  " + "-" * (28 + 12*len(algs)))
    for key, label in [("avg_time_ms","Avg opt time (ms)"),
                        ("avg_iters","Avg iterations"),
                        ("avg_nfe","Avg function evals")]:
        row = f"  {label:<28}"
        for a in algs:
            row += f"{timing_dict[a].get(key, float('nan')):>12.4f}"
        lines.append(row)

    lines.append("\n" + "=" * 72)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [TXT] saved → {path}")
