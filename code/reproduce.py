#!/usr/bin/env python3
"""Single entry point for reproducing this study.

LEVEL 1 (default, minutes) -- analysis only:
    python code/reproduce.py --from-data
Reads the committed raw per-run CSVs under data/ and regenerates every summary
table, statistic and figure reported in the paper. It runs no optimizer, no MPC
simulation and no benchmark environment, so it needs no GMPB state files.

LEVEL 2 (hours to days) -- full experiment rerun:
    python code/reproduce.py --list-full-rerun
prints the ordered entry points that regenerate the raw data itself.

Verification:
    python code/reproduce.py --verify
recomputes the headline numbers straight from the raw CSVs and prints
PASS/FAIL per check. This is independent of the table/figure generators.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
DATA = os.path.join(PROJECT, "data")

# Analysis-only stages, in dependency order. Each reads committed CSVs and
# writes tables/figures. None of them calls an optimizer or a simulator.
LEVEL1 = [
    ("phase2a_analysis.py", "fixed-MPC primary tables, budget accounting, timing, "
                            "submitted-vs-revised protocol comparison"),
    ("effect_sizes_primary.py", "pairwise effect sizes, Cliff's delta, "
                                "Hodges-Lehmann shifts, practical-significance labels"),
    ("drift_performance_analysis.py", "drift-vs-difficulty associations, "
                                      "frozen sweep by stratum, warm-start mechanism"),
    ("phase3_artifact_prep.py", "SGO ablation and warm-start tables"),
    ("gmpb_analysis.py", "GMPB primary summaries, pairwise statistics, factor "
                         "analysis, primary figure"),
    ("gmpb_temporal_analysis.py", "matched temporal-transfer effects, cross-optimizer "
                                  "classification, behavioural diagnostics"),
    ("gmpb_mechanism_analysis.py", "SGO grouping diagnostic, diversity, "
                                   "2x2 temporal interaction"),
    ("summarize_sgo_benchmark.py", "original-SGO 13-function reproduction comparison"),
    ("phase3b_manuscript_artifacts.py", "all main-paper figures (fig1-fig7) and "
                                        "tables (tab2-tab8, supplementary tables)"),
]

FULL_RERUN = [
    ("MPC", [
        ("phase2_protocol.py", "freeze the fixed-MPC protocol and held-out initial conditions"),
        ("initial_condition_protocol.py", "define and freeze the evaluation initial conditions"),
        ("reference_verification.py", "certify the deterministic reference (OSQP vs multistart L-BFGS-B)"),
        ("reference_nc_sweep.py", "certify the reference across control horizons"),
        ("pendcart_penalty_certificate.py", "exact box-extrema certificate for the soft cart penalty"),
        ("strict_fe_verification.py", "exact function-evaluation budget enforcement (2000 cells)"),
        ("tune_optimizer_only.py", "optimizer-only tuning on held-out initial conditions"),
        ("primary_experiment.py", "fixed-MPC closed-loop primary experiment (20 seeds x 2 regimes)"),
        ("objective_drift.py", "sampled objective drift (feeds the Nc sweep)"),
        ("objective_drift_full.py", "drift at every consecutive transition (2400 transitions)"),
        ("drift_strata.py", "freeze the drift strata and the frozen-instance list"),
        ("frozen_sweep_v2.py", "215 frozen subproblems x 4 optimizers x 3 starts x 20 trials"),
        ("budget_sensitivity_v2.py", "budget sweep 250-5000 FE on 9 selected instances"),
        ("multi_ic_experiment.py", "5 systems x 5 initial conditions x 4 optimizers x 5 seeds"),
        ("dimension_sweep.py", "decision-dimension sweep over the certified cells"),
        ("perturbation_experiment.py", "bounded noise / disturbance / reference-event study"),
        ("input_error_sensitivity.py", "first-input error vs closed-loop tracking"),
        ("tail_convention_check.py", "zero-tail vs hold-last prediction convention"),
        ("tail_convention_analysis.py", "paired tail-convention effects and ordering"),
        ("sgo_ablation.py", "SGO operator ablation on the MPC instances"),
    ]),
    ("GMPB", [
        ("gmpb_generate.sh <cache> <lo> <hi>", "GNU Octave: generate the official GMPB state files"),
        ("gmpb_equivalence.py", "gate: Python port vs the official Octave implementation"),
        ("gmpb_sgo_fidelity.py", "gate: dynamic SGO driver vs the static SGO implementation"),
        ("gmpb_launch.sh <cache> <out>", "primary 12 cases x 4 optimizers x 31 runs, plus SGO temporal modes"),
        ("gmpb_gates.py", "validation gates A-J on the produced runs"),
        ("gmpb_temporal_comparators.py", "matched temporal transfer: 4 optimizers x 4 cases x 3 modes x 31 runs"),
        ("gmpb_temporal_validation.py", "validation gate for the comparator study"),
        ("gmpb_mechanism_equivalence.py", "gate: random-grouping ablation must equal the validated SGO"),
        ("gmpb_mechanism_run.py", "grouping diagnostic: 12 cases + the 2x2 temporal interaction"),
    ]),
    ("Original-SGO reproduction", [
        ("sgo_benchmark_verification.py", "13 original benchmark functions, 100 dimensions, "
                                          "150000 evaluations, 100 runs, both branch conventions"),
    ]),
]


def run_stage(script, describe, cont):
    path = os.path.join(ROOT, script)
    print(f"\n=== {script} :: {describe} ===", flush=True)
    r = subprocess.run([sys.executable, path], cwd=PROJECT)
    if r.returncode != 0:
        print(f"*** {script} FAILED (exit {r.returncode})", file=sys.stderr)
        if not cont:
            sys.exit(r.returncode)
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-data", action="store_true",
                    help="Level 1: regenerate all tables and figures from the committed raw data")
    ap.add_argument("--verify", action="store_true",
                    help="recompute the headline reported numbers directly from the raw CSVs")
    ap.add_argument("--list-full-rerun", action="store_true",
                    help="Level 2: print the ordered entry points that regenerate the raw data")
    ap.add_argument("--only", metavar="SCRIPT", help="run a single Level-1 stage")
    ap.add_argument("--continue-on-error", action="store_true")
    a = ap.parse_args()

    if not (a.from_data or a.verify or a.list_full_rerun or a.only):
        ap.print_help()
        return

    if a.list_full_rerun:
        print("LEVEL 2 - full experiment rerun (expensive; see README for cost)\n")
        for group, items in FULL_RERUN:
            print(f"[{group}]")
            for s, d in items:
                pre = "bash code/" if s.endswith((".sh", ".sh <cache> <lo> <hi>")) or ".sh " in s else "python code/"
                print(f"  {pre}{s}\n      {d}")
            print()
        return

    if a.only:
        run_stage(a.only, "single stage", a.continue_on_error)

    if a.from_data:
        if not os.path.isdir(DATA):
            sys.exit(f"data directory not found: {DATA}")
        ok = [run_stage(s, d, a.continue_on_error) for s, d in LEVEL1]
        print(f"\nLevel-1 analysis reproduction: {sum(ok)}/{len(ok)} stages succeeded")
        print(f"figures -> data/figures_tables/figures, data/figures, data/gmpb*/figures")
        print(f"tables  -> data/figures_tables/tables, data/tables, data/gmpb*/tables")
        if not all(ok):
            sys.exit(1)

    if a.verify:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "verify_claims.py")], cwd=PROJECT)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
