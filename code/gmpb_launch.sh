#!/usr/bin/env bash
# Launch the frozen GMPB primary + SGO temporal study.
#   code/gmpb_launch.sh <state_cache_dir> <output_dir> [nproc]
# <state_cache_dir> must already hold the Octave-generated GMPB state files
# (see code/gmpb_generate.sh).  Run from any directory.
set -euo pipefail
CACHE="$1"; OUT="$2"; NP="${3:-16}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/.."
echo "=== PRIMARY: 12 cases x 4 optimizers x 31 runs ==="
python code/gmpb_run.py primary "$CACHE" "$OUT" ALL 31 "$NP"
echo "=== TEMPORAL: 4 cases x 3 modes x 31 runs (SGO) ==="
python code/gmpb_run.py temporal "$CACHE" "$OUT" ALL 31 "$NP"
echo "=== DONE ==="
