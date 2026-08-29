#!/usr/bin/env bash
# Generate official GMPB benchmark states with the unmodified EDOLAB Octave code.
#   code/gmpb_generate.sh <cache_dir> <seed_lo> <seed_hi> [nproc] [cases...]
# Requires GNU Octave on PATH (or set OCTAVE_ACTIVATE to a script that puts it there).
set -euo pipefail
CACHE="$1"; LO="$2"; HI="$3"; NPROC="${4:-8}"; shift 4 || true
CASES=("$@"); [ ${#CASES[@]} -eq 0 ] && CASES=(F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12)
W="$(cd "$(dirname "$0")/.." && pwd)/data/gmpb/octave"
mkdir -p "$CACHE"
# Optional: activate an environment that provides Octave.
# Set OCTAVE_ACTIVATE to a shell script to source, e.g.
#   OCTAVE_ACTIVATE=~/miniconda3/etc/profile.d/conda.sh CONDA_ENV=octave_gmpb ...
if [ -n "${OCTAVE_ACTIVATE:-}" ]; then
  # shellcheck disable=SC1090
  source "$OCTAVE_ACTIVATE"
  [ -n "${CONDA_ENV:-}" ] && conda activate "$CONDA_ENV"
fi
command -v octave >/dev/null || { echo "error: octave not found on PATH" >&2; exit 1; }
cd "$W"
for c in "${CASES[@]}"; do
  for s in $(seq "$LO" "$HI"); do
    if [ ! -f "$CACHE/${c}_seed${s}_state.mat" ]; then
      echo "$c $s"
    fi
  done
done | xargs -P "$NPROC" -n 2 bash -c 'octave --no-gui -q gmpb_dump.m "$0" "$1" '"$CACHE"' >/dev/null 2>&1 || echo "FAILED $0 $1"'
echo "generation done: $(ls "$CACHE"/*_state.mat 2>/dev/null | wc -l) states"
