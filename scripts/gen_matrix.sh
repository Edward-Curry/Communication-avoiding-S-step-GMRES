#!/bin/bash
#SBATCH -J gmres_gen
#SBATCH -o gmres_gen_%j.out
#SBATCH -e gmres_gen_%j.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00

# File: scripts/gen_matrix.sh
# Last updated by Edward Curry: 2026-08-23
# Generates one scaling matrix on Seagull.
# Inputs: output path, n1, optional diagonal boost, advection, and scatter block.
# Output: Matrix Market file at the requested path.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1

output="${1:?usage: gen_matrix.sh <output.mtx> <n1> [diag_boost] [advection] [scatter_block]}"
n1="${2:?missing n1}"
diag_boost="${3:-0.01}"
advection="${4:-0.0}"
scatter="${5:-1}"

echo "--- generating $output ---"
echo "n1=$n1  diag_boost=$diag_boost  advection=$advection  scatter_block=$scatter"

if [[ -f "$output" ]]; then
    echo "$output already exists, skipping generation"
    exit 0
fi

python3 scripts/generate_scaling_matrix.py \
    --size "$n1" \
    --diag-boost "$diag_boost" \
    --advection "$advection" \
    --scatter-block "$scatter" \
    --output "$output"

echo "done: $(ls -lh "$output" | awk '{print $5}')"
