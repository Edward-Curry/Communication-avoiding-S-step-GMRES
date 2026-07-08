#!/bin/bash
#SBATCH -J gmres_chol
#SBATCH -o gmres_chol_%j.out
#SBATCH -e gmres_chol_%j.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/setup_seagull_mpi.sh
export SLURM_EXPORT_ENV=ALL

matrix="${1:-data/matrices/parabolic_fem.mtx}"
config_file="include/common/config.hpp"

if [[ ! -f "$matrix" ]]; then
    echo "Matrix file not found: $matrix"
    exit 1
fi

if [[ ! -f "$config_file" ]]; then
    echo "Config file not found: $config_file"
    exit 1
fi

s_step="$(
    awk -F'[=;]' '/Index s_step/ {
        gsub(/[[:space:]]/, "", $2);
        print $2;
        exit
    }' "$config_file"
)"

restart_blocks="$(
    awk -F'[=;]' '/Index restart_blocks/ {
        gsub(/[[:space:]]/, "", $2);
        print $2;
        exit
    }' "$config_file"
)"

if [[ -z "$s_step" || -z "$restart_blocks" ]]; then
    echo "Could not read s_step or restart_blocks from $config_file"
    exit 1
fi

nodes="${SLURM_JOB_NUM_NODES:-1}"
ranks="${SLURM_NTASKS:-${SLURM_NPROCS:-16}}"
tasks_per_node="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-unknown}}"

matrix_name="$(basename "${matrix%.*}")"
output_directory="data/outputs/${matrix_name}/s${s_step}_blocks${restart_blocks}"

mkdir -p "$output_directory"

echo "--- MPI CA-GMRES BCGS2-CholQR: $nodes nodes, $ranks ranks ---"
echo "Tasks per node: $tasks_per_node"
echo "Matrix: $matrix"
echo "s_step: $s_step"
echo "restart_blocks: $restart_blocks"
echo "Output directory: $output_directory"

mpirun --prefix "$MPI_ROOT" \
    -np "$ranks" \
    --map-by slot \
    --bind-to core \
    ./build/run_gmres_ca_mpi_experiment \
    "$matrix" \
    "$output_directory"

echo "Results saved under $output_directory"
