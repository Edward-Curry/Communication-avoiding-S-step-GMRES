#!/bin/bash
#SBATCH -J gmres_comm
#SBATCH -o gmres_comm_%j.out
#SBATCH -e gmres_comm_%j.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --nice=10000

# File: scripts/run_comm.sh
# Last updated by Edward Curry: 2026-08-23
# Runs conventional MPI GMRES on Seagull.
# Inputs: matrix path, optional right-hand side, and optional exact solution.
# Outputs: CSV files under OUT_ROOT or data/outputs.
# Environment: REP, OUT_ROOT, and SAVE_SOLUTION_OUTPUT.
# The job uses exclusive nodes and a lower queue priority by default.

set -euo pipefail

# Record job context before environment setup.
echo "=== job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-?} ==="
echo "  submit dir : ${SLURM_SUBMIT_DIR:-<unset>}"
echo "  matrix arg : ${1:-<none>}"
echo "  ntasks     : ${SLURM_NTASKS:-<unset>}  nodes: ${SLURM_JOB_NUM_NODES:-<unset>}"
echo "  REP        : ${REP:-<unset>}"

cd "$SLURM_SUBMIT_DIR" || { echo "cannot cd to ${SLURM_SUBMIT_DIR:-<unset>}"; exit 1; }
source scripts/setup_seagull_mpi.sh
export SLURM_EXPORT_ENV=ALL

matrix="${1:-data/matrices/parabolic_fem.mtx}"
rhs="${2:-}"
exact_solution="${3:-}"
config_file="include/common/config.hpp"

if [[ ! -f "$matrix" ]]; then
    echo "Matrix file not found: $matrix"
    exit 1
fi

if [[ -n "$rhs" && ! -f "$rhs" ]]; then
    echo "Right-hand side file not found: $rhs"
    exit 1
fi

if [[ -n "$exact_solution" && ! -f "$exact_solution" ]]; then
    echo "Exact solution file not found: $exact_solution"
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

s_max="$(
    awk -F'[=;]' '/Index s_max/ {
        gsub(/[[:space:]]/, "", $2);
        print $2;
        exit
    }' "$config_file"
)"

# Read restart separately so output directories do not collide.
restart="$(
    awk -F'[=;]' '/Index restart / {
        gsub(/[[:space:]]/, "", $2);
        print $2;
        exit
    }' "$config_file"
)"

if [[ -z "$s_step" || -z "$restart_blocks" || -z "$s_max" || -z "$restart" ]]; then
    echo "Could not read s_step, restart_blocks, s_max or restart from $config_file"
    exit 1
fi

save_solution_output="${SAVE_SOLUTION_OUTPUT:-true}"
case "$save_solution_output" in
    true|false) ;;
    *)
        echo "SAVE_SOLUTION_OUTPUT must be true or false, got: $save_solution_output"
        exit 1
        ;;
esac

nodes="${SLURM_JOB_NUM_NODES:-1}"
ranks="${SLURM_NTASKS:-${SLURM_NPROCS:-16}}"
tasks_per_node="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-unknown}}"

matrix_name="$(basename "${matrix%.*}")"
# OUT_ROOT selects an alternate output root.
output_directory="${OUT_ROOT:-data/outputs}/${matrix_name}_comm/r${restart}_s${s_step}_smax${s_max}_blocks${restart_blocks}${REP:+_rep${REP}}"

mkdir -p "$output_directory"

echo "--- Regular MPI GMRES: $nodes nodes, $ranks ranks ---"
echo "Tasks per node: $tasks_per_node"
echo "Matrix: $matrix"
echo "Right-hand side: ${rhs:-<none, using b = A * exact_solution>}"
echo "Exact solution: ${exact_solution:-<none, using all ones>}"
echo "Write solution CSV: $save_solution_output"
echo "Output directory: $output_directory"

experiment_args=("$matrix" "$output_directory" "$rhs" "$exact_solution")
if [[ "$save_solution_output" == "false" ]]; then
    experiment_args+=("--no-solution-output")
fi

mpirun --prefix "$MPI_ROOT" \
    -np "$ranks" \
    --map-by slot \
    --bind-to core \
    ./build/run_gmres_mpi_experiment \
    "${experiment_args[@]}"

echo "Results saved under $output_directory"
