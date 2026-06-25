#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_gmres_experiments.sh MATRIX [PROCESS_COUNT ...]

Examples:
  bash scripts/run_gmres_experiments.sh data/matrices/parabolic_fem.mtx
  bash scripts/run_gmres_experiments.sh data/matrices/parabolic_fem.mtx 2 4 8 16 32

Environment:
  BUILD_DIR       Directory containing the four experiment executables.
                  Default: <repository>/build
  OUTPUT_DIR      Root output directory.
                  Default: <repository>/data/outputs/<matrix-name>
  TASKS_PER_NODE  Maximum MPI processes per node. Default: 16
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_directory}/.." && pwd)"

matrix_path="$1"
shift

if [[ ! -f "${matrix_path}" ]]; then
    printf 'Matrix file does not exist: %s\n' "${matrix_path}" >&2
    exit 1
fi

matrix_path="$(cd "$(dirname "${matrix_path}")" && pwd)/$(basename "${matrix_path}")"
matrix_filename="$(basename "${matrix_path}")"
matrix_name="${matrix_filename%.*}"

build_directory="${BUILD_DIR:-${repository_root}/build}"
output_directory="${OUTPUT_DIR:-${repository_root}/data/outputs/${matrix_name}}"
tasks_per_node="${TASKS_PER_NODE:-16}"

if [[ ! "${tasks_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'TASKS_PER_NODE must be a positive integer: %s\n' "${tasks_per_node}" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    process_counts=("$@")
else
    process_counts=(1 2 4 8 16)
fi

sequential_gmres="${build_directory}/run_gmres_experiments"
sequential_ca_gmres="${build_directory}/run_gmres_ca_experiment"
mpi_gmres="${build_directory}/run_gmres_mpi_experiment"
mpi_ca_gmres="${build_directory}/run_gmres_ca_mpi_experiment"

for executable in \
    "${sequential_gmres}" \
    "${sequential_ca_gmres}" \
    "${mpi_gmres}" \
    "${mpi_ca_gmres}"
do
    if [[ ! -x "${executable}" ]]; then
        printf 'Missing experiment executable: %s\n' "${executable}" >&2
        printf 'Build the four experiment targets before running this script.\n' >&2
        exit 1
    fi
done

mkdir -p "${output_directory}"

run_mpi() {
    local processes="$1"
    local executable="$2"
    local nodes=$(( (processes + tasks_per_node - 1) / tasks_per_node ))

    printf 'Running %s with %d processes across %d node(s)\n' \
        "$(basename "${executable}")" \
        "${processes}" \
        "${nodes}"

    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
        if [[ -n "${SLURM_NNODES:-}" ]] && (( nodes > SLURM_NNODES )); then
            printf 'Run requires %d nodes, but the SLURM allocation has %d.\n' \
                "${nodes}" \
                "${SLURM_NNODES}" >&2
            exit 1
        fi

        srun \
            --nodes="${nodes}" \
            --ntasks="${processes}" \
            --ntasks-per-node="${tasks_per_node}" \
            "${executable}" \
            "${matrix_path}" \
            "${output_directory}"
    elif command -v mpirun >/dev/null 2>&1; then
        mpirun \
            -np "${processes}" \
            "${executable}" \
            "${matrix_path}" \
            "${output_directory}"
    elif command -v mpiexec >/dev/null 2>&1; then
        mpiexec \
            -n "${processes}" \
            "${executable}" \
            "${matrix_path}" \
            "${output_directory}"
    else
        printf 'No MPI launcher found. Expected srun, mpirun, or mpiexec.\n' >&2
        exit 1
    fi
}

printf 'Matrix: %s\n' "${matrix_path}"
printf 'Output: %s\n' "${output_directory}"
printf 'Processes per node: %s\n' "${tasks_per_node}"

printf 'Running sequential GMRES\n'
"${sequential_gmres}" "${matrix_path}" "${output_directory}"

printf 'Running sequential CA-GMRES\n'
"${sequential_ca_gmres}" "${matrix_path}" "${output_directory}"

for processes in "${process_counts[@]}"; do
    if [[ ! "${processes}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Process count must be a positive integer: %s\n' "${processes}" >&2
        exit 1
    fi

    run_mpi "${processes}" "${mpi_gmres}"
    run_mpi "${processes}" "${mpi_ca_gmres}"
done

printf 'All experiments completed. Results are in %s\n' "${output_directory}"
