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

set -e

cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/setup_seagull_mpi.sh

matrix="${1:-data/matrices/parabolic_fem.mtx}"

if [[ ! -f "$matrix" ]]; then
    echo "Matrix file not found: $matrix"
    exit 1
fi

nodes="${SLURM_JOB_NUM_NODES:-1}"
ranks=$((nodes * 16))

matrix_name="$(basename "${matrix%.*}")"
output_directory="data/outputs/${matrix_name}_halo_s5"

mkdir -p "$output_directory"

echo "--- MPI CA-GMRES BCGS2-CholQR: $nodes nodes, $ranks ranks ---"
echo "Matrix: $matrix"
echo "Output directory: $output_directory"

mpirun -np "$ranks" \
    --map-by ppr:16:node \
    --bind-to core \
    ./build/run_gmres_ca_mpi_experiment \
    "$matrix" \
    "$output_directory"

echo "Results saved under $output_directory"