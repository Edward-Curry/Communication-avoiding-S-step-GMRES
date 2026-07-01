#!/bin/bash
#SBATCH -J gmres_chol
#SBATCH -o gmres_chol_%j.out
#SBATCH -e gmres_chol_%j.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00

set -e

cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/setup_seagull_mpi.sh

matrix="${1:-data/matrices/parabolic_fem.mtx}"
max_nodes="${SLURM_JOB_NUM_NODES:-1}"
output_directory="data/outputs/$(basename "${matrix%.*}")_chol_n${max_nodes}"

if [[ ! -f "$matrix" ]]; then
    echo "Matrix file not found: $matrix"
    exit 1
fi

for nodes in 1 2 4 8; do
    ranks=$((nodes * 16))

    echo "--- MPI CA-GMRES BCGS2-CholQR: $nodes nodes, $ranks ranks ---"
    mpirun -np "$ranks" \
        --map-by ppr:16:node \
        --bind-to core \
        ./build/run_gmres_ca_mpi_experiment \
        "$matrix" \
        "$output_directory"
done

echo "Results saved in $output_directory"
