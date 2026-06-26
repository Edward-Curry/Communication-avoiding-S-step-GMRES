#!/bin/bash
#SBATCH -J gmres_scaling
#SBATCH -o gmres_scaling_%j.out
#SBATCH -e gmres_scaling_%j.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=10:00:00

set -e

cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/setup_seagull_mpi.sh

matrix="${1:-data/matrices/parabolic_fem.mtx}"
output_directory="data/outputs/$(basename "${matrix%.*}")"

if [[ ! -f "$matrix" ]]; then
    echo "Matrix file not found: $matrix"
    exit 1
fi

echo "--- Sequential GMRES ---"
./build/run_gmres_experiments "$matrix" "$output_directory"

echo "--- Sequential CA-GMRES ---"
./build/run_gmres_ca_experiment "$matrix" "$output_directory"

for ranks in 1 2 4 8 16; do
    echo "--- MPI GMRES: $ranks ranks ---"
    mpirun -np "$ranks" \
        ./build/run_gmres_mpi_experiment \
        "$matrix" \
        "$output_directory"

    echo "--- MPI CA-GMRES: $ranks ranks ---"
    mpirun -np "$ranks" \
        ./build/run_gmres_ca_mpi_experiment \
        "$matrix" \
        "$output_directory"
done

echo "Results saved in $output_directory"
