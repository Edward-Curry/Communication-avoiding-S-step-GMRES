#!/usr/bin/env bash

module load gcc/15.2.0-gcc-8.5.0-r7c4jsu

export MPI_ROOT="/home/support/rl8/spack/1.1.1/opt/spack/linux-x86_64_v3/openmpi-5.0.9-2irqibqfnnap6fptp26b7bnwf5qhybuq"
export PATH="${MPI_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"

echo "MPI_ROOT=${MPI_ROOT}"
echo "mpicxx=$(command -v mpicxx)"
echo "mpirun=$(command -v mpirun)"
