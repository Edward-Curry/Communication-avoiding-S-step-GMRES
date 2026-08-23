#!/usr/bin/env bash

# File: scripts/setup_seagull_mpi.sh
# Last updated by Edward Curry: 2026-08-23
# Sets the compiler, MPI, and single-threaded BLAS environment on Seagull.
# Output: exported MPI_ROOT, PATH, LD_LIBRARY_PATH, and BLAS thread settings.

# Reports a setup failure to standard output and standard error.
# Input: failure message.
# Output: nonzero return status.
_fail() { echo "SETUP FAILED: $*" >&2; echo "SETUP FAILED: $*"; return 1; }

if ! command -v module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        . /etc/profile.d/modules.sh
    elif [ -r /usr/share/Modules/init/bash ]; then
        . /usr/share/Modules/init/bash
    fi
fi
command -v module >/dev/null 2>&1 || _fail "the 'module' command is not available"

module load gcc/15.2.0-gcc-8.5.0-r7c4jsu     || _fail "module load gcc/15.2.0-gcc-8.5.0-r7c4jsu"

export MPI_ROOT="/home/support/rl8/spack/1.1.1/opt/spack/linux-x86_64_v3/openmpi-5.0.9-2irqibqfnnap6fptp26b7bnwf5qhybuq"
export PATH="${MPI_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${LD_LIBRARY_PATH:-}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[ -d "$MPI_ROOT" ] || _fail "MPI_ROOT does not exist: $MPI_ROOT"
command -v mpirun >/dev/null 2>&1 || _fail "mpirun not on PATH after loading MPI_ROOT"

echo "MPI_ROOT=${MPI_ROOT}"
echo "mpicxx=$(command -v mpicxx || echo NOT-FOUND)"
echo "mpirun=$(command -v mpirun || echo NOT-FOUND)"
