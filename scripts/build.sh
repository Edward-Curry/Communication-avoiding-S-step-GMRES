#!/usr/bin/env bash

# File: scripts/build.sh
# Last updated by Edward Curry: 2026-08-23
# Configures and builds the project with Seagull MPI settings.
# Inputs: BUILD_DIR and BUILD_JOBS may override the defaults.
# Output: CMake build tree containing executables and tests.

set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_directory}/.." && pwd)"
build_directory="${BUILD_DIR:-${repository_root}/build}"
build_jobs="${BUILD_JOBS:-16}"

source "${script_directory}/setup_seagull_mpi.sh"

cmake \
    -S "${repository_root}" \
    -B "${build_directory}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${MPI_ROOT}/bin/mpicxx"

cmake \
    --build "${build_directory}" \
    --parallel "${build_jobs}"

echo "Build completed: ${build_directory}"
