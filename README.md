# Communication-Avoiding s-Step GMRES

Research implementation of restarted GMRES and communication-avoiding
s-step GMRES for sparse linear systems. The project provides sequential and
MPI solvers, Matrix Market input, reproducible experiment output, and Slurm
scripts for Seagull.

## Implemented Methods

- Sequential restarted GMRES
- Distributed restarted GMRES with cached SpMV halo exchange
- Sequential and distributed CA-GMRES
- Block modified Gram-Schmidt and BCGS2-CholQR orthogonalization
- Partial Cholesky truncation with pivot or triangular-condition stopping
- Adaptive s-step selection and initial-width probing
- Monomial, Newton, and scaled-Newton polynomial bases
- Harmonic-Ritz recycling between restart cycles

Solver parameters are defined in
[`include/common/config.hpp`](include/common/config.hpp). The default
configuration uses BCGS2-CholQR, adaptive s-step control, condition-limited
partial Cholesky, and a recycle dimension of two.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app/` | Small standalone entry point. |
| `experiments/` | Executables that run a solver and write CSV output. |
| `include/` | Public headers and solver interfaces. |
| `src/` | Sequential, distributed, and CA-GMRES implementations. |
| `tests/` | Unit and MPI regression tests. |
| `scripts/` | Build, matrix-generation, and Seagull batch scripts. |
| `data/matrices/` | Matrix Market inputs and small test fixtures. |
| `data/outputs/` | Generated experiment CSV files. |

## Requirements

- CMake 3.16 or later
- A C++23 compiler
- MPI with `mpicxx` and `mpirun`
- BLAS with CBLAS and LAPACKE headers, such as OpenBLAS

## Build

Build locally with an MPI compiler available on `PATH`:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build --parallel
```

On Seagull, from the repository root:

```bash
chmod +x scripts/*.sh
module load cmake/3.31.9-gcc-15.2.0-ylutpfi
module load openblas/0.3.30-gcc-15.2.0-jjpzwqx
source scripts/setup_seagull_mpi.sh
./scripts/build.sh
```

`build/` is deliberately ignored by Git.

## Tests

Run serial tests from the repository root:

```bash
./build/test_sparse_matrix
./build/test_vector_ops
./build/test_io
./build/test_givens
./build/test_orthogonalization
./build/test_arnoldi_seq
./build/test_gmres_seq
./build/test_bcgs2_cholqr
./build/test_gmres_ca
```

Run MPI tests with at least two ranks:

```bash
mpirun -np 2 ./build/test_distributed_sparse_matrix
mpirun -np 2 ./build/test_gmres_mpi
mpirun -np 2 ./build/test_sstep_arnoldi_mpi
mpirun -np 2 ./build/test_gmres_ca_mpi
```

## Input and Local Runs

The MPI experiment executables accept the following interface:

```text
<matrix> <output_directory> [right_hand_side] [exact_solution] [--no-solution-output]
```

If no right-hand side is supplied, the executable uses an all-ones exact
solution and constructs `b = A * 1`. A supplied right-hand side does not
support a forward-error check unless an exact-solution vector is also supplied.

For example:

```bash
mpirun -np 2 ./build/run_gmres_mpi_experiment \
    data/matrices/test_symmetric_3x3.mtx data/outputs/demo_gmres

mpirun -np 2 ./build/run_gmres_ca_mpi_experiment \
    data/matrices/test_symmetric_3x3.mtx data/outputs/demo_ca
```

The sequential experiment executables accept a matrix and an optional output
directory:

```bash
./build/run_gmres_experiments data/matrices/test_3x3.mtx data/outputs/demo_seq
./build/run_gmres_ca_experiment data/matrices/test_3x3.mtx data/outputs/demo_ca_seq
```

## Seagull Batch Runs

Use `scripts/run.sh` for CA-GMRES and `scripts/run_comm.sh` for conventional
MPI GMRES. Each takes a matrix, optional right-hand side, and optional exact
solution:

```bash
sbatch --nodes=1 --ntasks=16 --ntasks-per-node=16 \
    scripts/run.sh data/matrices/parabolic_fem.mtx

sbatch --nodes=1 --ntasks=16 --ntasks-per-node=16 \
    scripts/run_comm.sh data/matrices/parabolic_fem.mtx
```

For a recycle-dimension sweep, set `GMRES_RECYCLE_COUNT` when submitting the
CA job:

```bash
GMRES_RECYCLE_COUNT=4 sbatch --nodes=1 --ntasks=16 --ntasks-per-node=16 \
    scripts/run.sh data/matrices/parabolic_fem.mtx
```

The scripts place results below:

```text
data/outputs/<matrix_name>/s<s_step>_smax<s_max>_blocks<restart_blocks>/
data/outputs/<matrix_name>_comm/r<restart>_s<s_step>_smax<s_max>_blocks<restart_blocks>/
```

Set `OUT_ROOT` to send a campaign to a separate output tree, `REP` to label a
repeat, and `SAVE_SOLUTION_OUTPUT=false` to suppress solution-vector CSV files.

## Matrix Generation

Generate the small SPD fixtures locally:

```bash
python3 scripts/generate_easy_matrix.py --size 100 --kind easy27 \
    --output data/matrices/easy_3d_100.mtx

python3 scripts/generate_easy_matrix.py --size 171 --kind poisson7 \
    --output data/matrices/poisson_3d_171.mtx
```

Generate a scalable three-dimensional convection-diffusion matrix locally:

```bash
python3 scripts/generate_scaling_matrix.py --size 160 --diag-boost 0.001858 \
    --output data/matrices/scaling_160.mtx
```

On Seagull, submit the same generator through Slurm:

```bash
sbatch scripts/gen_matrix.sh data/matrices/scaling_160.mtx 160 0.001858
```

Large research matrices are intentionally local inputs and are ignored by
Git. Place any Matrix Market matrix and optional vector files under
`data/matrices/` before running an experiment. The small `test_*.mtx` fixtures
remain available for tests and smoke runs.

## Output Files

Each experiment writes CSV files with a common metadata prefix:

- `*_config.csv` records the solver configuration.
- `*_convergence.csv` records residual observations.
- `*_performance.csv` records elapsed time, iterations, and halo statistics.
- `*_accuracy.csv` records residual and forward-error metrics.
- `*_solution.csv` records the final solution when enabled.

The output directory is part of the experiment interface, so separate
campaigns can be retained without overwriting previous results.
