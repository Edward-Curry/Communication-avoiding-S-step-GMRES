# Communication-Avoiding s-Step GMRES

Research implementation of sequential GMRES, MPI-parallel GMRES, and communication-avoiding s-step GMRES for sparse linear systems.

The project studies how s-step Krylov methods reduce synchronization and global communication in distributed-memory runs, while tracking the numerical stability issues introduced by generating and orthogonalizing Krylov vectors in blocks.

## Current Solvers

Implemented solver paths:

- Sequential restarted GMRES
- MPI restarted GMRES
- Sequential communication-avoiding s-step GMRES
- MPI communication-avoiding s-step GMRES

The main experiment executables are:

- `run_gmres_experiments`
- `run_gmres_mpi_experiment`
- `run_gmres_ca_experiment`
- `run_gmres_ca_mpi_experiment`

The Seagull batch scripts use:

- `scripts/run.sh` for MPI CA-GMRES
- `scripts/run_comm.sh` for regular MPI GMRES

## Current CA-GMRES Setup

The CA-GMRES path currently uses:

- Monomial s-step basis generation
- BCGS2 for inter-block orthogonalization
- CholQR for intra-block orthogonalization
- Partial Cholesky truncation when a block becomes numerically unsafe
- Dense block storage for Krylov basis columns
- Per-block least-squares residual estimates using Givens rotations
- Restarted GMRES with the current solution carried across restart cycles

The default configuration is in `include/common/config.hpp`:

```cpp
Index restart = 30;
Index max_iterations = 100000;
Scalar tolerance = 1e-10;
Index restart_blocks = 6;
Index s_step = 5;
BlockOrthogonalizationMethod block_orthogonalization =
    BlockOrthogonalizationMethod::BCGS2CholQR;
```

So the current CA restart dimension is:

```text
restart_blocks * s_step = 6 * 5 = 30
```

## Completed Work

Core implementation:

- CSR sparse matrix storage and Matrix Market input
- Sequential vector operations, Arnoldi iteration, Givens rotations, and restarted GMRES
- Distributed vectors and distributed sparse matrix support
- MPI sparse matrix-vector multiplication with cached halo exchange
- MPI Arnoldi and restarted MPI GMRES
- Sequential and MPI CA-GMRES
- Experiment output for performance, residual history, and solver comparison

CA-GMRES algorithm work:

- s-step Krylov block generation
- BCGS2-CholQR block orthogonalization
- Partial Cholesky support for CholQR truncation
- Monomial Hessenberg assembly for accepted s-step blocks
- Dense block basis storage using `DenseBlock` and `DistributedDenseBlock`
- CA residual history with iteration number and residual type
- Mid-cycle convergence checks after each accepted s-step block
- Scaled Gram-matrix partial Cholesky to reduce false truncation from column norm imbalance

Recent code organization:

- Krylov blocks and basis columns are now stored as dense column-major blocks instead of separate vector lists.
- Orthogonalization routines take a basis block plus a valid leading-column count.
- Accepted `Q_block` columns are appended directly into the current cycle basis.
- Solution updates use dense block multiplication over active basis columns.
- Partial Cholesky now factors a diagonally scaled Gram matrix, then unscales the triangular factor.

## What Changed Recently

The recent solver speedup mainly came from removing repeated basis packing and unpacking.

Previous structure:

```text
store basis as separate vectors
pack old basis into DenseBlock every block
orthogonalize
unpack new Q vectors
append to vector list
repeat
```

Current structure:

```text
store cycle basis once as DenseBlock
pass basis plus basis_cols
orthogonalize against active leading columns
append accepted Q columns directly
update solution with one dense block operation
```

This reduces memory traffic and lets BLAS operate on contiguous column-major data.

The partial Cholesky stability fix is separate. It helps prevent useful monomial block columns from being rejected just because their norms differ by many orders of magnitude.

## Partial Cholesky Truncation

CholQR forms the block Gram matrix:

```text
G = X^T X
```

Partial Cholesky attempts to compute:

```text
G = R^T R
```

If the next pivot is too small, non-finite, or otherwise unsafe, the factorization stops and only the stable leading columns are accepted.

For example, if `s_step = 5` but only 3 columns are accepted:

```text
accepted_columns = 3
truncated = true
R is 3 x 3
Q has 3 columns
```

Then CholQR computes:

```text
Q = X_accepted R^{-1}
```

The remaining generated columns are discarded, and GMRES continues from the last accepted orthonormal column.

## Current Algorithm State

The current implementation has several pieces needed for a stable adaptive s-step GMRES method:

- BCGS2-CholQR block orthogonalization
- Partial CholQR truncation
- Discarding unaccepted higher-power basis columns
- Per-block convergence monitoring

The full adaptive version is not complete yet. The current implementation still uses a fixed configured `s_step` and a monomial polynomial basis.

## Remaining Algorithm Roadmap

Recommended implementation order:

1. **Condition-number based stopping for partial CholQR**
   - Extend partial Cholesky beyond pivot-only stopping.
   - Stop when the estimated condition number exceeds a bound:
     ```text
     kappa(R_1:j,1:j) > Omega
     ```
   - Start with SVD-based checking for simplicity.
   - Later replace or supplement it with an incremental condition estimator.

2. **Adaptive s-step update**
   - Use the accepted block size to choose the next attempted block size.
   - Planned update:
     ```text
     s_next = accepted_columns
     ```
   - This turns truncation into a true adaptive block-size mechanism.

3. **Newton or scaled Newton polynomial basis**
   - Add a better polynomial basis than monomial powers.
   - Target recurrence:
     ```text
     V_1 = q
     V_j = (A V_{j-1} - theta_{j-1} V_{j-1}) / gamma_{j-1}
     ```
   - `theta` values come from Ritz values.
   - Scaled Newton uses `gamma` values to control vector norm growth.

4. **Initial s estimator**
   - Use Ritz values to estimate a good starting `s0`.
   - Avoid starting with a block size that is much too large and wastes many generated columns.

5. **s-block recycling across restarts**
   - Keep useful accepted orthonormal `Q_block`s across restart cycles.
   - Reuse those blocks as an augmentation space in later cycles.
   - Start with a simple strategy such as keeping the best one or two accepted blocks.

## Build

Generic local build:

```bash
cmake -S . -B build
cmake --build build
```

On Seagull:

```bash
chmod +x scripts/*.sh
source scripts/setup_seagull_mpi.sh
./scripts/build.sh
```

## Run Experiments on Seagull

Run only CA-GMRES:

```bash
for matrix in \
  data/matrices/easy_3d_100.mtx \
  data/matrices/parabolic_fem.mtx \
  data/matrices/poisson_3d_171.mtx
do
  for p in 1 2 4 8 16 32 64 128 256
  do
    nodes=$(( (p + 15) / 16 ))
    tpn=$p
    if [ "$tpn" -gt 16 ]; then
      tpn=16
    fi

    sbatch --nodes="$nodes" \
           --ntasks="$p" \
           --ntasks-per-node="$tpn" \
           scripts/run.sh "$matrix"
  done
done
```

Run regular MPI GMRES for comparison by replacing `scripts/run.sh` with:

```text
scripts/run_comm.sh
```

Results are written under:

```text
data/outputs/<matrix_name>/s<s_step>_blocks<restart_blocks>/
```

## Test Targets

Important tests include:

- `test_sparse_matrix`
- `test_vector_ops`
- `test_givens`
- `test_orthogonalization`
- `test_arnoldi_seq`
- `test_gmres_seq`
- `test_gmres_mpi`
- `test_gmres_ca`
- `test_bcgs2_cholqr`
- `test_sstep_arnoldi_mpi`
- `test_gmres_ca_mpi`

Build all targets with:

```bash
cmake --build build
```

Run individual tests from the build directory or via the generated executable path.

## Notes

- The sparse matrix `A` remains sparse.
- GMRES and CA-GMRES basis vectors are dense, which is expected.
- `DenseBlock` stores generated Krylov vectors as dense columns; it does not convert the sparse matrix into a dense matrix.
- The current CA basis type is monomial only. `Newton` and `ScaledNewton` exist in the enum but are not implemented yet.
- The current adaptive behavior truncates unsafe blocks, but it does not yet update the next attempted `s_step`.
