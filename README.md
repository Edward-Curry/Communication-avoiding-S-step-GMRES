# Communication-avoiding-S-step-GMRES

Research implementation of:

- Sequential GMRES
- MPI-parallel classical GMRES
- Communication-avoiding s-step GMRES

for solving large sparse linear systems.

The project is based on research into communication-avoiding Krylov subspace methods for high-performance computing (HPC), with a focus on reducing synchronization and communication costs in distributed-memory environments.

---

## Project Goals

The aim of this project is to:

- Implement a baseline sequential GMRES solver
- Develop a classical MPI-parallel GMRES implementation
- Implement communication-avoiding / s-step GMRES algorithms
- Investigate communication bottlenecks in Krylov methods
- Compare convergence, scalability, and numerical stability

---

## Planned Solver Implementations

### 1. Sequential GMRES
Baseline implementation using:
- Arnoldi iteration
- Modified Gram-Schmidt orthogonalization
- Givens rotations
- CSR sparse matrices

Purpose:
- correctness verification
- debugging
- convergence reference

---

### 2. Parallel Classical GMRES
MPI-based distributed-memory implementation of standard GMRES.

Features:
- distributed sparse matrix-vector multiplication
- global reductions using MPI
- parallel Arnoldi iteration

Purpose:
- communication baseline
- scalability benchmarking

---

### 3. Communication-Avoiding GMRES
Implementation of communication-avoiding and s-step GMRES methods.

Planned features:
- s-step Krylov basis generation
- block orthogonalization
- reduced synchronization frequency
- adaptive basis sizing

Purpose:
- reduce communication overhead
- improve scalability on distributed systems

---

## Current Status

- [x] Project skeleton created
- [x] Basic CMake build configured
- [x] `run_seq` executable builds
- [ ] Basic vector operations
- [ ] CSR sparse matrix storage
- [ ] Sparse matrix-vector multiplication
- [ ] Modified Gram-Schmidt orthogonalization
- [ ] Arnoldi iteration
- [ ] Givens rotations
- [ ] Sequential restarted GMRES
- [ ] MPI-parallel GMRES
- [ ] Communication-avoiding s-step GMRES

---

## Repository Structure

```text
Communication-avoiding-S-step-GMRES/
│
├── CMakeLists.txt
├── README.md
│
├── build/
│
├── data/
│   ├── matrices/
│   └── outputs/
│
├── include/
│   ├── common/
│   │   ├── sparse_matrix.hpp
│   │   ├── vector_ops.hpp
│   │   ├── orthogonalization.hpp
│   │   ├── givens.hpp
│   │   ├── io.hpp
│   │   ├── utils.hpp
│   │   ├── types.hpp
│   │   └── config.hpp
│   │
│   ├── sequential/
│   │   ├── gmres_seq.hpp
│   │   └── arnoldi_seq.hpp
│   │
│   ├── parallel/
│   │   ├── gmres_mpi.hpp
│   │   ├── arnoldi_mpi.hpp
│   │   ├── distributed_vector.hpp
│   │   └── distributed_matrix.hpp
│   │
│   └── communication_avoiding/
│       ├── gmres_ca.hpp
│       ├── sstep_arnoldi.hpp
│       ├── block_orthogonalization.hpp
│       ├── krylov_basis.hpp
│       └── polynomial_basis.hpp
│
├── src/
│   ├── common/
│   │   ├── sparse_matrix.cpp
│   │   ├── vector_ops.cpp
│   │   ├── orthogonalization.cpp
│   │   ├── givens.cpp
│   │   ├── io.cpp
│   │   └── utils.cpp
│   │
│   ├── sequential/
│   │   ├── gmres_seq.cpp
│   │   └── arnoldi_seq.cpp
│   │
│   ├── parallel/
│   │   ├── gmres_mpi.cpp
│   │   ├── arnoldi_mpi.cpp
│   │   ├── distributed_vector.cpp
│   │   └── distributed_matrix.cpp
│   │
│   └── communication_avoiding/
│       ├── gmres_ca.cpp
│       ├── sstep_arnoldi.cpp
│       ├── block_orthogonalization.cpp
│       ├── krylov_basis.cpp
│       └── polynomial_basis.cpp
│
├── app/
│   ├── run_seq.cpp
│   ├── run_mpi.cpp
│   └── run_ca.cpp
│
├── tests/
│   ├── test_spmv.cpp
│   ├── test_vector_ops.cpp
│   ├── test_orthogonalization.cpp
│   ├── test_arnoldi.cpp
│   ├── test_givens.cpp
│   ├── test_gmres_seq.cpp
│   ├── test_gmres_mpi.cpp
│   └── test_gmres_ca.cpp
│
└── scripts/
    ├── build.sh
    ├── benchmark.sh
    ├── scaling_tests.sh
    └── run_tests.sh
```

---

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

## README Outline

- Project overview
- Project goals
- Planned solver implementations
- Current status
- Repository structure
- Build
- Run
- Roadmap
- Known limitations