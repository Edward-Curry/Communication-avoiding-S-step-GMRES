#ifndef COMMUNICATION_AVOIDING_GMRES_CA_MPI_STEP_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_MPI_STEP_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

// Distributed GMRES-DR deflation subspace carried across restart cycles: the
// MPI counterpart of DeflationSubspace. V is distributed by rows (n x (k+1),
// orthonormal columns); Hbar is small and replicated on every rank. Together
// they satisfy A V(:,0:k) = V(:,0:k+1) Hbar. Empty (k == 0) on the first cycle.
struct DistributedDeflationSubspace {
    DistributedDenseBlock V; // global n x (k+1), row-distributed
    DenseMatrix Hbar;        // (k+1) x k, replicated
    Index k = 0;
};

struct CAGMRESMPICycleResult {
    DistributedVector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;

    // The deflation subspace for the NEXT cycle, computed by GMRES-DR from this
    // cycle's harmonic Ritz vectors plus the residual direction (empty, k == 0,
    // when recycling is off or too few columns were built). Only
    // gmres_ca_dr_mpi_cycle populates this; the plain gmres_ca_mpi_cycle leaves
    // it empty.
    DistributedDeflationSubspace next_deflation;

    // Real, Leja-ordered Ritz-value shifts extracted from THIS cycle's own
    // Hessenberg matrix. Only populated when this cycle ran in bootstrap mode
    // (config.polynomial_basis wants Newton/ScaledNewton but the shifts
    // passed in were empty); empty otherwise. Replicated (identical on every
    // rank). Callers adopt this wholesale as the shift list for subsequent
    // cycles.
    Vector bootstrap_shifts;
};

// Plain restarted CA-GMRES cycle (no deflation), used when
// config.enable_recycling is false. shifts behaves as in the sequential
// gmres_ca_cycle.
CAGMRESMPICycleResult gmres_ca_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                         const DistributedVector& b,
                                         const DistributedVector& x_start,
                                         const DistributedVector& r_start,
                                         Scalar beta,
                                         const GMRESConfig& config,
                                         const Vector& shifts = Vector());

// GMRES-DR cycle (deflated restart), used when config.enable_recycling is true.
// The harmonic Ritz eigenproblem, QR, and least-squares all run on the small
// replicated Hessenberg, so the only new communication versus the plain cycle
// is one Allreduce of the k+1 residual coordinates at a deflated start; the
// deflation vectors are mapped back with a purely local basis combination.
CAGMRESMPICycleResult gmres_ca_dr_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                            const DistributedVector& b,
                                            const DistributedVector& x_start,
                                            const DistributedVector& r_start,
                                            Scalar beta,
                                            const GMRESConfig& config,
                                            const DistributedDeflationSubspace& deflation =
                                                DistributedDeflationSubspace(),
                                            const Vector& shifts = Vector());

}

#endif
