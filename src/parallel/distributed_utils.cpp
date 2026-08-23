/**
 * @file src/parallel/distributed_utils.cpp
 * @brief Implements distributed residual, error, and convergence utilities.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "parallel/distributed_utils.hpp"

#include "parallel/distributed_vector_ops.hpp"

namespace gmres
{
    DistributedVector compute_residual_mpi(const DistributedSparseMatrixCSR& A,
                                           const DistributedVector& b,
                                           const DistributedVector& x)
    {
        DistributedVector residual = b;
        DistributedVector Ax = A.multiply(x);

        axpy_local(-1.0, Ax, residual);

        return residual;
    }

    Scalar residual_norm_mpi(const DistributedSparseMatrixCSR& A,
                             const DistributedVector& b,
                             const DistributedVector& x)
    {
        return norm2_mpi(compute_residual_mpi(A, b, x));
    }

    Scalar relative_residual_norm_mpi(const DistributedSparseMatrixCSR& A,
                                      const DistributedVector& b,
                                      const DistributedVector& x)
    {
        const Scalar denominator = norm2_mpi(b);
        const Scalar numerator = residual_norm_mpi(A, b, x);

        if (denominator == 0.0)
        {
            return numerator;
        }

        return numerator / denominator;
    }

    Scalar forward_error_norm_mpi(const DistributedVector& x,
                                  const DistributedVector& x_true)
    {
        DistributedVector error = x;

        axpy_local(-1.0, x_true, error);

        return norm2_mpi(error);
    }

    Scalar relative_forward_error_mpi(const DistributedVector& x,
                                      const DistributedVector& x_true)
    {
        const Scalar denominator = norm2_mpi(x_true);
        const Scalar numerator = forward_error_norm_mpi(x, x_true);

        if (denominator == 0.0)
        {
            return numerator;
        }

        return numerator / denominator;
    }

    bool has_converged_absolute_residual_mpi(Scalar residual,
                                             Scalar tolerance)
    {
        return residual <= tolerance;
    }

    bool has_converged_relative_residual_mpi(Scalar relative_residual,
                                             Scalar tolerance)
    {
        return relative_residual <= tolerance;
    }

    HaloExchangeSummary halo_exchange_summary_mpi(
        const DistributedSparseMatrixCSR& A)
    {
        const HaloExchangeStatistics local = A.halo_exchange_statistics();
        const unsigned long long local_counts[] = {
            static_cast<unsigned long long>(local.receive_peer_count),
            static_cast<unsigned long long>(local.send_peer_count),
            static_cast<unsigned long long>(local.receive_value_count),
            static_cast<unsigned long long>(local.send_value_count)
        };
        unsigned long long sums[4] = {};
        unsigned long long maxima[4] = {};

        MPI_Reduce(local_counts,
                   sums,
                   4,
                   MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM,
                   0,
                   A.communicator());
        MPI_Reduce(local_counts,
                   maxima,
                   4,
                   MPI_UNSIGNED_LONG_LONG,
                   MPI_MAX,
                   0,
                   A.communicator());

        int rank = 0;
        int process_count = 1;
        MPI_Comm_rank(A.communicator(), &rank);
        MPI_Comm_size(A.communicator(), &process_count);

        HaloExchangeSummary summary;
        if (rank == 0)
        {
            summary.available = true;
            summary.mean_receive_peers =
                static_cast<double>(sums[0]) / process_count;
            summary.max_receive_peers = static_cast<Index>(maxima[0]);
            summary.mean_send_peers =
                static_cast<double>(sums[1]) / process_count;
            summary.max_send_peers = static_cast<Index>(maxima[1]);
            summary.mean_receive_values =
                static_cast<double>(sums[2]) / process_count;
            summary.max_receive_values = static_cast<Index>(maxima[2]);
            summary.mean_send_values =
                static_cast<double>(sums[3]) / process_count;
            summary.max_send_values = static_cast<Index>(maxima[3]);
        }

        return summary;
    }
}
