/**
 * @file include/parallel/distributed_vector.hpp
 * @brief Declares a contiguous row-distributed vector.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_DISTRIBUTED_VECTOR_HPP
#define PARALLEL_DISTRIBUTED_VECTOR_HPP

#include "common/types.hpp"

#include <mpi.h>
#include <vector>

namespace gmres
{
    /**
     * @brief Stores one contiguous local range of a global vector.
     */
    class DistributedVector
    {
    public:
        /** @brief Constructs an empty distributed vector. */
        DistributedVector();

        /**
         * @brief Constructs a vector from local values.
         * @param global_size Total vector length.
         * @param local_start Global index of the first local value.
         * @param local_values Values owned by this rank.
         * @param comm Communicator defining the distribution.
         */
        DistributedVector(Index global_size,
                          Index local_start,
                          Vector local_values,
                          MPI_Comm comm);

        /** @return Global vector length. */
        Index global_size() const;
        /** @return Number of values owned by this rank. */
        Index local_size() const;
        /** @return Global index of the first local value. */
        Index local_start() const;
        /** @return Exclusive global end index of local values. */
        Index local_end() const;

        /** @return Communicator defining the distribution. */
        MPI_Comm communicator() const;

        /** @return Const reference to locally owned values. */
        const Vector& local_values() const;
        /** @return Mutable reference to locally owned values. */
        Vector& local_values();

    private:
        Index global_size_ = 0;
        Index local_start_ = 0;
        Vector local_values_;
        MPI_Comm comm_ = MPI_COMM_WORLD;
    };

    /// @brief Collection of distributed vectors with a shared layout.
    using DistributedVectorList = std::vector<DistributedVector>;
}

#endif
