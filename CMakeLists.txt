cmake_minimum_required(VERSION 3.16)

project(CommunicationAvoidingSstepGMRES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(run_seq
    app/run_seq.cpp
)

target_include_directories(run_seq PRIVATE include)

add_executable(test_sparse_matrix
    tests/test_sparse_matrix.cpp
    src/common/sparse_matrix.cpp
)

target_include_directories(test_sparse_matrix PRIVATE include)
target_link_libraries(test_sparse_matrix PRIVATE stdc++exp)

add_executable(test_vector_ops
    tests/test_vector_ops.cpp
    src/common/vector_ops.cpp
)

target_include_directories(test_vector_ops PRIVATE include)
target_link_libraries(test_vector_ops PRIVATE stdc++exp)

add_executable(test_io
    tests/test_io.cpp
    src/common/io.cpp
    src/common/sparse_matrix.cpp
)

target_include_directories(test_io PRIVATE include)
target_link_libraries(test_io PRIVATE stdc++exp)

add_executable(test_givens
    tests/test_givens.cpp
    src/common/givens.cpp
)

target_include_directories(test_givens PRIVATE include)
target_link_libraries(test_givens PRIVATE stdc++exp)

add_executable(test_orthogonalization
    tests/test_orthogonalization.cpp
    src/common/orthogonalization.cpp
    src/common/vector_ops.cpp
)

target_include_directories(test_orthogonalization PRIVATE include)
target_link_libraries(test_orthogonalization PRIVATE stdc++exp)

add_executable(test_arnoldi_seq
    tests/test_arnoldi_seq.cpp
    src/sequential/arnoldi_seq.cpp
    src/common/orthogonalization.cpp
    src/common/vector_ops.cpp
    src/common/sparse_matrix.cpp
)

target_include_directories(test_arnoldi_seq PRIVATE include)
target_link_libraries(test_arnoldi_seq PRIVATE stdc++exp)

add_executable(test_gmres_seq
    tests/test_gmres_seq.cpp
    src/sequential/gmres_seq.cpp
    src/sequential/gmres_seq_step.cpp
    src/common/givens.cpp
    src/common/orthogonalization.cpp
    src/common/vector_ops.cpp
    src/common/sparse_matrix.cpp
)

target_include_directories(test_gmres_seq PRIVATE include)
target_link_libraries(test_gmres_seq PRIVATE stdc++exp)

find_package(MPI REQUIRED)

add_executable(test_gmres_mpi
    tests/test_gmres_mpi.cpp
    src/parallel/gmres_mpi.cpp
    src/parallel/gmres_mpi_step.cpp
    src/parallel/distributed_vector.cpp
    src/parallel/distributed_vector_ops.cpp
    src/parallel/distributed_sparse_matrix.cpp
    src/parallel/distributed_orthogonalization.cpp
    src/common/givens.cpp
)

target_include_directories(test_gmres_mpi PRIVATE include)

target_link_libraries(test_gmres_mpi
    PRIVATE
    MPI::MPI_CXX
    stdc++exp
)

add_executable(test_gmres_ca
    tests/test_gmres_ca.cpp
    src/communication_avoiding/gmres_ca.cpp
    src/communication_avoiding/gmres_ca_step.cpp
    src/communication_avoiding/sstep_arnoldi.cpp
    src/communication_avoiding/block_orthogonalization.cpp
    src/communication_avoiding/krylov_basis.cpp
    src/communication_avoiding/polynomial_basis.cpp
    src/common/givens.cpp
    src/common/vector_ops.cpp
    src/common/sparse_matrix.cpp
)

target_include_directories(test_gmres_ca PRIVATE include)
target_link_libraries(test_gmres_ca PRIVATE stdc++exp)

add_executable(test_sstep_arnoldi_mpi
    tests/test_sstep_arnoldi_mpi.cpp
    src/communication_avoiding/sstep_arnoldi_mpi.cpp
    src/communication_avoiding/krylov_basis_mpi.cpp
    src/communication_avoiding/block_orthogonalization_mpi.cpp
    src/communication_avoiding/polynomial_basis.cpp
    src/parallel/distributed_vector.cpp
    src/parallel/distributed_vector_ops.cpp
    src/parallel/distributed_sparse_matrix.cpp
)

target_include_directories(test_sstep_arnoldi_mpi PRIVATE include)

target_link_libraries(test_sstep_arnoldi_mpi
    PRIVATE
    MPI::MPI_CXX
    stdc++exp
)

add_executable(test_gmres_ca_mpi
    tests/test_gmres_ca_mpi.cpp
    src/communication_avoiding/gmres_ca_mpi.cpp
    src/communication_avoiding/gmres_ca_mpi_step.cpp
    src/communication_avoiding/sstep_arnoldi_mpi.cpp
    src/communication_avoiding/krylov_basis_mpi.cpp
    src/communication_avoiding/block_orthogonalization_mpi.cpp
    src/communication_avoiding/polynomial_basis.cpp
    src/parallel/distributed_vector.cpp
    src/parallel/distributed_vector_ops.cpp
    src/parallel/distributed_sparse_matrix.cpp
    src/common/givens.cpp
)

target_include_directories(test_gmres_ca_mpi PRIVATE include)

target_link_libraries(test_gmres_ca_mpi
    PRIVATE
    MPI::MPI_CXX
    stdc++exp
)

add_executable(run_gmres_experiments
    experiments/run_gmres_experiments.cpp
    src/sequential/gmres_seq.cpp
    src/sequential/gmres_seq_step.cpp
    src/common/givens.cpp
    src/common/io.cpp
    src/common/orthogonalization.cpp
    src/common/sparse_matrix.cpp
    src/common/utils.cpp
    src/common/vector_ops.cpp
)

target_include_directories(run_gmres_experiments PRIVATE include)
target_link_libraries(run_gmres_experiments PRIVATE stdc++exp)

add_executable(run_gmres_ca_experiment
    experiments/run_gmres_ca_experiment.cpp
    src/communication_avoiding/gmres_ca.cpp
    src/communication_avoiding/gmres_ca_step.cpp
    src/communication_avoiding/sstep_arnoldi.cpp
    src/communication_avoiding/block_orthogonalization.cpp
    src/communication_avoiding/krylov_basis.cpp
    src/communication_avoiding/polynomial_basis.cpp
    src/common/givens.cpp
    src/common/io.cpp
    src/common/sparse_matrix.cpp
    src/common/utils.cpp
    src/common/vector_ops.cpp
)

target_include_directories(run_gmres_ca_experiment PRIVATE include)
target_link_libraries(run_gmres_ca_experiment PRIVATE stdc++exp)

add_executable(run_gmres_mpi_experiment
    experiments/run_gmres_mpi_experiment.cpp
    src/parallel/gmres_mpi.cpp
    src/parallel/gmres_mpi_step.cpp
    src/parallel/distributed_io.cpp
    src/parallel/distributed_utils.cpp
    src/parallel/distributed_vector.cpp
    src/parallel/distributed_vector_ops.cpp
    src/parallel/distributed_sparse_matrix.cpp
    src/parallel/distributed_orthogonalization.cpp
    src/common/givens.cpp
    src/common/io.cpp
    src/common/sparse_matrix.cpp
)

target_include_directories(run_gmres_mpi_experiment PRIVATE include)

target_link_libraries(run_gmres_mpi_experiment
    PRIVATE
    MPI::MPI_CXX
    stdc++exp
)

add_executable(run_gmres_ca_mpi_experiment
    experiments/run_gmres_ca_mpi_experiment.cpp
    src/communication_avoiding/gmres_ca_mpi.cpp
    src/communication_avoiding/gmres_ca_mpi_step.cpp
    src/communication_avoiding/sstep_arnoldi_mpi.cpp
    src/communication_avoiding/krylov_basis_mpi.cpp
    src/communication_avoiding/block_orthogonalization_mpi.cpp
    src/communication_avoiding/polynomial_basis.cpp
    src/parallel/distributed_io.cpp
    src/parallel/distributed_utils.cpp
    src/parallel/distributed_vector.cpp
    src/parallel/distributed_vector_ops.cpp
    src/parallel/distributed_sparse_matrix.cpp
    src/common/givens.cpp
    src/common/io.cpp
    src/common/sparse_matrix.cpp
)

target_include_directories(run_gmres_ca_mpi_experiment PRIVATE include)

target_link_libraries(run_gmres_ca_mpi_experiment
    PRIVATE
    MPI::MPI_CXX
    stdc++exp
)
