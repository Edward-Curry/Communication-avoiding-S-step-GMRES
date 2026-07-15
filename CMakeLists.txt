cmake_minimum_required(VERSION 3.16)

project(CommunicationAvoidingSstepGMRES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)

find_path(GMRES_CBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATH_SUFFIXES openblas
)

find_path(GMRES_LAPACKE_INCLUDE_DIR
    NAMES lapacke.h
    PATH_SUFFIXES openblas lapacke
)

if(NOT GMRES_CBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "BLAS was found, but cblas.h was not found.")
endif()

if(NOT GMRES_LAPACKE_INCLUDE_DIR)
    message(FATAL_ERROR "BLAS was found, but lapacke.h was not found.")
endif()

add_library(gmres_blas INTERFACE)
target_include_directories(gmres_blas INTERFACE
    ${GMRES_CBLAS_INCLUDE_DIR}
    ${GMRES_LAPACKE_INCLUDE_DIR}
)
target_link_libraries(gmres_blas INTERFACE
    ${BLAS_LIBRARIES}
)

set(GMRES_BCGS2_SHARED_SOURCES
    src/common/dense_block.cpp
    src/communication_avoiding/partial_cholesky.cpp
)

set(GMRES_BCGS2_SEQUENTIAL_SOURCES
    ${GMRES_BCGS2_SHARED_SOURCES}
    src/communication_avoiding/bcgs.cpp
    src/communication_avoiding/cholqr.cpp
    src/communication_avoiding/bcgs2_cholqr.cpp
)

set(GMRES_BCGS2_MPI_SOURCES
    ${GMRES_BCGS2_SHARED_SOURCES}
    src/parallel/distributed_dense_block.cpp
    src/communication_avoiding/bcgs_mpi.cpp
    src/communication_avoiding/cholqr_mpi.cpp
    src/communication_avoiding/bcgs2_cholqr_mpi.cpp
)

set(GMRES_CA_SHARED_SOURCES
    src/communication_avoiding/hessenberg_assembly.cpp
)

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

add_executable(test_distributed_sparse_matrix
    tests/test_distributed_sparse_matrix.cpp
    src/parallel/distributed_sparse_matrix.cpp
    src/parallel/distributed_vector.cpp
)

target_include_directories(test_distributed_sparse_matrix PRIVATE include)
target_link_libraries(test_distributed_sparse_matrix PRIVATE MPI::MPI_CXX stdc++exp)

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
    ${GMRES_CA_SHARED_SOURCES}
    ${GMRES_BCGS2_SEQUENTIAL_SOURCES}
    src/communication_avoiding/krylov_basis.cpp
    src/communication_avoiding/polynomial_basis.cpp
    src/common/givens.cpp
    src/common/vector_ops.cpp
    src/common/sparse_matrix.cpp
)

target_include_directories(test_gmres_ca PRIVATE include)
target_link_libraries(test_gmres_ca PRIVATE stdc++exp)

add_executable(test_bcgs2_cholqr
    tests/test_bcgs2_cholqr.cpp
    ${GMRES_BCGS2_SEQUENTIAL_SOURCES}
    src/common/vector_ops.cpp
)

target_include_directories(test_bcgs2_cholqr PRIVATE include)
target_link_libraries(test_bcgs2_cholqr PRIVATE stdc++exp)

add_executable(test_sstep_arnoldi_mpi
    tests/test_sstep_arnoldi_mpi.cpp
    src/communication_avoiding/sstep_arnoldi_mpi.cpp
    src/communication_avoiding/krylov_basis_mpi.cpp
    src/communication_avoiding/block_orthogonalization_mpi.cpp
    ${GMRES_BCGS2_MPI_SOURCES}
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
    ${GMRES_CA_SHARED_SOURCES}
    ${GMRES_BCGS2_MPI_SOURCES}
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
    ${GMRES_CA_SHARED_SOURCES}
    ${GMRES_BCGS2_SEQUENTIAL_SOURCES}
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
    ${GMRES_CA_SHARED_SOURCES}
    ${GMRES_BCGS2_MPI_SOURCES}
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

set(GMRES_BLAS_TARGETS
    test_vector_ops
    test_orthogonalization
    test_arnoldi_seq
    test_gmres_seq
    test_gmres_mpi
    test_gmres_ca
    test_bcgs2_cholqr
    test_sstep_arnoldi_mpi
    test_gmres_ca_mpi
    run_gmres_experiments
    run_gmres_ca_experiment
    run_gmres_mpi_experiment
    run_gmres_ca_mpi_experiment
)

foreach(target IN LISTS GMRES_BLAS_TARGETS)
    target_link_libraries(${target} PRIVATE gmres_blas)
endforeach()
