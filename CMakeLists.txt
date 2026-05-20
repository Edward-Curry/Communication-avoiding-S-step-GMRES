cmake_minimum_required(VERSION 3.16)

project(CommunicationAvoidingSstepGMRES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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

add_executable(test_vector_ops
    tests/test_vector_ops.cpp
    src/common/vector_ops.cpp
)

target_include_directories(test_vector_ops PRIVATE include)