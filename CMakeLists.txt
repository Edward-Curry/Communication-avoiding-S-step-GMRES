cmake_minimum_required(VERSION 3.16)

project(CommunicationAvoidingSstepGMRES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(run_seq
    app/run_seq.cpp
)