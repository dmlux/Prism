# Minimal C++ consumer: fetch a pinned Prism release, link the aggregate
# `prism` target — headers, ExecuTorch, XNNPACK, and the quantized
# kernels all arrive transitively.
cmake_minimum_required(VERSION 3.24)
project(prism_cpp_quickstart CXX)

# Prism's public headers use std::filesystem and std::optional; the
# v0.4.0 target does not yet export this requirement to consumers.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    prism
    GIT_REPOSITORY https://github.com/dmlux/Prism.git
    GIT_TAG v0.4.0
    SOURCE_SUBDIR cpp
)
FetchContent_MakeAvailable(prism)

add_executable(quickstart main.cpp)
target_link_libraries(quickstart PRIVATE prism)
