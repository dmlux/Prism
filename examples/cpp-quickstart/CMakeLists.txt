# Minimal C++ consumer: fetch a pinned Prism release, link the aggregate
# `prism` target — headers, ExecuTorch, XNNPACK, and the quantized
# kernels all arrive transitively.
cmake_minimum_required(VERSION 3.24)
project(prism_cpp_quickstart CXX)

include(FetchContent)
FetchContent_Declare(
    prism
    GIT_REPOSITORY https://github.com/dmlux/Prism.git
    GIT_TAG v0.5.0
    SOURCE_SUBDIR cpp
)
FetchContent_MakeAvailable(prism)

add_executable(quickstart main.cpp)
target_link_libraries(quickstart PRIVATE prism)
