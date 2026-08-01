cmake_minimum_required(VERSION 3.24)
project(prism-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored, version-pinned third-party code (no network at configure time):
# - vendor/nlohmann: json.hpp single-header amalgamation v3.11.3 (MIT)
# - vendor/googletest: GoogleTest v1.17.0 sources (BSD-3-Clause)
add_library(prism_vendor_json INTERFACE)
target_include_directories(prism_vendor_json INTERFACE vendor)

add_library(prism_vendor_gtest STATIC
    vendor/googletest/src/gtest-all.cc
    vendor/googletest/src/gtest_main.cc
)
target_include_directories(prism_vendor_gtest
    PUBLIC vendor/googletest/include
    PRIVATE vendor/googletest
)

add_library(prism_segmentation src/segmentation.cpp)
target_include_directories(prism_segmentation PUBLIC include)
target_compile_options(prism_segmentation PRIVATE -Wall -Wextra)

add_library(prism_subword src/subword.cpp)
target_include_directories(prism_subword PUBLIC include)
target_link_libraries(prism_subword PUBLIC prism_segmentation PRIVATE prism_vendor_json)
target_compile_options(prism_subword PRIVATE -Wall -Wextra)

enable_testing()

# Tests receive the repository root for the local, untracked fixtures.
add_executable(prism_tests
    tests/segmentation_tests.cpp
    tests/subword_tests.cpp
)
target_link_libraries(prism_tests PRIVATE prism_segmentation prism_subword prism_vendor_json prism_vendor_gtest)
target_compile_definitions(prism_tests PRIVATE PRISM_REPOSITORY_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME prism_tests COMMAND prism_tests)
