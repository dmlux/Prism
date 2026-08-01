cmake_minimum_required(VERSION 3.24)
project(prism-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(prism_segmentation src/segmentation.cpp)
target_include_directories(prism_segmentation PUBLIC include)
target_compile_options(prism_segmentation PRIVATE -Wall -Wextra)

enable_testing()
add_executable(segmentation_tests tests/segmentation_tests.cpp)
target_link_libraries(segmentation_tests PRIVATE prism_segmentation)
add_test(NAME segmentation_tests COMMAND segmentation_tests ${CMAKE_CURRENT_SOURCE_DIR}/..)
