cmake_minimum_required(VERSION 3.24)
project(prism-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(prism_segmentation src/segmentation.cpp)
target_include_directories(prism_segmentation PUBLIC include)
target_compile_options(prism_segmentation PRIVATE -Wall -Wextra)

add_library(prism_subword src/subword.cpp)
target_include_directories(prism_subword PUBLIC include)
target_link_libraries(prism_subword PUBLIC prism_segmentation PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(prism_subword PRIVATE -Wall -Wextra)

enable_testing()
add_executable(segmentation_tests tests/segmentation_tests.cpp)
target_link_libraries(segmentation_tests PRIVATE prism_segmentation)
add_test(NAME segmentation_tests COMMAND segmentation_tests ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(subword_tests tests/subword_tests.cpp)
target_link_libraries(subword_tests PRIVATE prism_subword nlohmann_json::nlohmann_json)
add_test(NAME subword_tests COMMAND subword_tests ${CMAKE_CURRENT_SOURCE_DIR}/..)
