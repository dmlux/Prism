# Minimal C consumer of the stable C ABI. The tagger core is C++, so
# CMake links the executable with the C++ toolchain; the example source
# itself is plain C99 against <prism/prism_c.h>.
cmake_minimum_required(VERSION 3.24)
project(prism_c_quickstart C CXX)

include(FetchContent)
FetchContent_Declare(
    prism
    GIT_REPOSITORY https://github.com/dmlux/Prism.git
    GIT_TAG v0.4.0
    SOURCE_SUBDIR cpp
)
FetchContent_MakeAvailable(prism)

add_executable(quickstart main.c)
target_link_libraries(quickstart PRIVATE prism_c)
set_target_properties(quickstart PROPERTIES LINKER_LANGUAGE CXX)
