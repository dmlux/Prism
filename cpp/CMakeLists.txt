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

add_library(prism_artifact src/artifact.cpp)
target_include_directories(prism_artifact PUBLIC include)
target_link_libraries(prism_artifact PRIVATE prism_vendor_json)
target_compile_options(prism_artifact PRIVATE -Wall -Wextra)

# The ExecuTorch runtime cannot be vendored as an amalgamation (the source
# tree with submodules spans hundreds of megabytes), so the engine pins it
# to the exact exporter version via FetchContent and stays optional: a
# checkout without network access still builds everything else.
option(PRISM_ENGINE "Build the ExecuTorch engine" ON)
if(PRISM_ENGINE)
    include(FetchContent)
    # ExecuTorch locates the torch headers through the Python interpreter;
    # prefer the repository environment, which has the pinned torch install.
    if(NOT DEFINED Python3_EXECUTABLE
        AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
        set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
    endif()
    set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        executorch
        GIT_REPOSITORY https://github.com/pytorch/executorch.git
        GIT_TAG v1.3.1
        GIT_SHALLOW TRUE
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/executorch"
    )
    FetchContent_MakeAvailable(executorch)

    add_library(prism_engine src/engine.cpp)
    target_include_directories(prism_engine PUBLIC include)
    target_link_libraries(prism_engine
        PRIVATE executorch extension_module_static extension_tensor
                extension_data_loader extension_threadpool
    )
    # Kernel and backend registration happens in static initializers, so the
    # final binary must keep every archive member.
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
    executorch_target_link_options_shared_lib(xnnpack_backend)
    target_link_libraries(prism_engine PRIVATE optimized_native_cpu_ops_lib xnnpack_backend)

    add_library(prism_tagger src/tagger.cpp)
    target_include_directories(prism_tagger PUBLIC include)
    target_link_libraries(prism_tagger
        PUBLIC prism_artifact prism_segmentation
        PRIVATE prism_subword prism_engine
    )
    target_compile_options(prism_tagger PRIVATE -Wall -Wextra)

    # The plain-C embedding surface for applications whose core links C
    # libraries only (see include/prism/prism_c.h).
    add_library(prism_c src/prism_c.cpp)
    target_include_directories(prism_c PUBLIC include)
    target_link_libraries(prism_c PRIVATE prism_tagger)
    target_compile_options(prism_c PRIVATE -Wall -Wextra)

    add_executable(prism_chapter_benchmark tools/chapter_benchmark.cpp)
    target_link_libraries(prism_chapter_benchmark PRIVATE prism_tagger)
    target_compile_options(prism_chapter_benchmark PRIVATE -Wall -Wextra)
endif()

# The Java binding: a thin JNI bridge over the C++ tagger plus the
# dependency-free Java layer in java/. The JAR is also buildable with
# Maven (java/pom.xml) for Maven/Gradle consumers; this block is the
# repository's canonical build and drives the test suite.
option(PRISM_JAVA "Build the Java (JNI) binding" ON)
if(PRISM_ENGINE AND PRISM_JAVA)
    find_package(JNI)
    find_package(Java 21 COMPONENTS Development)
    if(JNI_FOUND AND Java_FOUND)
        include(UseJava)

        add_library(prism_jni SHARED src/jni.cpp)
        target_include_directories(prism_jni PRIVATE ${JNI_INCLUDE_DIRS})
        target_link_libraries(prism_jni PRIVATE prism_tagger)
        target_compile_options(prism_jni PRIVATE -Wall -Wextra)

        file(GLOB PRISM_JAVA_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/../java/src/main/java/io/github/dmlux/prism/*.java")
        add_jar(prism_java SOURCES ${PRISM_JAVA_SOURCES} OUTPUT_NAME prism)
        add_jar(prism_java_test
            SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/../java/src/test/java/io/github/dmlux/prism/PrismTaggerTest.java"
                "${CMAKE_CURRENT_SOURCE_DIR}/../java/src/test/java/io/github/dmlux/prism/PrismChapterBenchmark.java"
            INCLUDE_JARS prism_java
        )
    else()
        message(STATUS "Prism: JNI or a Java 21 JDK is missing; skipping the Java binding.")
    endif()
endif()

# Aggregate convenience target: link `prism`, then `#include <prism>`
# pulls in the complete public API. The umbrella header is an extensionless
# file, so it lives in its own include directory next to the per-module
# headers (a file cannot share the name of the include/prism/ directory).
add_library(prism INTERFACE)
target_include_directories(prism INTERFACE include umbrella)
target_link_libraries(prism INTERFACE prism_segmentation prism_subword prism_artifact)
if(PRISM_ENGINE)
    target_link_libraries(prism INTERFACE prism_engine prism_tagger prism_c)
endif()

enable_testing()

# Tests receive the repository root for the local, untracked fixtures.
add_executable(prism_tests
    tests/segmentation_tests.cpp
    tests/subword_tests.cpp
    tests/umbrella_tests.cpp
)
target_link_libraries(prism_tests PRIVATE prism prism_vendor_json prism_vendor_gtest)
if(PRISM_ENGINE)
    target_sources(prism_tests PRIVATE tests/engine_tests.cpp tests/tagger_tests.cpp)
endif()
target_compile_definitions(prism_tests PRIVATE PRISM_REPOSITORY_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME prism_tests COMMAND prism_tests)

if(TARGET prism_java_test)
    add_test(NAME prism_java_tests
        COMMAND ${Java_JAVA_EXECUTABLE}
            -Djava.library.path=$<TARGET_FILE_DIR:prism_jni>
            -cp "$<TARGET_PROPERTY:prism_java,JAR_FILE>:$<TARGET_PROPERTY:prism_java_test,JAR_FILE>"
            io.github.dmlux.prism.PrismTaggerTest
            "${CMAKE_CURRENT_SOURCE_DIR}/../models/prism-no-0.2.2"
    )
endif()
