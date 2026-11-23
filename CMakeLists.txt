cmake_minimum_required(VERSION 3.16)
project(cpp_parallel_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PE_PROFILE "Build with gprof instrumentation (-pg) and debug info" OFF)
set(PE_SANITIZE "" CACHE STRING "Sanitizers, e.g. 'address,undefined' or 'thread'")

# -ffp-contract=off : no fused multiply-add, so scalar and SIMD kernels round identically.
# -fno-math-errno   : lets sqrt compile to a single instruction (and be vectorised).
# -fopenmp-simd     : honours "#pragma omp simd" (portable vectorisation hint; no OpenMP runtime).
set(PE_FLAGS -Wall -Wextra -ffp-contract=off -fno-math-errno -fopenmp-simd)
if(PE_PROFILE)
  list(APPEND PE_FLAGS -pg -g -fno-omit-frame-pointer)
  add_link_options(-pg)
endif()

if(PE_SANITIZE)
  list(APPEND PE_FLAGS -fsanitize=${PE_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${PE_SANITIZE})
endif()

find_package(Threads REQUIRED)

add_library(pe STATIC src/kernels.cpp src/transpose.cpp src/records.cpp)
target_include_directories(pe PUBLIC include)
target_compile_options(pe PRIVATE ${PE_FLAGS})
target_link_libraries(pe PUBLIC Threads::Threads)

add_executable(pe_bench bench/bench.cpp)
target_link_libraries(pe_bench PRIVATE pe)
target_compile_options(pe_bench PRIVATE ${PE_FLAGS})

enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
  add_executable(pe_tests tests/test_kernels.cpp tests/test_thread_pool.cpp tests/test_pool.cpp
                          tests/test_transpose.cpp)
  target_link_libraries(pe_tests PRIVATE pe GTest::gtest_main)
  target_compile_options(pe_tests PRIVATE ${PE_FLAGS})
  add_test(NAME pe_tests COMMAND pe_tests)
else()
  message(STATUS "GTest not found: tests disabled (apt install libgtest-dev / brew install googletest)")
endif()
