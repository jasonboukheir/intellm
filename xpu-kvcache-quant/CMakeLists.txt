cmake_minimum_required(VERSION 3.25)
project(xpu-kvcache-quant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KVQ_BUILD_TESTS "Build tests" ON)
option(KVQ_BUILD_BENCHMARKS "Build benchmarks" ON)
option(KVQ_BUILD_PYTHON "Build Python bindings" ON)
option(KVQ_SYCL "Enable SYCL compilation" ON)

if(KVQ_SYCL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")
endif()

# Header-only kernel libraries
add_library(turboquant INTERFACE)
target_include_directories(turboquant INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

add_library(rotorquant INTERFACE)
target_include_directories(rotorquant INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

add_library(kvcache_quant INTERFACE)
target_link_libraries(kvcache_quant INTERFACE turboquant rotorquant)

if(SYCL_TLA_DIR)
  target_include_directories(turboquant INTERFACE ${SYCL_TLA_DIR}/include)
  target_include_directories(rotorquant INTERFACE ${SYCL_TLA_DIR}/include)
endif()

if(KVQ_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(KVQ_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(KVQ_BUILD_PYTHON)
  find_package(pybind11 QUIET)
  if(pybind11_FOUND)
    add_subdirectory(src/bindings)
  else()
    message(STATUS "pybind11 not found, skipping Python bindings")
  endif()
endif()
