cmake_minimum_required(VERSION 3.25)
project(xpu-flash-attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(XFA_BUILD_TESTS "Build tests" ON)
option(XFA_BUILD_BENCHMARKS "Build benchmarks" ON)
option(XFA_BUILD_PYTHON "Build Python bindings" ON)
option(XFA_SYCL "Enable SYCL compilation" ON)

if(XFA_SYCL)
  # JIT (default) consistently beat AOT for our SIMT kernels — the JIT path
  # avoids the heavy register-spill the AOT graphics compiler produces.
  # Set XFA_SYCL_TARGET=intel_gpu_bmg_g31 (or similar) only if you need AOT
  # for a specific reason (e.g. once joint_matrix lowering for Xe2 lands —
  # see results/xmx_investigation.md).
  set(XFA_SYCL_TARGET "" CACHE STRING "AOT SYCL target arch (empty = JIT)")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")
  if(NOT XFA_SYCL_TARGET STREQUAL "")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=${XFA_SYCL_TARGET}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl-targets=${XFA_SYCL_TARGET}")
  endif()

  find_package(LevelZero QUIET)
  if(NOT LevelZero_FOUND)
    find_library(LEVEL_ZERO_LIB ze_loader HINTS ENV LEVEL_ZERO_LIB_PATH)
    find_path(LEVEL_ZERO_INCLUDE level_zero/ze_api.h HINTS ENV LEVEL_ZERO_INCLUDE_PATH)
  endif()
endif()

# header-only kernel library
add_library(xpu_flash_attention INTERFACE)
target_include_directories(xpu_flash_attention INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

if(SYCL_TLA_DIR)
  target_include_directories(xpu_flash_attention INTERFACE ${SYCL_TLA_DIR}/include)
endif()

if(XFA_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(XFA_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(XFA_BUILD_PYTHON)
  find_package(pybind11 QUIET)
  if(pybind11_FOUND)
    add_subdirectory(src/bindings)
  else()
    message(STATUS "pybind11 not found, skipping Python bindings")
  endif()
endif()
