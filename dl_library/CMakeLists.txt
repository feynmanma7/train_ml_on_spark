cmake_minimum_required(VERSION 3.14)
project(torch_lite_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 (must be installed: pip install pybind11)
find_package(pybind11 REQUIRED)

# Library source
pybind11_add_module(_torch_lite_core
    csrc/tensor.cpp
    csrc/bindings.cpp
)

target_include_directories(_torch_lite_core PRIVATE csrc)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(_torch_lite_core PRIVATE -O3 -march=native -ffast-math)
endif()
