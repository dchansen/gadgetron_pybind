cmake_minimum_required(VERSION 3.15)
project(gadgetron_pybind)

set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)
pybind11_add_module(library library.cpp)
