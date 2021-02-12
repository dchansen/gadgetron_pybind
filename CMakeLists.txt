cmake_minimum_required(VERSION 3.10)
project(gadgetron_pybind)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(gadgetron CONFIG REQUIRED)
pybind11_add_module(gadgetron_toolbox MODULE demons.cpp)
target_link_libraries(gadgetron_toolbox PRIVATE Gadgetron::gadgetron_toolbox_demons Gadgetron::gadgetron_toolbox_cpucore Gadgetron::gadgetron_toolbox_cpucore_math Gadgetron::gadgetron_toolbox_t1 Gadgetron::gadgetron_toolbox_denoise)
