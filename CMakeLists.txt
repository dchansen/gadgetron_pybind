cmake_minimum_required(VERSION 3.15)
project(gadgetron_pybind)

set(CMAKE_CXX_STANDARD 17)

include_directories(  "/home/dchansen/.gadgetron/include")
find_package(pybind11 REQUIRED)
find_package(gadgetron REQUIRED)
pybind11_add_module(gadgetron_toolbox MODULE demons.cpp)
target_link_libraries(gadgetron_toolbox PRIVATE Gadgetron::gadgetron_toolbox_demons Gadgetron::gadgetron_toolbox_cpucore Gadgetron::gadgetron_toolbox_cpucore_math Gadgetron::gadgetron_toolbox_t1)
