#pragma once

#include <gadgetron/hoNDArray.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace Gadgetron {
    namespace Python {

        template<class T>
        hoNDArray<T> to_ndarray(py::array_t<T,py::array::f_style | py::array::forcecast> array){

            std::vector<size_t> dims(array.ndim());
            for (size_t i = 0 ; i < dims.size(); i++){
                dims[i] = array.shape(i);
            }

            auto result = hoNDArray<T>(dims);
            std::copy_n(array.data(),array.size(),result.begin());
            return result;
        }
        template<class T>
        py::array_t<T> to_pyarray(const hoNDArray<T>& array){

        }


    }

}
void hello();

