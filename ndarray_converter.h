#pragma once

#include <gadgetron/hoNDArray.h>
#include <gadgetron/vector_td.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace pybind11 {
namespace detail {
template<class T, unsigned int D> struct type_caster<Gadgetron::hoNDArray<Gadgetron::vector_td<T,D>>> {
 public:
  using hoNDArrayV = Gadgetron::hoNDArray<Gadgetron::vector_td<T,D>>;
 PYBIND11_TYPE_CASTER(hoNDArrayV,_("hoNDArrayV"));

  bool load(py::handle src, bool convert){

    if (!convert && !isinstance<array_t<T>>(src))
      return false;

    auto buf  = array_t<T,py::array::c_style | py::array::forcecast>::ensure(src);
    if (!buf) return false;

    if (buf.shape(buf.ndim()-1) != D) return false;


    std::vector<size_t> dims(buf.ndim()-1);


    for (size_t i = 0 ; i < dims.size(); i++){
      dims[i] = buf.shape(buf.ndim()-i-2);
    }

    value = Gadgetron::hoNDArray<Gadgetron::vector_td<T,D>>(dims);
    std::copy_n(buf.data(),buf.size(),(T*)value.data());

    return true;
  }


  static handle cast(const Gadgetron::hoNDArray<Gadgetron::vector_td<T,D>>& data, return_value_policy, handle parent ){
    std::vector<size_t> rev_dims(data.dimensions().rbegin(),data.dimensions().rend());
    rev_dims.push_back(D);
    array_t<T, py::array::c_style | py::array::forcecast> pyarray(rev_dims);
    std::copy_n((const T*)data.data(),data.get_number_of_elements()*D,pyarray.mutable_data());
    return pyarray.release();
  }

};
template<class T> struct type_caster<Gadgetron::hoNDArray<T>> {
 public:
  PYBIND11_TYPE_CASTER(Gadgetron::hoNDArray<T>,_("hoNDArray"));

  bool load(py::handle src, bool convert){

    if (!convert && !isinstance<array_t<T>>(src))
      return false;

    auto buf  = array_t<T,py::array::c_style | py::array::forcecast>::ensure(src);
    if (!buf) return false;

    std::vector<size_t> dims(buf.ndim());
    for (size_t i = 0 ; i < dims.size(); i++){
      dims[i] = buf.shape(buf.ndim()-i-1);
    }

    value = Gadgetron::hoNDArray<T>(dims);
    std::copy_n(buf.data(),buf.size(),value.data());

    return true;
  }


  static handle cast(const Gadgetron::hoNDArray<T>& data, return_value_policy, handle parent ){

    std::vector<size_t> rev_dims(data.dimensions().rbegin(),data.dimensions().rend());
    array_t<T, py::array::c_style | py::array::forcecast> pyarray(rev_dims);
    std::copy(data.begin(),data.end(),pyarray.mutable_data());
    return pyarray.release();

  }

};
}
}
