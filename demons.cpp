#include "ndarray_converter.h"

#include <gadgetron/hoNDArray.h>
#include <gadgetron/demons_registration.h>
#include <gadgetron/t1fit.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
using namespace Gadgetron;
//
//hoNDArray<float> crazy_sum(const hoNDArray<std::complex<float>>& array){
//  hoNDArray<float> result(array.dimensions());
//
//  std::transform(array.begin(),array.end(),result.begin(),[](const auto& val){ return val.real() + val.imag();});
//  return result;
//}



std::tuple<hoNDArray<float>,hoNDArray<float>,hoNDArray<float>> moco_t1_fit(const hoNDArray<std::complex<float>>& data, const std::vector<float>& TI, unsigned int iterations){

  auto [A,B,T1star] = Gadgetron::T1::motion_compensated_t1_fit(data,TI,iterations);

  return {A,B,T1star};

}


PYBIND11_MODULE(gadgetron_toolbox,m){
 m.doc() = "Registration module";
 m.def("demons",&Gadgetron::Registration::diffeomorphic_demons<float,2>, "Registers an image");
  m.def("deform_image",&Gadgetron::Registration::deform_image<float,2>, "Registers an image");
  m.def("motion_compensated_t1_fit",moco_t1_fit);
}