#include "ndarray_converter.h"

#include <gadgetron/hoNDArray.h>
#include <gadgetron/demons_registration.h>
#include <gadgetron/t1fit.h>
#include <gadgetron/hoNDArray_utils.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <gadgetron/cmr_motion_correction.h>
#include <gadgetron/non_local_bayes.h>

using namespace Gadgetron;
//
//hoNDArray<float> crazy_sum(const hoNDArray<std::complex<float>>& array){
//  hoNDArray<float> result(array.dimensions());
//
//  std::transform(array.begin(),array.end(),result.begin(),[](const auto& val){ return val.real() + val.imag();});
//  return result;
//}
hoNDArray<vector_td<float, 2>> cmr_registration(const hoNDArray<float> &fixed, const hoNDArray<float> &moving, float hilbert_strength)
{
  using ImageType = hoNDImage<float, 2>;
  using RegType = Gadgetron::hoImageRegContainer2DRegistration<ImageType, ImageType, double>;

  RegType reg;

  std::vector<unsigned int> iters = {32, 64, 100, 100}; //Stolen from MocoSASAH

  perform_moco_pair_wise_frame_2DT(fixed, moving, hilbert_strength, iters, true, false, reg);

  hoNDArray<double> dx;
  hoNDArray<double> dy;
  reg.deformation_field_[0].to_NDArray(0, dx);
  reg.deformation_field_[1].to_NDArray(0, dy);

  //hoNDArray<float> output = fixed;
  //apply_deformation_field(moving,dx,dy,output);

  hoNDArray<vector_td<float, 2>> vfield(dx.dimensions());

  for (int64_t i = 0; i < vfield.size(); i++)
  {
    vfield[i] = vector_td<float, 2>(dx[i], dy[i]);
  }

  return vfield;
}
hoNDArray<vector_td<float, 2>> cmr_registration_MI(const hoNDArray<float> &fixed, const hoNDArray<float> &moving, float hilbert_strength)
{
  using ImageType = hoNDImage<float, 2>;
  using RegType = Gadgetron::hoImageRegContainer2DRegistration<ImageType, ImageType, double>;

  RegType reg;
  reg.dissimilarity_type_ = GT_IMAGE_DISSIMILARITY::GT_IMAGE_DISSIMILARITY_NMI;

  std::vector<unsigned int> iters = {32, 64, 100, 100}; //Stolen from MocoSASAH

  perform_moco_pair_wise_frame_2DT(fixed, moving, hilbert_strength, iters, true, false, reg);

  hoNDArray<double> dx;
  hoNDArray<double> dy;
  reg.deformation_field_[0].to_NDArray(0, dx);
  reg.deformation_field_[1].to_NDArray(0, dy);

  //hoNDArray<float> output = fixed;
  //apply_deformation_field(moving,dx,dy,output);

  hoNDArray<vector_td<float, 2>> vfield(dx.dimensions());

  for (int64_t i = 0; i < vfield.size(); i++)
  {
    vfield[i] = vector_td<float, 2>(dx[i], dy[i]);
  }

  return vfield;
}

hoNDArray<vector_td<float, 2>> diff_demons(const hoNDArray<float> &fixed, const hoNDArray<float> &moving, int iterations, float sigma)
{
  return Gadgetron::Registration::diffeomorphic_demons<float, 2>(fixed, moving, iterations, sigma);
}

hoNDArray<vector_td<float, 2>> ngf_demons(const hoNDArray<float> &fixed, const hoNDArray<float> &moving, int iterations, float sigma)
{
  return Gadgetron::Registration::ngf_diffeomorphic_demons<float, 2>(fixed, moving, iterations, sigma);
}
hoNDArray<vector_td<float, 2>> diff_demons_start(const hoNDArray<float> &fixed, const hoNDArray<float> &moving, hoNDArray<vector_td<float, 2>> vfield, int iterations, float sigma)
{
  return Gadgetron::Registration::diffeomorphic_demons<float, 2>(fixed, moving, vfield, iterations, sigma);
}
hoNDArray<vector_td<float, 2>> multi_diff_demons(const hoNDArray<float> &fixed, const hoNDArray<float> &moving, int levels, int iterations, float sigma)
{
  return Gadgetron::Registration::multi_scale_diffeomorphic_demons<float, 2>(fixed, moving, levels, iterations, sigma);
}
hoNDArray<vector_td<float, 2>> multi_diff_ngf_demons(const hoNDArray<float> &fixed, const hoNDArray<float> &moving, int levels, int iterations, float sigma)
{
  return Gadgetron::Registration::multi_scale_ngf_diffeomorphic_demons<float, 2>(fixed, moving, levels, iterations, sigma);
}

std::tuple<hoNDArray<float>, hoNDArray<float>> t1_2param(const hoNDArray<float> &data, const std::vector<float> &TI)
{
  auto [A, T1] = T1::fit_T1_2param(data, TI);
  return {A, T1};
}
std::tuple<hoNDArray<float>, hoNDArray<float>> t1_2param_auto_phase(const hoNDArray<std::complex<float>> &data, const std::vector<float> &TI)
{
  auto [A, T1] = T1::fit_T1_2param(data, TI);
  return {A, T1};
}
std::tuple<hoNDArray<float>, hoNDArray<float>, hoNDArray<float>> t1_3param(const hoNDArray<float> &data, const std::vector<float> &TI)
{
  auto [A, B, T1] = T1::fit_T1_3param(data, TI);
  return {A, B, T1};
}

std::tuple<hoNDArray<float>, hoNDArray<float>, hoNDArray<float>> t1_3param_auto_phase(const hoNDArray<std::complex<float>> &data, const std::vector<float> &TI)
{
  auto [A, B, T1] = T1::fit_T1_3param(data, TI);
  return {A, B, T1};
}

hoNDArray<float> predict_signal_2param(const hoNDArray<float> &A, const hoNDArray<float> &T1, const std::vector<float> &TI)
{
  return T1::predict_signal(T1::T1_2param{A, T1}, TI);
}

hoNDArray<vector_td<float, 2>> t1_registration(const hoNDArray<std::complex<float>> &data, const std::vector<float> &TI, unsigned int iterations, unsigned int demons_iterations, float demons_sigma, float step_size)
{
  return T1::t1_registration(data, TI, iterations, {demons_iterations, demons_sigma, step_size});
}

hoNDArray<vector_td<float, 2>> t1_registration_cmr(const hoNDArray<std::complex<float>> &data, const std::vector<float> &TI, unsigned int iterations)
{
  return T1::t1_moco_cmr(data, TI, iterations);
}

hoNDArray<float> non_local_bayes_float(const hoNDArray<float>& data, float noise, unsigned int search_radius){
  return Denoise::non_local_bayes(data,noise,search_radius);
}
hoNDArray<std::complex<float>> non_local_bayes_cplx(const hoNDArray<std::complex<float>>& data, float noise, unsigned int search_radius){
  return Denoise::non_local_bayes(data,noise,search_radius);
}



PYBIND11_MODULE(gadgetron_toolbox, m)
{
  m.doc() = "Registration module";
  m.def("demons", &diff_demons, "Registers an image");
  m.def("ngf_demons", &ngf_demons, "Registers an image");
  m.def("multi_demons", &multi_diff_demons, "Registers an image");
  m.def("multi_ngf_demons", &multi_diff_ngf_demons, "Registers an image");
  m.def("demons_start", &diff_demons_start, "Registers an image");
  m.def("deform_image", &Gadgetron::Registration::deform_image<float, 2>, "Registers an image");
  m.def("deform_image_bspline", &Gadgetron::Registration::deform_image_bspline<float, 2>, "Registers an image");
  m.def("deform_vfield", &Gadgetron::Registration::deform_image<vector_td<float, 2>, 2, float>, "Deforms a vector field");
  m.def("fit_T1_2param", &t1_2param, "T1 2 parameter fit");
  m.def("fit_T1_2param_auto_phase", &t1_2param_auto_phase, "T1 2 parameter fit");
  m.def("fit_T1_3param", &t1_3param, "T1 3 parameter fit");
  m.def("fit_T1_3param_auto_phase", &t1_3param_auto_phase, "T1 3 parameter fit");
  m.def("gaussian_filter", &Gadgetron::Registration::gaussian_filter<float>, "Performs gaussian filter");
  m.def("gaussian_filter_vfield", &Gadgetron::Registration::gaussian_filter<vector_td<float, 2>>, "Performs gaussian filter");
  m.def("compose_vfield", &Gadgetron::Registration::compose_fields<float, 2>, "Composes vector fields");
  m.def("vector_field_exponential", &Gadgetron::Registration::vector_field_exponential<float, 2>, "Takes the exponential of a vector field");
  m.def("demons_step", &Gadgetron::Registration::demons_step_ext, "Single demons step");
  m.def("predict_signal", &predict_signal_2param, "Predicts signal");
  m.def("phase_correct", &Gadgetron::T1::phase_correct, "Phase correction");
  m.def("t1_registration", &t1_registration, "");
  m.def("t1_registration_cmr", &t1_registration_cmr, "");
  m.def("deform_groups", py::overload_cast<const hoNDArray<float>&, const hoNDArray<vector_td<float,2>>&>(&Gadgetron::T1::deform_groups), "");
  m.def("deform_groups", py::overload_cast<const hoNDArray<std::complex<float>>&, const hoNDArray<vector_td<float,2>>&>(&Gadgetron::T1::deform_groups), "");
  m.def("upsample", &upsample<float, 2>, "Upsamples image");
  m.def("downsample", &downsample<float, 2>, "Downsamples image");
  m.def("cmr_registration", &cmr_registration, "CMR based registration");
  m.def("cmr_registration_MI", &cmr_registration_MI, "CMR based registration using mutual information");
  m.def("non_local_bayes",non_local_bayes_float, "Non local Bayes denoising");
  m.def("non_local_bayes",non_local_bayes_cplx, "Non local Bayes denoising");
}