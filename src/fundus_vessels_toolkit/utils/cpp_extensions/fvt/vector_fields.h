#ifndef VECTOR_FIELDS_H
#define VECTOR_FIELDS_H

#include "common.h"

torch::Tensor inverse_displacement(const torch::Tensor& disp_field, const torch::Tensor& points,
                                   const int max_iters = 50, const float sqr_tol = 0.5);
torch::Tensor vec_bilinear_interpolate(const torch::Tensor& vector_field, const torch::Tensor& points);

Point _vec_bilinear_interpolate(const Tensor3DAcc<float>& vec_acc, const Point& p, const IntPoint& field_size);

torch::Tensor smooth_binary_mask(const torch::Tensor& mask, float sigma = 1.0, const float tol = 1e-3);

#endif  // VECTOR_FIELDS_H