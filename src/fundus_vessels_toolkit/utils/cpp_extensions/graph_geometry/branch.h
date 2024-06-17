#ifndef BRANCH_TRACKING_H
#define BRANCH_TRACKING_H

#include "../common.h"

/**************************************************************************************
 *              === BRANCH_TRACKING.CPP ===
 **************************************************************************************/

std::vector<std::array<IntPoint, 2>> find_branch_endpoints(const torch::Tensor &branch_labels,
                                                           const torch::Tensor &node_yx,
                                                           const torch::Tensor &branch_list);

std::vector<CurveYX> track_branches(const torch::Tensor &branch_labels, const torch::Tensor &node_yx,
                                    const torch::Tensor &branch_list);

IntPoint track_nearest_border(const IntPoint &start, const Point &direction, const Tensor2DAccessor<bool> &segmentation,
                              int max_distance = 40);

std::tuple<int, float> findClosestPixel(const CurveYX &curve, const Point &p, int start, int end,
                                        bool findFirstLocalMinimum = false);

/**************************************************************************************
 *              === BRANCH_CURVES.CPP ===
 **************************************************************************************/
Point curve_tangent(const CurveYX &curveYX, std::size_t i, const std::function<float(float)> &weighting,
                    const bool forward = true, const bool backward = true, std::size_t stride = 1);
std::vector<Point> curve_tangent(const CurveYX &curveYX, const std::vector<float> &kernelStd,
                                 const std::vector<int> &evaluateAtID = {});

static const std::vector<float> TANGENT_HALF_GAUSS = gaussianHalfKernel1D(3, 10);
Point fast_curve_tangent(const CurveYX &curveYX, std::size_t i,
                         const std::vector<float> &GaussKernel = TANGENT_HALF_GAUSS, const bool forward = true,
                         const bool backward = true);

std::vector<Point> fast_curve_tangent(const CurveYX &curveYX,
                                      const std::vector<float> &GaussKernel = TANGENT_HALF_GAUSS,
                                      const std::vector<int> &evaluateAtID = {});

std::vector<int> curve_inflections_points(const std::vector<Point> &tangents,
                                          const std::vector<float> &dthetaSmoothHalfKernel, float angle_threshold);

/**************************************************************************************
 *              === BRANCH_CALIBRE.CPP ===
 **************************************************************************************/
std::array<IntPoint, 2> fast_branch_boundaries(const CurveYX &curveYX, const std::size_t i,
                                               const Tensor2DAccessor<bool> &segmentation, const Point &tangent);
std::array<IntPoint, 2> fast_branch_boundaries(const CurveYX &curveYX, const std::size_t i,
                                               const Tensor2DAccessor<bool> &segmentation);
std::vector<std::array<IntPoint, 2>> fast_branch_boundaries(const CurveYX &curveYX,
                                                            const Tensor2DAccessor<bool> &segmentation,
                                                            const std::vector<Point> &tangents,
                                                            const std::vector<int> &evaluateAtID = {});
inline std::vector<std::array<IntPoint, 2>> fast_branch_boundaries(const CurveYX &curveYX,
                                                                   const torch::Tensor &segmentation,
                                                                   const std::vector<Point> &tangents,
                                                                   const std::vector<int> &evaluateAtID = {}) {
    return fast_branch_boundaries(curveYX, segmentation.accessor<bool, 2>(), tangents, evaluateAtID);
}

std::vector<float> fast_branch_calibre(const CurveYX &curveYX, const Tensor2DAccessor<bool> &segmentation,
                                       const std::vector<Point> &tangents, const std::vector<int> &evaluateAtID = {});
inline std::vector<float> fast_branch_calibre(const CurveYX &curveYX, const torch::Tensor &segmentation,
                                              const std::vector<Point> &tangents,
                                              const std::vector<int> &evaluateAtID = {}) {
    return fast_branch_calibre(curveYX, segmentation.accessor<bool, 2>(), tangents, evaluateAtID);
}

std::vector<int> clean_branch_skeleton_around_node(const std::vector<CurveYX> &branchCurves, const int nodeID,
                                                   const std::set<Edge> &node_adjacency,
                                                   const Tensor2DAccessor<bool> &segmentation,
                                                   const int maxRemovedLength);

#endif  // BRANCH_TRACKING_H