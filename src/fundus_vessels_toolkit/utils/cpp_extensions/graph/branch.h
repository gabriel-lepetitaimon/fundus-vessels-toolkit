#ifndef BRANCH_TRACKING_H
#define BRANCH_TRACKING_H

#include "common.h"
#include "fit_bezier.h"

using CurveTangents = std::vector<Point>;

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

float fast_branch_calibre(const CurveYX &curveYX, std::size_t i, const Tensor2DAccessor<bool> &segmentation,
                          const Point &tangent);
std::vector<float> fast_branch_calibre(const CurveYX &curveYX, const Tensor2DAccessor<bool> &segmentation,
                                       const std::vector<Point> &tangents, const std::vector<int> &evaluateAtID = {});
std::vector<float> fast_branch_calibre(const CurveYX &curveYX, const Tensor2DAccessor<bool> &segmentation,
                                       const std::vector<int> &evaluateAtID = {});

/**************************************************************************************
 *              === BRANCH_TANGENT.CPP ===
 **************************************************************************************/
Point curve_tangent(const CurveYX &curveYX, std::size_t i, const std::function<float(float)> &weighting,
                    std::size_t stride = 1, const bool forward = true, const bool backward = true,
                    const std::size_t curveStart = 0, std::size_t curveEnd = 0);
Point curve_tangent(const CurveYX &curveYX, std::size_t i, const float std, const bool forward = true,
                    const bool backward = true, const std::size_t curveStart = 0, const std::size_t curveEnd = 0);
std::vector<Point> curve_tangent(const CurveYX &curveYX, const std::vector<float> &kernelStd,
                                 const std::vector<int> &evaluateAtID = {});

static const std::vector<float> TANGENT_HALF_GAUSS = gaussianHalfKernel1D(3, 10);
Point fast_curve_tangent(const CurveYX &curveYX, std::size_t i,
                         const std::vector<float> &GaussKernel = TANGENT_HALF_GAUSS, const bool forward = true,
                         const bool backward = true, const std::size_t curveStart = 0, std::size_t curveEnd = 0);

std::vector<Point> fast_curve_tangent(const CurveYX &curveYX,
                                      const std::vector<float> &GaussKernel = TANGENT_HALF_GAUSS,
                                      const std::vector<int> &evaluateAtID = {});

BSpline bspline_regression(const CurveYX &curve, const CurveTangents &tangents, double bspline_max_error,
                           const std::vector<float> &inflection_halfKernel = GAUSSIAN_HALF_STD3,
                           double inflection_max_angle = 1.5, std::size_t start = 0, std::size_t end = 0);

std::vector<std::size_t> curve_inflections_points(const std::vector<Point> &tangents,
                                                  const std::vector<float> &dthetaSmoothHalfKernel,
                                                  double angle_threshold, std::size_t start, std::size_t end);

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

std::list<SizePair> splitInContiguousCurves(const CurveYX &curve);

/**************************************************************************************
 *              === BRANCH_FIXING.CPP ===
 **************************************************************************************/
void clean_branches_skeleton(std::vector<CurveYX> &branchCurves, Tensor2DAccessor<int> &branchesLabelMap,
                             const Tensor2DAccessor<bool> &segmentation, const GraphAdjList &adjacency,
                             const int maxRemovedLength);

std::vector<int> clean_branch_skeleton_around_node(const std::vector<CurveYX> &branchCurves, const int nodeID,
                                                   const std::set<Edge> &node_adjacency,
                                                   const Tensor2DAccessor<bool> &segmentation,
                                                   const int maxRemovedLength);

std::vector<Edge> find_small_spurs(const std::vector<CurveYX> &branchCurves,
                                   const Tensor2DAccessor<int> &branchesLabelMap, const GraphAdjList &adjacency,
                                   const Tensor2DAccessor<bool> &segmentation, float max_spurs_length,
                                   const float max_calibre_ratio);

float largest_near_calibre(const Edge &edge, const GraphAdjList &adjacency, const std::vector<CurveYX> &branchesCurves,
                           const Tensor2DAccessor<bool> &segmentation);

/**************************************************************************************
 *              === BRANCH_GEOMETRY.CPP ===
 **************************************************************************************/
std::tuple<std::vector<CurveTangents>, std::vector<std::vector<float>>, std::vector<BSpline>> extract_branches_geometry(
    std::vector<CurveYX> branch_curves, const Tensor2DAccessor<bool> &segmentation,
    std::map<std::string, double> options = {}, bool assume_contiguous = false);

#endif  // BRANCH_TRACKING_H