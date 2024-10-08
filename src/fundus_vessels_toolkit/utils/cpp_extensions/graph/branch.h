#ifndef BRANCH_TRACKING_H
#define BRANCH_TRACKING_H

#include "common.h"
#include "fit_bezier.h"

using CurveTangents = std::vector<Point>;

/**************************************************************************************
 *              === BRANCH_CALIBRE.CPP ===
 **************************************************************************************/

static const float INVALID_CALIBRE = std::numeric_limits<float>::quiet_NaN();

inline bool is_valid_calibre(float calibre) { return !std::isnan(calibre); }

std::array<IntPoint, 2> fast_branch_boundaries(const CurveYX &curveYX, const std::size_t i,
                                               const Tensor2DAcc<bool> &segmentation, const Point &tangent);
std::array<IntPoint, 2> fast_branch_boundaries(const CurveYX &curveYX, const std::size_t i,
                                               const Tensor2DAcc<bool> &segmentation);
std::vector<std::array<IntPoint, 2>> fast_branch_boundaries(const CurveYX &curveYX,
                                                            const Tensor2DAcc<bool> &segmentation,
                                                            const std::vector<Point> &tangents,
                                                            const std::vector<int> &evaluateAtID = {});

float fast_branch_calibre(const Point &boundL, const Point &boundR, const Point &tangent = {0, 0});
float fast_branch_calibre(const CurveYX &curveYX, std::size_t i, const Tensor2DAcc<bool> &segmentation,
                          const Point &tangent);
Scalars fast_branch_calibre(const CurveYX &curveYX, const Tensor2DAcc<bool> &segmentation,
                            const std::vector<Point> &tangents, const std::vector<int> &evaluateAtID = {});
Scalars fast_branch_calibre(const CurveYX &curveYX, const Tensor2DAcc<bool> &segmentation,
                            const std::vector<int> &evaluateAtID = {});

/**************************************************************************************
 *              === BRANCH_TANGENT.CPP ===
 **************************************************************************************/
Point adaptative_curve_tangent(const CurveYX &curveYX, std::size_t i, const float calibre, const bool forward,
                               const bool backward, const std::size_t curveStart = 0, const std::size_t curveEnd = 0);

Point curve_tangent(const CurveYX &curveYX, std::size_t i, const std::function<float(float)> &weighting,
                    std::size_t stride = 1, const bool forward = true, const bool backward = true,
                    const std::size_t curveStart = 0, std::size_t curveEnd = 0);
Point curve_tangent(const CurveYX &curveYX, std::size_t i, const float std, const bool forward = true,
                    const bool backward = true, const std::size_t curveStart = 0, const std::size_t curveEnd = 0);
std::vector<Point> curve_tangent(const CurveYX &curveYX, const Scalars &kernelStd,
                                 const std::vector<int> &evaluateAtID = {});

static const Scalars TANGENT_HALF_GAUSS = gaussianHalfKernel1D(3, 10);
Point fast_curve_tangent(const CurveYX &curveYX, std::size_t i, const Scalars &GaussKernel = TANGENT_HALF_GAUSS,
                         const bool forward = true, const bool backward = true, const std::size_t curveStart = 0,
                         std::size_t curveEnd = 0);

std::vector<Point> fast_curve_tangent(const CurveYX &curveYX, const Scalars &GaussKernel = TANGENT_HALF_GAUSS,
                                      const std::vector<int> &evaluateAtID = {});

std::vector<std::size_t> curve_inflections_points(const Scalars &signedCurvature, const float K_threshold = 0.05,
                                                  int idOffset = 0);

static const Scalars TANGENT_SMOOTHING_KERNEL = gaussianHalfKernel1D(5, 15);
Point smooth_tangents(const CurveTangents &tangents, std::size_t i,
                      const std::vector<float> weight = TANGENT_SMOOTHING_KERNEL, std::size_t start = 0,
                      std::size_t end = 0);
Point smooth_tangents(const CurveTangents &tangents, std::size_t i, const float gaussianStd, std::size_t curveStart = 0,
                      std::size_t curveEnd = 0);

Scalars tangents_to_curvature(const CurveTangents &tangents, bool signedCurvature = false, const float gaussianStd = 5,
                              std::size_t start = 0, std::size_t end = 0);

/**************************************************************************************
 *              === BRANCH_TRACKING.CPP ===
 **************************************************************************************/

std::vector<std::array<IntPoint, 2>> find_branch_endpoints(const torch::Tensor &branch_labels,
                                                           const torch::Tensor &node_yx,
                                                           const torch::Tensor &branch_list);

std::vector<CurveYX> track_branches(const torch::Tensor &branch_labels, const torch::Tensor &node_yx,
                                    const torch::Tensor &branch_list);

IntPoint track_nearest_border(const IntPoint &start, const Point &direction, const Tensor2DAcc<bool> &segmentation,
                              int max_distance = 40);

std::tuple<int, float> findClosestPixel(const CurveYX &curve, const Point &p, int start, int end,
                                        bool findFirstLocalMinimum = false);

std::list<SizePair> splitInContiguousCurves(const CurveYX &curve);

/**************************************************************************************
 *              === BRANCH_FIXING.CPP ===
 **************************************************************************************/
std::vector<std::array<std::tuple<Vector, float, IntPoint, IntPoint>, 2>> clean_branches_skeleton(
    std::vector<CurveYX> &branchCurves, Tensor2DAcc<int> &branchesLabelMap, const Tensor2DAcc<bool> &segmentation,
    const GraphAdjList &adjacency, int maxRemovedLengthNode, int maxRemovedLengthEnd = -1,
    bool adaptativeTangent = false);

std::vector<std::tuple<int, Vector, float, IntPoint, IntPoint>> clean_branch_skeleton_around_node(
    const std::vector<CurveYX> &branchCurves, int nodeID, const std::set<Edge> &node_adjacency,
    const Tensor2DAcc<bool> &segmentation, int maxRemovedLength, bool adaptativeTangent);

std::tuple<int, Vector, float, IntPoint, IntPoint> clean_branch_skeleton_tip(const std::vector<CurveYX> &branchCurves,
                                                                             int branchID, bool startTermination,
                                                                             const Tensor2DAcc<bool> &segmentation,
                                                                             int maxRemovedLength,
                                                                             bool adaptativeTangent);

void remove_small_spurs(float min_length, EdgeList &edgeList, std::vector<CurveYX> &branchCurves,
                        std::vector<IntPoint> &nodeCoords, Tensor2DAcc<int> &labelMap);

std::vector<Edge> find_spurs(const std::vector<CurveYX> &branchCurves, const EdgeList &edgeList,
                             const Tensor2DAcc<bool> &segmentation, float min_length, const float calibre_factor,
                             const float max_length = 100);

float largest_node_calibre(const int nodeId, const GraphAdjList &adjacency, const std::vector<CurveYX> &branchesCurves,
                           const Tensor2DAcc<bool> &segmentation);

/**************************************************************************************
 *              === BRANCH_GEOMETRY.CPP ===
 **************************************************************************************/
std::tuple<std::vector<CurveTangents>, std::vector<Scalars>, std::vector<IntPointPairs>, std::vector<Scalars>,
           std::vector<Sizes>, std::vector<BSpline>>
extract_branches_geometry(std::vector<CurveYX> branch_curves, const Tensor2DAcc<bool> &segmentation,
                          std::map<std::string, double> options = {}, bool assume_contiguous = false);

BSpline bspline_regression(const CurveYX &curve, const CurveTangents &tangents, const Scalars &curvatures,
                           double bspline_max_error, const float K_threshold = 0.15, std::size_t start = 0,
                           std::size_t end = 0);

BSpline bspline_regression(const CurveYX &curve, const CurveTangents &tangents, double bspline_max_error,
                           const float K_threshold = 0.15, std::size_t start = 0, std::size_t end = 0);

BSpline bspline_regression(const CurveYX &curve, const CurveTangents &tangents,
                           const std::vector<std::size_t> &splitCandidate, double targetSqrError, std::size_t start = 0,
                           std::size_t end = 0);

BSpline iterative_fit_bspline(const CurveYX &d, const std::vector<Point> &tangents, const BezierCurve &bezier,
                              const std::vector<double> &u, const std::vector<double> &bezierSqrErrors,
                              const std::vector<std::size_t> &splitCandidates, double error, std::size_t first,
                              std::size_t last);

#endif  // BRANCH_TRACKING_H