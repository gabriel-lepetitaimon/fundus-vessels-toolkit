#ifndef RASTERIZE_TOPO_H
#define RASTERIZE_TOPO_H

#include "bezier.h"
#include "common.h"
#include "ray_iterators.h"

/**
 * @brief Rasterize the topology of the branches from the curves and boundaries.
 *
 * @param branch_list The edge list of the branches, where each edge is represented by a pair of node indices.
 * @param curves A vector of tensors representing the curves of the branches.
 * @param boundaries A vector of tensors representing the boundaries of the branches.
 * @param branchLabelsMap The tensor to store the branch labels.
 * @param topoMap The tensor to store the topology map.
 */
void rasterize_topology(const torch::Tensor& branch_list, const torch::Tensor& branch_parents,
                        const torch::Tensor& branch_dirs, std::vector<torch::Tensor> curves_tensor,
                        std::vector<torch::Tensor> boundaries, const torch::Tensor& nodes_yx_tensor,
                        float bezier_interpolate, bool fill_junctions, int expand, const torch::Tensor& branchMapping,
                        torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, torch::Tensor& fuzzySkeletonMap);

torch::Tensor& rasterize_branch(const torch::Tensor& curveTensor, const torch::Tensor& boundariesTensor,
                                torch::Tensor& outTensor, int fill_value = 1, float bspline_interpolate = 0.5);

void rasterize_branch_topo(const torch::Tensor& curve, const torch::Tensor& boundaries, int branchID, float branchRank,
                           torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, float bspline_interpolate = 0.5);

void rasterize_bezier(std::function<void(IntPoint, float, float)> updater, const IntPoint& p0, const IntPoint& p1,
                      const Point& t0, const Point& t1, const IntPointPair& b0, const IntPointPair& b1,
                      float bezier_smoothness, const IntPoint& maxShape);
void rasterize_bezier(std::function<void(IntPoint, float, float)> updater, const BezierCubic& bezier,
                      const IntPointPair& b0, const IntPointPair& b1, const IntPoint& maxShape);

void rasterize_branch_topo(const CurveYX& curve, const Tensor3DAcc<int>& boundaries,
                           std::function<void(IntPoint, float, float)> draw, const IntPoint& maxShape,
                           float bspline_interpolate = 0.5);

void rasterize_branch_topo(const CurveYX& curve, const std::vector<Point>& tangents, const std::vector<float>& calibres,
                           std::function<void(IntPoint, float, float)> draw, const IntPoint& maxShape,
                           float bspline_interpolate, float expand);

torch::Tensor drawQuad(const IntPair& p1, const IntPair& p2, const IntPair& p3, const IntPair& p4,
                       const IntPair& maxShape);

class QuadIterator {
   public:
    QuadIterator(const IntPoint& p1, const IntPoint& p2, const IntPoint& p3, const IntPoint& p4,
                 const IntPoint& maxPoint);

    bool iter();
    bool finished() const;
    bool isConvex() const;
    bool mergeP2P3IfNotConvex();
    const IntPoint& point() const;
    const std::array<int, 4>& crossProd() const;

    double fromP12toP34() const;
    double fromP1toP4() const;
    double fromP14() const;
    inline const IntPoint& d12() const { return pDiff[0]; }
    inline const IntPoint& d32() const { return pDiff[1]; }
    inline const IntPoint& d34() const { return pDiff[2]; }
    inline const IntPoint& d14() const { return pDiff[3]; }
    inline const double& invNorm12() const { return _invDiffNorm[0]; }
    inline const double& invNorm32() const { return _invDiffNorm[1]; }
    inline const double& invNorm34() const { return _invDiffNorm[2]; }
    inline const double& invNorm14() const { return _invDiffNorm[3]; }
    inline const int& cross12() const { return _crossProd[0]; }
    inline const int& cross23() const { return _crossProd[1]; }
    inline const int& cross34() const { return _crossProd[2]; }
    inline const int& cross41() const { return _crossProd[3]; }
    void precomputeInvDiffNorms();

    IntPoint p1, p2, p3, p4, pMin, pMax;
    std::array<IntPoint, 4> pDiff;  // Differences between points for cross product calculations

   private:
    IntPoint p;
    std::array<int, 4> _crossProd;
    std::array<double, 4> _invDiffNorm;
    bool _p23Inverted;  //, _p34Inverted;
    // bool fastIt;
    //  RayIterator _it12, _it34, _it;
    bool _positiveCrossProd, _p1p4Adjacent;
};

#endif  // RASTERIZE_TOPO_H