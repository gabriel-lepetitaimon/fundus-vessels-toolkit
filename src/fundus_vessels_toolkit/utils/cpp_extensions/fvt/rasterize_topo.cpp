#include "rasterize_topo.h"

#include "bezier.h"
#include "branch.h"
#include "ray_iterators.h"

void rasterize_topology(const torch::Tensor& branch_list, const torch::Tensor& branch_parents,
                        const torch::Tensor& branch_dirs, std::vector<torch::Tensor> curves_tensor,
                        std::vector<torch::Tensor> boundaries, const torch::Tensor& nodes_yx_tensor,
                        float bezier_interpolate, bool fill_junctions, torch::Tensor& branchLabelsMap,
                        torch::Tensor& topoMap) {
    // Ensure the branchLabelsMap and topoMap are initialized correctly
    TORCH_CHECK(branchLabelsMap.dim() == 2 && topoMap.dim() == 2, "branchLabelsMap and topoMap must be 2D tensors.");
    TORCH_CHECK(branchLabelsMap.size(0) == topoMap.size(0) && branchLabelsMap.size(1) == topoMap.size(1),
                "branchLabelsMap and topoMap must have the same shape.");
    TORCH_CHECK(branchLabelsMap.scalar_type() == torch::kInt32 && topoMap.scalar_type() == torch::kFloat32,
                "branchLabelsMap and topoMap must be of type Int32.");

    auto branchLabelsMapAcc = branchLabelsMap.accessor<int, 2>();
    auto topoMapAcc = topoMap.accessor<float, 2>();
    IntPoint maxShape = {(int)branchLabelsMap.size(0), (int)branchLabelsMap.size(1)};

    // === Initialize the adjacency list for the topology ===
    const auto& branchListAcc = branch_list.accessor<int, 2>();
    const auto& branchParentsAcc = branch_parents.accessor<int, 1>();
    const auto& branchDirsAcc = branch_dirs.accessor<bool, 1>();
    std::size_t N_branches = branchListAcc.size(0);
    auto [hierarchy, max_rank] = edge_list_to_hierarchy(branchListAcc, branchParentsAcc, branchDirsAcc);

    std::vector<std::list<int>> branchByRank(max_rank + 1);
    for (std::size_t b = 0; b < N_branches; b++) branchByRank[hierarchy[b].rank].push_back(b);

    // === Flip curves and boundaries according to branch direction ===
    std::vector<Tensor3DAcc<int>> boundariesAcc;
    boundariesAcc.reserve(boundaries.size());
    for (std::size_t b = 0; b < N_branches; b++) {
        if (!branchDirsAcc[b]) {
            curves_tensor[b] = curves_tensor[b].flip({0});
            boundaries[b] = boundaries[b].flip({0, 1});
        }
        boundariesAcc.push_back(boundaries[b].accessor<int, 3>());
    }
    const auto& curves = tensors_to_curves(curves_tensor);
    const auto& nodes_yx = tensor_to_curve(nodes_yx_tensor);

    // === Compute Tips info ===
    struct TipInfo {
        IntPoint yx = IntPoint::Invalid();
        Point t = {0, 0};
        IntPointPair b = {IntPoint::Invalid(), IntPoint::Invalid()};
        float w = -1;
    };
    std::vector<std::array<TipInfo, 2>> tips(N_branches);
    for (auto branches = branchByRank.rbegin(); branches != branchByRank.rend(); ++branches) {
        for (const int branchID : *branches) {
            const CurveYX& curve = curves[branchID];
            const auto& boundary = boundariesAcc[branchID];
            if (curve.size() == 0) continue;
            for (const auto headTip : {0, 1}) {
                std::size_t i = headTip ? curve.size() - 1 : 0;
                tips[branchID][headTip].yx = curve[i];
                IntPointPair b = {IntPoint(boundary[i][0]), IntPoint(boundary[i][1])};
                float w = distance(b[0], b[1]);
                Point t = adaptative_curve_tangent(curve, i, w, headTip == 0, headTip == 1).normalize();
                tips[branchID][headTip].t = t;
                tips[branchID][headTip].b = b;
                tips[branchID][headTip].w = w;
            }
        }
    }

    // Infer missing tips
    for (auto branches = branchByRank.rbegin(); branches != branchByRank.rend(); ++branches) {
        for (const int branchID : *branches) {
            auto &tailTip = tips[branchID][0], &headTip = tips[branchID][1];
            const auto& branch = hierarchy[branchID];

            if (headTip.w < 0 && !branch.children.empty()) {
                // If missing head tip try to copy tail tip from child

                if (branch.children.size() == 1) {
                    // If only one child, copy its tip directly
                    const auto& childTailTip = tips[branch.children[0]][0];
                    if (childTailTip.w >= 0) headTip = childTailTip;
                } else {
                    // otherwise average tips from all children
                    headTip.yx = nodes_yx[branch.head_node];  // set head position to node

                    float w = 1.0;
                    Point t = {0, 0};
                    int child_N = 0;
                    for (const int childID : branch.children) {
                        const auto& childTailTip = tips[childID][0];
                        if (childTailTip.w >= 0) {
                            w *= childTailTip.w;
                            const Point& t0 =
                                infer_bezier_t0(headTip.yx, childTailTip.yx, -childTailTip.t) * childTailTip.w;
                            t += t0;
                            child_N++;
                        }
                    }
                    if (child_N > 0) {
                        headTip.w = child_N > 1 ? std::pow(w, 1.0 / child_N) : w;
                        headTip.b = headTip.yx.left_right_pair(t, headTip.w * 0.5);
                        // Leave tangent as zero (to not affect the drawn bezier curve )
                    }
                }
            }

            if (tailTip.w < 0) {  // If missing tail tip ...
                if (branch.parent != -1 && tips[branch.parent][1].w >= 0) {
                    // ... try to copy head tip from direct parent
                    tailTip = tips[branch.parent][1];
                } else if (headTip.w >= 0) {
                    // ... or propagate from head tip
                    tailTip.w = headTip.w;
                    tailTip.yx = nodes_yx[branch.tail_node];  // set tail position to node
                    // leave tangent as zero
                    // Infer boundary points by estimating tangent with bezier cubic
                    const auto& t = infer_bezier_t0(tailTip.yx, headTip.yx, -headTip.t);
                    tailTip.b = tailTip.yx.left_right_pair(t, tailTip.w * 0.5);
                }
            }

            // Not handled cases are:
            //      - isolated branch
            //      - no child (or without tip info) and missing tip info from direct parent
            // In those cases, at least set tip position to node
            if (!headTip.yx.is_valid()) headTip.yx = nodes_yx[branch.head_node];
            if (!tailTip.yx.is_valid()) tailTip.yx = nodes_yx[branch.tail_node];
        }
    }

    // === Draw the tree from root to leaves ===
    const std::vector<int> roots(branchByRank[0].begin(), branchByRank[0].end());
#pragma omp parallel for schedule(dynamic)
    for (const int root : roots) {
        std::stack<int> q;
        q.push(root);

        while (!q.empty()) {
            // Read branch info
            int branchID = q.top();
            q.pop();
            const auto& branch = hierarchy[branchID];
            const auto& curve = curves[branchID];
            const auto& boundary = boundariesAcc[branchID];
            const auto N = curve.size();

            // === DRAW THE BRANCH ===
            auto drawTopo = [&](IntPoint pt, float u, int branchID, float rank) {
                branchLabelsMapAcc[pt.y][pt.x] = branchID + 1;
                float topoValue = rank + u;
                if (topoMapAcc[pt.y][pt.x] < topoValue) topoMapAcc[pt.y][pt.x] = topoValue;
            };
            auto drawBranchTopo = [&](IntPoint pt, float u) { drawTopo(pt, 0.1 + 0.9 * u, branchID, branch.rank); };
            if (N != 0) {  // If the branch is not empty rasterize it
                rasterize_branch_topo(curve, boundary, drawBranchTopo, maxShape, bezier_interpolate);
            } else {  // Otherwise draw bezier cubic interpolation
                const auto &tailTip = tips[branchID][0], &headTip = tips[branchID][1];
                if (tailTip.w >= 0 && headTip.w >= 0 && bezier_interpolate > 0.0f) {
                    rasterize_bezier(drawBranchTopo, tailTip.yx, headTip.yx, tailTip.t, headTip.t, tailTip.b, headTip.b,
                                     bezier_interpolate, maxShape);
                }
            }

            // === FILL HEAD JUNCTION ===
            if (fill_junctions) {
                const auto& headTip = tips[branchID][1];
                if (headTip.w < 0) continue;  // If the head tip is invalid, skip this filling

                for (const auto& childID : branch.children) {
                    const auto& childTip = tips[childID][0];
                    if (childTip.w < 0) continue;  // If the child tip is invalid, skip this filling

                    auto drawJunctionTopo = [&](IntPoint pt, float u) {
                        drawTopo(pt, 0.1 * u, childID, branch.rank + 1);
                    };

                    if ((distance(headTip.yx, childTip.yx) <= (headTip.w + childTip.w) &&
                         headTip.t.dot(childTip.t) > 0.5) ||
                        bezier_interpolate <= 0.0f) {
                        // If tips are close enough, draw a simple quad
                        auto it = QuadIterator(headTip.b[0], headTip.b[1], childTip.b[1], childTip.b[0], maxShape);
                        it.precomputeInvDiffNorms();
                        while (it.iter()) drawJunctionTopo(it.point(), it.fromP12toP34());
                    } else {
                        // Otherwise draw bezier cubic interpolation
                        // const IntPoint& node_yx = nodes_yx[branch.head_node];
                        double d = distance(Point(headTip.yx), Point(childTip.yx)) * bezier_interpolate;
                        const Point c0 = headTip.yx + headTip.t * d;    // distance(Point(headTip.yx), Point(node_yx));
                        const Point c1 = childTip.yx - childTip.t * d;  // distance(Point(childTip.yx), Point(node_yx));
                        rasterize_bezier(drawJunctionTopo, {headTip.yx, c0, c1, childTip.yx}, headTip.b, childTip.b,
                                         maxShape);
                    }
                }
            }

            // Enqueue the branch's children
            for (const auto& nextBranchID : branch.children) q.push(nextBranchID);
        }
    }
}

void rasterize_branch_topo(const torch::Tensor& curve, const torch::Tensor& boundaries, int branchID, float branchRank,
                           torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, float bspline_interpolate) {
    // Ensure the branchLabelsMap and topoMap are initialized correctly
    TORCH_CHECK(branchLabelsMap.dim() == 2 && topoMap.dim() == 2, "branchLabelsMap and topoMap must be 2D tensors.");
    TORCH_CHECK(branchLabelsMap.size(0) == topoMap.size(0) && branchLabelsMap.size(1) == topoMap.size(1),
                "branchLabelsMap and topoMap must have the same shape.");
    TORCH_CHECK(branchLabelsMap.scalar_type() == torch::kInt32 && topoMap.scalar_type() == torch::kFloat32,
                "branchLabelsMap must be of type Int32 and topoMap must be of type Float32.");

    // Ensure the curve and boundaries tensors are of the correct shape
    TORCH_CHECK(curve.dim() == 2 && boundaries.dim() == 3,
                "curve must be a 2D tensor and boundaries must be a 3D tensor.");
    TORCH_CHECK(curve.size(1) == 2 && boundaries.size(1) == 2 && boundaries.size(2) == 2,
                "curve must have shape [N, 2] and boundaries must have shape [N, 2, 2].");
    TORCH_CHECK(curve.size(0) == boundaries.size(0), "curve and boundaries must have the same first dimension size.");

    auto branchLabelsMapAcc = branchLabelsMap.accessor<int, 2>();
    auto topoMapAcc = topoMap.accessor<float, 2>();
    auto drawBranchTopo = [&](IntPoint pt, float u) {
        branchLabelsMapAcc[pt.y][pt.x] = branchID;
        float topoValue = branchRank + 0.9 * u;
        if (topoMapAcc[pt.y][pt.x] < topoValue) topoMapAcc[pt.y][pt.x] = topoValue;
    };

    IntPoint maxShape = {(int)branchLabelsMap.size(0), (int)branchLabelsMap.size(1)};
    const auto& curve_vec = tensor_to_curve(curve);

    return rasterize_branch_topo(curve_vec, boundaries.accessor<int, 3>(), drawBranchTopo, maxShape,
                                 bspline_interpolate);
}

void rasterize_bezier(std::function<void(IntPoint, float)> updater, const IntPoint& p0, const IntPoint& p1,
                      const Point& t0, const Point& t1, const IntPointPair& b0, const IntPointPair& b1,
                      float bezier_smoothness, const IntPoint& maxShape) {
    bezier_smoothness *= distance(Point(p0), Point(p1));
    BezierCubic bezier = {Point(p0), Point(p0) + t0 * bezier_smoothness, Point(p1) - t1 * bezier_smoothness, Point(p1)};
    rasterize_bezier(updater, bezier, b0, b1, maxShape);
}

void rasterize_bezier(std::function<void(IntPoint, float)> updater, const BezierCubic& bezier, const IntPointPair& b0,
                      const IntPointPair& b1, const IntPoint& maxShape) {
    const float w0 = std::max(distance(b0[0], b0[1]), 1.0f), w1 = std::max(distance(b1[0], b1[1]), 1.0f);

    // == Discretize Bezier ==
    auto [interpPoints, us] = discretizeBezier(bezier);
    auto tangents = evaluate_bezier_tangent(bezier, us);
    auto N = interpPoints.size();

    double u = 0.0f, nextU;
    IntPoint p = interpPoints[0], nextP;
    Point t = tangents[0].normalize(), nextT;
    float w = w0, nextW;
    IntPointPair b = b0, nextB;

    for (std::size_t i = 0; i < N - 1; i++) {
        if (i != N - 2) {
            nextU = us[i + 1];
            nextP = interpPoints[i + 1];
            nextT = tangents[i + 1].normalize();
            nextW = lerp(w0, w1, nextU);
            nextB = nextP.left_right_pair(nextT, nextW * 0.5, true);
        } else {
            nextU = 1.0f;
            nextP = interpPoints[N - 1];
            nextT = tangents[N - 1].normalize();
            nextW = w1;
            nextB = b1;
        }
        updater(p, u);
        auto externalError = t.angle(nextT) * w * 0.5;
        for (int lr = 0; lr < 2; ++lr) {  // Iterate over left and right quads
            int lr_sign = 1 - lr * 2;     // +1 for left, -1 for right
            if (externalError * lr_sign < 0 && std::ceil(std::abs(externalError)) > 1) {
                // Subdivide exterior perimeter
                int N_splits = std::ceil(std::abs(externalError));
                IntPoint prev_b = b[lr], b;
                float ds = 1.0f / N_splits;
                for (float s = ds; s < 1; s += ds) {
                    double s_u = lerp(u, nextU, s);
                    auto s_t = lerp(t, nextT, s);
                    auto s_w = lerp(w, nextW, s) * 0.5;
                    b = evaluate_bezier(bezier, s_u).toInt().left_right_pair(s_t, s_w)[lr];
                    QuadIterator it(p, prev_b, b, nextP, maxShape);
                    it.precomputeInvDiffNorms();
                    while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP1toP4()));
                    prev_b = b;
                }
                QuadIterator it(p, prev_b, nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP1toP4()));
            } else {
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP12toP34()));
            }
        }

        // Move to the next point
        p = nextP;
        t = nextT;
        w = nextW;
        b = nextB;
        u = nextU;
    }
}

void rasterize_branch_topo(const CurveYX& curve, const Tensor3DAcc<int>& boundaries,
                           std::function<void(IntPoint, float)> draw, const IntPoint& maxShape,
                           float bspline_interpolate) {
    auto N = curve.size();

    auto last = N - 1;
    IntPoint p = curve[0];
    IntPointPair b = {boundaries[0][0], boundaries[0][1]}, nextB;

    for (std::size_t i = 0; i != last; i++) {
        auto nextI = i + 1;
        const auto& nextP = curve[nextI];
        IntPointPair nextB = {boundaries[nextI][0], boundaries[nextI][1]};

        auto localDraw = [&](IntPoint pt, float u) { draw(pt, (u + i) / N); };

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() <= 9) {
            for (int lr = 0; lr < 2; ++lr) {
                // Draw the center point
                localDraw(p, 0.0f);

                // Iterate over left and right quads
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) localDraw(it.point(), it.fromP12toP34());
            }
        } else if (bspline_interpolate > 0.0f) {
            // Rasterize a bezier curve between p and nextP
            Point t = adaptative_curve_tangent(curve, i, distance(b[0], b[1]), false, true),
                  nextT = adaptative_curve_tangent(curve, nextI, distance(nextB[0], nextB[1]), true, false);
            rasterize_bezier(localDraw, p, nextP, t, nextT, b, nextB, bspline_interpolate, maxShape);
        }

        // Move to the next point
        p = nextP;
        b = nextB;
    }
}

torch::Tensor& rasterize_branch(const torch::Tensor& curveTensor, const torch::Tensor& boundariesTensor,
                                torch::Tensor& outTensor, int fill_value, float bspline_interpolate) {
    // Check input shapes
    TORCH_CHECK(curveTensor.dim() == 2 && curveTensor.size(1) == 2, "curve must have shape [N, 2]");
    TORCH_CHECK(boundariesTensor.dim() == 3 && boundariesTensor.size(1) == 2 && boundariesTensor.size(2) == 2,
                "boundaries must have shape [N, 2, 2]");
    TORCH_CHECK(curveTensor.size(0) == boundariesTensor.size(0),
                "curve and boundaries must have the same first dimension size.");

    // Check output tensor
    TORCH_CHECK(outTensor.dim() == 2, "out must be a 2D tensor");
    TORCH_CHECK(outTensor.dtype() == torch::kInt, "out must be an integer tensor");

    // Prepare accessors and constants
    const CurveYX& curve = tensor_to_curve(curveTensor);
    auto boundaries = boundariesTensor.accessor<int, 3>();
    auto out = outTensor.accessor<int, 2>();

    const IntPoint maxShape = {(int)out.size(0), (int)out.size(1)};
    int N = (int)curve.size();

    IntPoint p = curve[0], nextP;
    IntPointPair b = {boundaries[0][0], boundaries[0][1]}, nextB;

    for (int i = 0; i < N - 1; i++) {
        int nextI = i + 1;
        IntPoint nextP = curve[nextI];
        IntPointPair nextB = {boundaries[nextI][0], boundaries[nextI][1]};

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() <= 9) {
            out[p.y][p.x] = fill_value;       // Draw the center point
            for (int lr = 0; lr < 2; ++lr) {  // Iterate over left and right quads
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                while (it.iter()) out[it.point().y][it.point().x] = fill_value;
            }
        } else if (bspline_interpolate > 0.0f) {
            // Rasterize a bezier curve between p and nextP
            Point t = adaptative_curve_tangent(curve, i, distance(b[0], b[1]), false, true),
                  nextT = adaptative_curve_tangent(curve, nextI, distance(nextB[0], nextB[1]), true, false);

            rasterize_bezier([&](IntPoint pt, float u) { out[pt.y][pt.x] = fill_value; }, p, nextP, t, nextT, b, nextB,
                             bspline_interpolate, maxShape);
        }

        // Move to the next point
        p = nextP;
        b = nextB;
    }

    return outTensor;
}

/**********************************************************************************************************************
 *     === QUAD RASTERIZATION ===
 *********************************************************************************************************************/
bool isPointInQuad(const std::array<int, 4>& d) {
    // Check if the point is inside the quad defined by the cross products
    return (d[0] >= 0 && d[1] >= 0 && d[2] >= 0 && d[3] >= 0) || (d[0] <= 0 && d[1] <= 0 && d[2] <= 0 && d[3] <= 0);
}

IntPoint minBoundingPoint(const IntPoint& p1, const IntPoint& p2, const IntPoint& p3, const IntPoint& p4,
                          const IntPoint& maxPoint) {
    // Calculate the minimum bounding point of the quad defined by points p1, p2, p3, and p4
    return IntPoint(std::max({std::min({p1.y, p2.y, p3.y, p4.y, maxPoint.y - 1}), 0}),
                    std::max({std::min({p1.x, p2.x, p3.x, p4.x, maxPoint.x - 1}), 0}));
}

IntPoint maxBoundingPoint(const IntPoint& p1, const IntPoint& p2, const IntPoint& p3, const IntPoint& p4,
                          const IntPoint& maxPoint) {
    // Calculate the minimum bounding point of the quad defined by points p1, p2, p3, and p4
    return IntPoint(std::min({std::max({p1.y, p2.y, p3.y, p4.y, 0}), maxPoint.y - 1}),
                    std::min({std::max({p1.x, p2.x, p3.x, p4.x, 0}), maxPoint.x - 1}));
}

QuadIterator::QuadIterator(const IntPoint& p1, const IntPoint& p2, const IntPoint& p3, const IntPoint& p4,
                           const IntPoint& maxShape)
    : p1(p1),
      p2(p2),
      p3(p3),
      p4(p4),
      pMin(minBoundingPoint(p1, p2, p3, p4, maxShape)),
      pMax(maxBoundingPoint(p1, p2, p3, p4, maxShape)),
      pDiff{{p2 - p1, p2 - p3, p4 - p3, p4 - p1}},
      p(pMin.y, pMin.x - 1),
      _crossProd{0, 0, 0, 0},
      hourGlassQuad(pDiff[0].dot(pDiff[2]) > 0) {}

bool QuadIterator::finished() const {
    // Check if the iterator has finished iterating over the quad
    return p.y >= pMax.y && p.x >= pMax.x;
}

bool QuadIterator::iter() {
    while (true) {
        // === Move to the next point ===
        if (p.x < pMax.x) {
            p.x++;
        } else if (p.y < pMax.y) {
            p.x = pMin.x;  // Reset x to the minimum x value
            p.y++;         // Move to the next row
        } else {
            return false;  // No more points to iterate
        }

        // === Check if the point is inside the quad ===
        // Compute the cross products
        auto pp1 = p - p1, pp3 = p - p3;
        _crossProd[0] = pDiff[0].cross(pp1);                      //  (p2-p1) x (p-p1)
        _crossProd[1] = -pDiff[1].cross(pp3);                     // -(p2-p3) x (p-p3)
        _crossProd[3] = -pDiff[3].cross(pp1);                     // -(p4-p1) x (p-p1)
        if (!hourGlassQuad) _crossProd[2] = pDiff[2].cross(pp3);  //  (p4-p3) x (p-p3)

        // Check if the point is inside the quad using the cross products
        if (isPointInQuad(_crossProd)) return true;
    }
}

const IntPoint& QuadIterator::point() const { return p; }
const std::array<int, 4>& QuadIterator::crossProd() const { return _crossProd; }

double QuadIterator::fromP12toP34() const {
    double d = abs(_crossProd[0]) * _invDiffNorm[0];
    double D = abs(_crossProd[2]) * _invDiffNorm[3] + d;
    return D > 0 ? d / D : 0;
}

double QuadIterator::fromP1toP4() const {
    IntPoint pp1 = p - p1;
    return clip(pp1.normalize().dot(pDiff[3]) * _invDiffNorm[3], 0.0, 1.0);
}

void QuadIterator::precomputeInvDiffNorms() {
    for (int i = 0; i < 4; ++i) _invDiffNorm[i] = 1.0 / pDiff[i].norm();
}

torch::Tensor drawQuad(const IntPair& p1, const IntPair& p2, const IntPair& p3, const IntPair& p4,
                       const IntPair& maxShape) {
    auto scene = torch::zeros({maxShape[0], maxShape[1]}, torch::kFloat);
    auto sceneAcc = scene.accessor<float, 2>();

    QuadIterator it(p1, p2, p3, p4, maxShape);

    const float invNorm = it.pDiff[0].norm(), nextInvNorm = it.pDiff[3].norm();
    while (it.iter()) {
        IntPoint p = it.point();
        float d = abs(it.crossProd()[0]) * invNorm;
        float sumD = abs(it.crossProd()[2]) * nextInvNorm + d;
        d = sumD > 0 ? d / sumD : 0;  // Avoid division by zero
        sceneAcc[p.y][p.x] = 1 + d;
    }

    return scene;
}