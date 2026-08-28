#include "rasterize_topo.h"

#include "bezier.h"
#include "branch.h"
#include "ray_iterators.h"

void expand_boundaries(IntPointPair& b, float expand, const Point& t) {
    IntPoint offset;
    if (b[0] == b[1])
        offset = (t.rot90() * expand).toInt();
    else {
        RayIterator ray(b[0] - b[1]);
        offset = ray.extrapolate(ray.stepsCountTo(expand));
    }
    b[0] += offset;
    b[1] -= offset;
}

void rasterize_topology(const torch::Tensor& branch_list, const torch::Tensor& branch_parents,
                        const torch::Tensor& branch_dirs, std::vector<torch::Tensor> curves_tensor,
                        std::vector<torch::Tensor> boundaries, const torch::Tensor& nodes_yx_tensor,
                        float bezier_interpolate, bool fill_junctions, int expand, const torch::Tensor& branchMapping,
                        torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, torch::Tensor& fuzzySkeletonMap) {
    // Ensure the branchLabelsMap and topoMap are initialized correctly
    TORCH_CHECK(branchLabelsMap.dim() == 2 && topoMap.dim() == 2, "branchLabelsMap and topoMap must be 2D tensors.");
    TORCH_CHECK(branchLabelsMap.size(0) == topoMap.size(0) && branchLabelsMap.size(1) == topoMap.size(1),
                "branchLabelsMap and topoMap must have the same shape.");
    TORCH_CHECK(branchLabelsMap.scalar_type() == torch::kInt64 && topoMap.scalar_type() == torch::kFloat32,
                "branchLabelsMap and topoMap must be of type Int64.");
    TORCH_CHECK(fuzzySkeletonMap.dim() == 2, "fuzzySkeletonMap must be a 2D tensor.");
    TORCH_CHECK(fuzzySkeletonMap.size(0) == topoMap.size(0) && fuzzySkeletonMap.size(1) == topoMap.size(1),
                "fuzzySkeletonMap must have the same shape as topoMap.");
    TORCH_CHECK(fuzzySkeletonMap.scalar_type() == torch::kFloat32, "fuzzySkeletonMap must be of type Float32.");

    // Check provided inputs
    TORCH_CHECK(branch_list.dim() == 2 && branch_list.size(1) == 2,
                "branch_list must be a 2D tensor with shape (num_branches, 2).");
    TORCH_CHECK(branch_parents.dim() == 1 && branch_parents.size(0) == branch_list.size(0),
                "branch_parents must be a 1D tensor with the same length as branch_list.");
    TORCH_CHECK(branch_dirs.dim() == 1 && branch_dirs.size(0) == branch_list.size(0),
                "branch_dirs must be a 1D tensor with the same length as branch_list.");
    TORCH_CHECK(nodes_yx_tensor.dim() == 2 && nodes_yx_tensor.size(1) == 2,
                "nodes_yx_tensor must be a 2D tensor with shape (num_nodes, 2).");
    TORCH_CHECK(curves_tensor.size() == (std::size_t)branch_list.size(0),
                "curves_tensor must have the same number of elements as branch_list.");
    TORCH_CHECK(boundaries.size() == (std::size_t)branch_list.size(0),
                "boundaries must have the same number of elements as branch_list.");
    if (branchMapping.numel() > 0) {
        TORCH_CHECK(branchMapping.dim() == 1 && branchMapping.size(0) == branch_list.size(0) + 1,
                    "branchMapping must be a 1D tensor with the same length as branch_list.");
        TORCH_CHECK(branchMapping.scalar_type() == torch::kInt64, "branchMapping must be of type Int64.");
    }

    auto branchLabelsMapAcc = branchLabelsMap.accessor<int64_t, 2>();
    auto topoMapAcc = topoMap.accessor<float, 2>();
    auto fuzzySkeletonMapAcc = fuzzySkeletonMap.accessor<float, 2>();
    IntPoint maxShape = {(int)branchLabelsMap.size(0), (int)branchLabelsMap.size(1)};
    auto branchMappingAcc = branchMapping.accessor<int64_t, 1>();
    const bool useBranchMapping = branchMapping.numel() > 0;

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

    std::vector<CurveYX> curves;
    tensors_to_curves(curves_tensor, curves);
    std::vector<std::vector<Point>> tangents(curves.size());
    std::vector<std::vector<float>> calibres(curves.size());
    const auto& nodes_yx = tensor_to_curve(nodes_yx_tensor);

    Scalars smoothKernel;
    if (expand > 0) smoothKernel = gaussianHalfKernel1D(expand, ceil(expand * 3) + 1);

    // === Compute Tips info ===
    struct TipInfo {
        IntPoint yx = IntPoint::Invalid();
        Point t = {0, 0};
        IntPointPair b = {IntPoint::Invalid(), IntPoint::Invalid()};
        float w = -1;
    };
    std::vector<std::array<TipInfo, 2>> tips(N_branches);

#pragma omp parallel for
    for (std::size_t branchID = 0; branchID < N_branches; ++branchID) {
        const CurveYX& curve = curves[branchID];
        const auto& boundary = boundariesAcc[branchID];
        std::vector<Point>& tangent = tangents[branchID];
        std::vector<float>& calibre = calibres[branchID];
        if (curve.size() != (std::size_t)boundary.size(0)) {
            throw std::runtime_error("Curve, boundary, and tangent sizes do not match for branch " +
                                     std::to_string(branchID));
        }
        if (expand > 0) {
            // If the tangent vector is not provided or has a different size, recompute it
            tangent.resize(curve.size());
            std::vector<Point> rawTangent;
            rawTangent.reserve(curve.size());
            std::vector<float> rawCalibre;
            rawCalibre.reserve(curve.size());
            for (std::size_t i = 0; i < curve.size(); ++i) {
                const auto& bounds = boundary[i];
                float w = distance(bounds[0], bounds[1]);
                rawTangent.push_back(adaptative_curve_tangent(curve, i, w, true, true).normalize());
                rawCalibre.push_back(w);
            }

            // Smooth the tangent vectors to avoid abrupt changes
            for (std::size_t i = 0; i < curve.size(); ++i) tangent[i] = smooth_tangents(rawTangent, i, smoothKernel);
            calibre = movingAvg(rawCalibre, smoothKernel);  // Smooth calibres as well
        }

        if (curve.size() == 0) continue;
        for (const auto headTip : {0, 1}) {
            std::size_t i = headTip ? curve.size() - 1 : 0;
            tips[branchID][headTip].yx = curve[i];

            if (expand <= 0) {
                IntPointPair b = {IntPoint(boundary[i][0]), IntPoint(boundary[i][1])};
                tips[branchID][headTip].b = b;
                float w = distance(b[0], b[1]);
                tips[branchID][headTip].t = adaptative_curve_tangent(curve, i, w, true, true).normalize();
                tips[branchID][headTip].w = w;
            } else {
                tips[branchID][headTip].t = tangent[i];
                tips[branchID][headTip].w = calibre[i] + 2 * expand;
                tips[branchID][headTip].b = curve[i].left_right_pair(tangent[i], calibre[i] * 0.5 + expand);
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
            const auto& tangent = tangents[branchID];
            const auto& calibre = calibres[branchID];
            const auto N = curve.size();

            // === DRAW THE BRANCH ===
            auto drawTopo = [&](IntPoint pt, float u, float d, int branchID, float rank) {
                d = 100 - d;  // Convert distance to fuzzy skeleton map value
                if (!pt.is_inside(maxShape) || fuzzySkeletonMapAcc[pt.y][pt.x] > d) return;
                float topoValue = rank + u;
                if (fuzzySkeletonMapAcc[pt.y][pt.x] == d && topoMapAcc[pt.y][pt.x] >= topoValue) return;

                topoMapAcc[pt.y][pt.x] = topoValue;
                fuzzySkeletonMapAcc[pt.y][pt.x] = d;
                branchLabelsMapAcc[pt.y][pt.x] = useBranchMapping ? branchMappingAcc[branchID + 1] : branchID + 1;
            };
            auto drawBranchTopo = [&](IntPoint pt, float u, float d) {
                drawTopo(pt, 0.1 + 0.85 * u, d, branchID, branch.rank);
            };
            if (N != 0) {  // If the branch is not empty rasterize it
                if (expand)
                    rasterize_branch_topo(curve, tangent, calibre, drawBranchTopo, maxShape, bezier_interpolate,
                                          expand);
                else
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

                // Decided weither to draw the junction to each side as a quad or a bezier cubic interpolation
                std::vector<int> near_children, far_children;
                for (const auto& childID : branch.children) {
                    const auto& childTip = tips[childID][0];
                    if (childTip.w < 0) continue;  // If the child tip is invalid, skip this filling
                    if (bezier_interpolate <= 0.0f || (distance(headTip.yx, childTip.yx) <= headTip.w * SQRT2))
                        near_children.push_back(childID);
                    else
                        far_children.push_back(childID);
                }

                // Draw the connection to near children as quads
                if (!near_children.empty()) {
                    // Compute the junction center by averaging the curve tip of near children
                    Point junctionBarycentre = headTip.yx * (headTip.w + 0.5f);
                    float weightSum = headTip.w + 0.5f;
                    for (const auto& childID : near_children) {
                        junctionBarycentre += tips[childID][0].yx * (tips[childID][0].w + 0.5f);
                        weightSum += tips[childID][0].w + 0.5f;
                    }
                    IntPoint junctionCenter = (junctionBarycentre / weightSum).toInt();

                    // Order children by clockwise position around the junction center
                    std::vector<std::pair<float, int>> near_angles;
                    Point v0 = headTip.yx - junctionCenter;
                    for (const auto& childID : near_children) {
                        const auto& childTip = tips[childID][0];
                        float angle = v0.angle(childTip.yx - junctionCenter);
                        if (angle < 0) angle += 2 * M_PI;
                        near_angles.emplace_back(angle, childID);
                    }
                    std::sort(near_angles.begin(), near_angles.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
                    near_children.clear();
                    for (const auto& [angle, childID] : near_angles) near_children.push_back(childID);

                    // Compute quads coordinates by a pairwise average of the boundary points of near children
                    std::vector<IntPoint> midB;
                    IntPoint prevB = headTip.b[0];
                    for (const auto& childID : near_children) {
                        const auto& childTip = tips[childID][0];
                        midB.push_back(((prevB + childTip.b[0]) / 2).toInt());
                        prevB = childTip.b[1];
                    }
                    midB.push_back(((prevB + headTip.b[1]) / 2).toInt());

                    // Draw the quad from the head tip to the junction center
                    auto drawJunctionHeadQuad = [&](const IntPoint& bound, const IntPoint& midBound) {
                        QuadIterator it(headTip.yx, bound, midBound, junctionCenter, maxShape);
                        if (!it.isConvex()) it = QuadIterator(headTip.yx, bound, bound, junctionCenter, maxShape);
                        it.precomputeInvDiffNorms();
                        while (it.iter()) {
                            const double u = it.fromP12toP34(), d = distance(headTip.yx, it.point());
                            drawTopo(it.point(), 0.95 + u * 0.05, d, branchID, branch.rank);
                        }
                    };
                    drawJunctionHeadQuad(headTip.b[0], midB.front());
                    drawJunctionHeadQuad(headTip.b[1], midB.back());

                    // Draw the quads from the junction center to each near child
                    auto drawJunctionChildQuad = [&](int32_t childID, IntPoint p, IntPoint bound, IntPoint midBound) {
                        QuadIterator it(p, bound, midBound, junctionCenter, maxShape);
                        if (!it.isConvex()) it = QuadIterator(p, bound, bound, junctionCenter, maxShape);
                        it.precomputeInvDiffNorms();
                        while (it.iter()) {
                            const double u = it.fromP12toP34(), d = distance(p, it.point());
                            drawTopo(it.point(), 0.1 * u, d, childID, branch.rank + 1);
                        }
                    };
                    for (std::size_t i = 0; i < near_children.size(); ++i) {
                        const auto& childID = near_children[i];
                        const auto& childTip = tips[childID][0];
                        drawJunctionChildQuad(childID, childTip.yx, childTip.b[0], midB[i]);
                        drawJunctionChildQuad(childID, childTip.yx, childTip.b[1], midB[i + 1]);
                    }
                }

                // Draw the bezier cubic interpolation from the head branch to each far child
                for (auto childID : far_children) {
                    const auto& childTip = tips[childID][0];
                    auto drawJunctionBezier = [&](IntPoint pt, float u, float d) {
                        drawTopo(pt, 0.1 * u, distance(pt, childTip.yx), childID, branch.rank + 1);
                    };
                    double d = distance(headTip.yx, childTip.yx) * bezier_interpolate;
                    const Point c0 = headTip.yx + headTip.t * d;
                    const Point c1 = childTip.yx - childTip.t * d;
                    rasterize_bezier(drawJunctionBezier, {headTip.yx, c0, c1, childTip.yx}, headTip.b, childTip.b,
                                     maxShape);
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
    auto drawBranchTopo = [&](IntPoint pt, float u, float d) {
        float topoValue = branchRank + 0.9 * u;
        if (topoMapAcc[pt.y][pt.x] >= topoValue) return;
        branchLabelsMapAcc[pt.y][pt.x] = branchID;
        topoMapAcc[pt.y][pt.x] = topoValue;
    };

    IntPoint maxShape = {(int)branchLabelsMap.size(0), (int)branchLabelsMap.size(1)};
    const auto& curve_vec = tensor_to_curve(curve);

    std::vector<Point> tangents_vec;

    return rasterize_branch_topo(curve_vec, boundaries.accessor<int, 3>(), drawBranchTopo, maxShape,
                                 bspline_interpolate);
}

void rasterize_bezier(std::function<void(IntPoint, float, float)> updater, const IntPoint& p0, const IntPoint& p1,
                      const Point& t0, const Point& t1, const IntPointPair& b0, const IntPointPair& b1,
                      float bezier_smoothness, const IntPoint& maxShape) {
    bezier_smoothness *= distance(Point(p0), Point(p1));
    BezierCubic bezier = {Point(p0), Point(p0) + t0 * bezier_smoothness, Point(p1) - t1 * bezier_smoothness, Point(p1)};
    rasterize_bezier(updater, bezier, b0, b1, maxShape);
}

void rasterize_bezier(std::function<void(IntPoint, float, float)> updater, const BezierCubic& bezier,
                      const IntPointPair& b0, const IntPointPair& b1, const IntPoint& maxShape) {
    float w0 = std::max(distance(b0[0], b0[1]), 1.0f), w1 = std::max(distance(b1[0], b1[1]), 1.0f);

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
        updater(p, u, 0);
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
                    while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP1toP4()), it.fromP14());
                    prev_b = b;
                }
                QuadIterator it(p, prev_b, nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP1toP4()), it.fromP14());
            } else {
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP12toP34()), it.fromP14());
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
                           std::function<void(IntPoint, float, float)> draw, const IntPoint& maxShape,
                           float bspline_interpolate) {
    auto N = curve.size();

    auto last = N - 1;
    IntPoint p = curve[0];
    IntPointPair b = {boundaries[0][0], boundaries[0][1]}, nextB;

    for (std::size_t i = 0; i != last; i++) {
        auto nextI = i + 1;
        const auto& nextP = curve[nextI];
        IntPointPair nextB = {boundaries[nextI][0], boundaries[nextI][1]};

        auto localDraw = [&](IntPoint pt, float u, float d) { draw(pt, (u + i) / N, d); };

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() <= 9) {
            for (int lr = 0; lr < 2; ++lr) {
                // Draw the center point
                localDraw(p, 0.0f, 0.0f);

                // Iterate over left and right quads
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) localDraw(it.point(), it.fromP12toP34(), it.fromP14());
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

void rasterize_branch_topo(const CurveYX& curve, const std::vector<Point>& tangents, const std::vector<float>& calibres,
                           std::function<void(IntPoint, float, float)> draw, const IntPoint& maxShape,
                           float bspline_interpolate, float expand) {
    auto N = curve.size();

    auto last = N - 1;
    IntPoint p = curve[0];
    IntPointPair b = p.left_right_pair(tangents[0], calibres[0] * 0.5 + expand, true), nextB;

    for (std::size_t i = 0; i != last; i++) {
        auto nextI = i + 1;
        const auto& nextP = curve[nextI];
        IntPointPair nextB = nextP.left_right_pair(tangents[nextI], calibres[nextI] * 0.5 + expand, true);

        auto localDraw = [&](IntPoint pt, float u, float d) { draw(pt, (u + i) / N, d); };

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() <= 9) {
            for (int lr = 0; lr < 2; ++lr) {
                // Draw the center point
                localDraw(p, 0.0f, 0.0f);

                // Iterate over left and right quads
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                if (expand > 0) it.mergeP2P3IfNotConvex();
                it.precomputeInvDiffNorms();
                while (it.iter()) localDraw(it.point(), it.fromP12toP34(), it.fromP14());
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

            rasterize_bezier([&](IntPoint pt, float u, float d) { out[pt.y][pt.x] = fill_value; }, p, nextP, t, nextT,
                             b, nextB, bspline_interpolate, maxShape);
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
      _p23Inverted(d32().is_null() || d14().dot(d32()) > 0),
      //_p34Inverted(d34().is_null() || d12().dot(d34()) > 0),
      _p1p4Adjacent(p1.is_adjacent(p4)) {
    int crossProd = d12().cross(d14());
    if (crossProd != 0)
        _positiveCrossProd = crossProd > 0;
    else
        _positiveCrossProd = d34().cross(d32()) > 0;

    // fastIt = false;
    // if (!_p34Inverted) {
    //     _it12 = RayIterator(pDiff[0].x >= 0 ? pDiff[0] : -pDiff[0]);
    //     _it34 = RayIterator(pDiff[2].x >= 0 ? pDiff[2] : -pDiff[2]);
    //     fastIt = false;  // _it12.octant() == it34.octant();
    //     // If the two rays are parallel we
    //     if (fastIt) {
    //         _it12.skip(_it12.stepsCountTo(pMin));
    //         _it = RayIterator(_it12.point(), 0, rotOctant45(_it12.octant(), 2));
    //         if (_it.stepsCountTo(_it34.point()) < 0) _it = _it.oppositeRay();
    //         _it.skip(-1);
    //     }
    // }
}

bool QuadIterator::finished() const {
    // Check if the iterator has finished iterating over the quad
    return p.y >= pMax.y && p.x >= pMax.x;
}

bool QuadIterator::iter() {
    while (true) {
        // === Move to the next point ===
        // if (fastIt) {
        //     _it.next();
        //     if (_it34.stepsCountTo(_it.point()) < 0) {
        //         if (_it12.stepsCountTo(pMax) <= 0) return false;
        //         _it12.next();
        //         _it34.next();
        //         _it.reset(_it12.point());
        //     }
        //     p = _it.point();
        //     if (!(p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y && p.y <= pMax.y)) continue;
        // } else {
        if (p.x < pMax.x) {
            p.x++;
        } else if (p.y < pMax.y) {
            p.x = pMin.x;  // Reset x to the minimum x value
            p.y++;         // Move to the next row
        } else {
            return false;  // No more points to iterate
        }
        //}

        // === Check if the point is inside the quad ===
        // Compute the cross products
        auto pp = p - p1;
        _crossProd[0] = d12().cross(pp);  //  (p2-p1) x (p-p1)
        // Early exit if the point is outside the first edge
        if (_positiveCrossProd ? _crossProd[0] < 0 : _crossProd[0] > 0) continue;

        // Early exit if the point is outside the last edge
        _crossProd[3] = -d14().cross(pp);  // -(p4-p1) x (p-p1)
        if (_positiveCrossProd ? _crossProd[3] < 0 : _crossProd[3] > 0) continue;

        pp = p - p3;
        int cross34 = d34().cross(pp);  //  (p4-p3) x (p-p3)
        // if (!_p34Inverted) {            // If the edge p3-p4 is not inverted check the point is on its correct side
        _crossProd[2] = cross34;
        if (_positiveCrossProd ? cross34 < 0 : cross34 > 0) continue;
        //} else
        //    _crossProd[2] = -cross34;  // Otherwise simply store the opposite value for fromP12toP34() computation

        if (!_p23Inverted) {
            _crossProd[1] = -d32().cross(pp);  // -(p2-p3) x (p-p3)
            if (_positiveCrossProd ? _crossProd[1] < 0 : _crossProd[1] > 0) continue;
        }

        return true;
    }
}

bool QuadIterator::isConvex() const {
    if (_positiveCrossProd) {  // d12 x d14 > 0
        if (d34().cross(d32()) < 0 || d32().cross(d12()) < 0 || d14().cross(d34()) < 0) return false;
    } else {  // d12 x d14 < 0
        if (d34().cross(d32()) > 0 || d32().cross(d12()) > 0 || d14().cross(d34()) > 0) return false;
    }
    return true;
}

bool QuadIterator::mergeP2P3IfNotConvex() {
    if (_positiveCrossProd ? d34().cross(d32()) < 0 : d34().cross(d32()) > 0) {
        // p3 is inside p1-p2-p4: move p3 to p2
        p3 = p2;
        pDiff[2] = p4 - p3;
    } else if (_positiveCrossProd ? d32().cross(d12()) < 0 : d32().cross(d12()) > 0) {
        // p2 is inside p1-p3-p4: move p2 to p3
        p2 = p3;
        pDiff[1] = p2 - p1;
    } else
        return false;
    pDiff[1] = {0, 0};  // Set p2-p3 to 0
    _p23Inverted = true;
    return true;
}

const IntPoint& QuadIterator::point() const { return p; }
const std::array<int, 4>& QuadIterator::crossProd() const { return _crossProd; }

double QuadIterator::fromP12toP34() const {
    double d = abs(cross12()) * invNorm12();
    double D = abs(cross34()) * invNorm34() + d;
    return D > 0 ? d / D : 0;
}

double QuadIterator::fromP14() const {
    // Compute the distance from point p to the line segment p1-p4
    if (_p1p4Adjacent) {
        // If p1 and p4 are adjacent, return the distance to the closest endpoint...
        int sqrNormP1 = (p - p1).squaredNorm(), sqrNormP4 = (p - p4).squaredNorm();
        return sqrNormP1 <= sqrNormP4 ? sqrt(sqrNormP1) : sqrt(sqrNormP4);
    } else {
        // ... otherwise find the closest point on the line segment p1-p4
        double t = clip((p - p1).normalize().dot(d14()) * invNorm14(), 0.0, 1.0);
        IntPoint closestPoint = p1 + (d14() * t).toInt();
        return (p - closestPoint).norm();
    }
}

double QuadIterator::fromP1toP4() const {
    IntPoint pp1 = p - p1;
    return clip(pp1.normalize().dot(d14()) * invNorm14(), 0.0, 1.0);
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