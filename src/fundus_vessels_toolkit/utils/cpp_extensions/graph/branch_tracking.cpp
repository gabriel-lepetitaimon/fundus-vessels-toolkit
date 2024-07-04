#include "branch.h"

/**
 * @brief Find the first and last endpoint of each branch.
 *
 * This method compute returns, for each branches, a pair of Point representing
 * the first and last pixels of the branch. The first and last pixels are
 * defined following the branch_list arguments:
 *   - the first pixel of the branch i is the branch endpoint, which is closest
 * to the node given by branch_list[i][0].
 *   - the last pixel of the branch i is the branch endpoint, which is closest
 * to the node given by branch_list[i][1]. In case of conflict, the closest
 * pixel wins.
 *
 * @param branch_labels A 2D tensor of shape (H, W) containing the branch
 * labels.
 * @param node_yx A 2D tensor of shape (N, 2) containing the coordinates of the
 * nodes.
 * @param branch_list A 2D tensor of shape (B, 2) containing the list of
 * branches.
 *
 * @return A list of pairs of points representing the first and last pixels of
 * the branches.
 */
std::vector<std::array<IntPoint, 2>> find_branch_endpoints(const torch::Tensor &branch_labels,
                                                           const torch::Tensor &node_yx,
                                                           const torch::Tensor &branch_list) {
    const int B = branch_list.size(0);
    const int H = branch_labels.size(0), W = branch_labels.size(1);
    auto bLabels_acc = branch_labels.accessor<int, 2>();
    auto node_yx_acc = node_yx.accessor<int, 2>();
    auto branch_list_acc = branch_list.accessor<int, 2>();

    // Find the branch start and endpoints
    std::vector<std::array<IntPoint, 2>> branches_endpoints(B);
    std::vector<std::atomic_flag> first_endpoints(B);

// -- Search junctions and endpoints --
#pragma omp parallel for collapse(2)
    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            const int b = bLabels_acc[y][x];

            // If the current pixel is part of a branch ...
            if (b == 0) continue;

            // ... check if it's an endpoint.
            bool is_endpoint = false;
            for (const PointWithID &n : NEIGHBORHOOD) {
                if (bLabels_acc[y + n.y][x + n.x] == b) {
                    if (!is_endpoint)
                        is_endpoint = true;  // First neighbor found
                    else {
                        is_endpoint = false;  // Second neighbor found
                        break;                // => not an endpoint
                    }
                }
            }
            if (!is_endpoint) continue;

            // If it's an endpoint, store it.
            bool first_endpoint = first_endpoints[b - 1].test_and_set();  // TODO: Check with openmp
            branches_endpoints[b - 1][first_endpoint] = {y, x};
        }
    }

// -- Ensure the first endpoints is the closest to the first node --
#pragma omp parallel for
    for (int b = 0; b < B; b++) {
        int node0_id = branch_list_acc[b][0], node1_id = branch_list_acc[b][1];
        const IntPoint n0 = {node_yx_acc[node0_id][0], node_yx_acc[node0_id][1]};
        const IntPoint n1 = {node_yx_acc[node1_id][0], node_yx_acc[node1_id][1]};
        const IntPoint p0 = branches_endpoints[b][0];
        const IntPoint p1 = branches_endpoints[b][1];

        const float min_dist = std::min(distance(p0, n0), distance(p1, n1));
        // If the distance n1p0 or n0p1 is shorter than n1p1 and n0p0 ...
        if (distance(n0, p1) < min_dist || distance(n1, p0) < min_dist) {
            // ... swap the endpoints.
            branches_endpoints[b][0] = p1;
            branches_endpoints[b][1] = p0;
        }
    }

    return branches_endpoints;
}

/**
 * @brief Track the orientation of branches in a vessel graph.
 *
 * This method compute returns, for each branches, the list of the coordinates
 * of its pixels. The first and last pixels are defined following the
 * branch_list arguments:
 *   - the first pixel of the branch i is the branch endpoint, which is closest
 * to the node given by branch_list[i][0].
 *   - the last pixel of the branch i is the branch endpoint, which is closest
 * to the node given by branch_list[i][1]. In case of conflict, the closest
 * pixel wins.
 *
 * @param branch_labels A 2D tensor of shape (H, W) containing the branch
 * labels.
 * @param node_yx A 2D tensor of shape (N, 2) containing the coordinates of the
 * nodes.
 * @param branch_list A 2D tensor of shape (B, 2) containing the list of
 * branches.
 *
 * @return A list of the yx coordinates of the branches pixels.
 */
std::vector<CurveYX> track_branches(const torch::Tensor &branch_labels, const torch::Tensor &node_yx,
                                    const torch::Tensor &branch_list) {
    const int B = branch_list.size(0);
    const int H = branch_labels.size(0), W = branch_labels.size(1);
    auto bLabels_acc = branch_labels.accessor<int, 2>();

    // Find the branch start and endpoints
    auto const &branch_endpoints = find_branch_endpoints(branch_labels, node_yx, branch_list);
    std::vector<CurveYX> branches_pixels(B);

// Track the branches
#pragma omp parallel for
    for (int b = 0; b < B; b++) {
        const IntPoint start_p = branch_endpoints[b][0];
        const IntPoint end_p = branch_endpoints[b][1];
        const int bLabel = b + 1;

        // Initialize the branch pixels list with the start pixel
        CurveYX *branch_pixels = &branches_pixels[b];
        branch_pixels->reserve(16);
        branch_pixels->push_back(start_p);

        PointWithID current_p = start_p;
        for (const PointWithID &n : NEIGHBORHOOD) {
            const IntPoint neighbor = start_p + n;
            if (!neighbor.is_inside(H, W)) continue;
            if (bLabels_acc[neighbor.y][neighbor.x] == bLabel) {
                current_p = PointWithID(neighbor, n.id);
                break;
            }
        }

        while (!current_p.isSamePoint(end_p)) {
            branch_pixels->push_back(current_p);

            // Track the next pixel of the branch...
            //  (TRACK_NEXT_NEIGHBORS is used to avoid tracking back and to favor the
            //  pixels diametrically opposed to the previous one.)
            for (const int &n_id : TRACK_NEXT_NEIGHBORS[current_p.id]) {
                const IntPoint neighbor = NEIGHBORHOOD[n_id] + current_p;
                if (!neighbor.is_inside(H, W)) continue;
                if (bLabels_acc[neighbor.y][neighbor.x] == bLabel) {
                    current_p = PointWithID(neighbor, n_id);
                    break;
                }
            }
            // ... and store it.
        }

        // Append the final pixel
        branch_pixels->push_back(end_p);
    }

    return branches_pixels;
}

/**
 * @brief Track the not-segmented pixel on a semi-infinite line defined by a start point and a direction.
 *
 * @param start The start point of the line.
 * @param direction The direction of the line.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the binary segmentation.
 * @param max_distance The maximum distance to track.
 *
 * @return The first point of the line for which the segmentation is false. If no such point is found, return
 * IntPoint::Invalid().
 */
IntPoint track_nearest_border(const IntPoint &start, const Point &direction, const Tensor2DAccessor<bool> &segmentation,
                              int max_distance) {
    if (direction.is_null()) return IntPoint::Invalid();
    const int H = segmentation.size(0), W = segmentation.size(1);

    Point current = {(float)start.y, (float)start.x};
    IntPoint next = start;
    IntPoint last;
    int i = 0;
    do {
        last = next;
        current += direction;
        next = current.toInt();
        i++;
    } while (i < max_distance && next.is_inside(H, W) && segmentation[next.y][next.x]);
    if (i == max_distance) return IntPoint::Invalid();
    return last;
}

/**
 * @brief Find the closest pixel to a point in a curve.
 *
 * This method returns the index of the closest pixel to a point in a curve.
 * The search is performed between the start and end indexes.
 *
 * @param curve A list of points defining the curve.
 * @param p The point to which the distance should be computed.
 * @param start The start index of the search.
 * @param end The end index of the search.
 * @param findFirstLocalMinimum If true, the search stops at the first local minimum.
 *
 * @return A tuple containing the index of the closest pixel and the distance to the point.
 */
std::tuple<int, float> findClosestPixel(const CurveYX &curve, const Point &p, int start, int end,
                                        bool findFirstLocalMinimum) {
    const int inc = (start < end) ? 1 : -1;
    std::tuple<int, float> min_point = {0, distance(curve[start], p)};
    for (int i = start + inc; i != end; i += inc) {
        float dist = distance(curve[i], p);
        if (dist <= std::get<1>(min_point)) {
            min_point = {i, dist};
        } else if (findFirstLocalMinimum) {
            break;
        }
    }
    return min_point;
}

/**
 * @brief Split a curve into contiguous curves.
 *
 * All points of the returned contiguous curves are adjacent (horizontally, vertically or diagonally).
 *
 * @param curve The curve to split.
 *
 * @return A list of contiguous curves.
 *
 */
std::list<SizePair> splitInContiguousCurves(const CurveYX &curve) {
    if (curve.size() < 2) return {{0, curve.size()}};

    std::list<SizePair> curvesBoundaries;
    std::size_t start = 0;
    for (auto it = curve.begin() + 1; it != curve.end(); it++) {
        auto const &&diff = *it - *(it - 1);
        if (abs(diff.x) > 1 || abs(diff.y) > 1) {
            curvesBoundaries.push_back(SizePair{start, (std::size_t)(it - curve.begin())});
            start = it - curve.begin();
        }
    }
    curvesBoundaries.push_back(SizePair{start, curve.size()});

    return curvesBoundaries;
}