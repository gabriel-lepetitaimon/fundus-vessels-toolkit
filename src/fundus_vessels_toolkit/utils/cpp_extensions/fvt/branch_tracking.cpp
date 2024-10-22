#include "branch.h"
#include "ray_iterators.h"

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
std::vector<std::array<IntPoint, 2>> find_branch_endpoints(
    const torch::Tensor &branch_labels, const torch::Tensor &node_yx,
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
      bool first_endpoint =
          first_endpoints[b - 1].test_and_set();  // TODO: Check with openmp
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
std::vector<CurveYX> track_branches(const torch::Tensor &branch_labels,
                                    const torch::Tensor &node_yx,
                                    const torch::Tensor &branch_list) {
  const int B = branch_list.size(0);
  const int H = branch_labels.size(0), W = branch_labels.size(1);
  auto bLabels_acc = branch_labels.accessor<int, 2>();

  // Find the branch start and endpoints
  auto const &branch_endpoints =
      find_branch_endpoints(branch_labels, node_yx, branch_list);
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
 * @brief Track the not-segmented pixel on a semi-infinite line defined by a
 * start point and a direction.
 *
 * @param start The start point of the line.
 * @param direction The direction of the line.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the
 * binary segmentation.
 * @param max_distance The maximum distance to track.
 *
 * @return The first point of the line for which the segmentation is false. If
 * no such point is found, return IntPoint::Invalid().
 */
IntPoint track_nearest_edge(const IntPoint &start, const Point &direction,
                            const Tensor2DAcc<bool> &segmentation,
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
  } while (i < max_distance && next.is_inside(H, W) &&
           segmentation[next.y][next.x]);
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
 * @param findFirstLocalMinimum If true, the search stops at the first local
 * minimum.
 *
 * @return A tuple containing the index of the closest pixel and the distance to
 * the point.
 */
std::tuple<int, float> find_closest_pixel(const CurveYX &curve, const Point &p,
                                          int start, int end,
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

std::pair<torch::Tensor, torch::Tensor> find_closest_branches(
    const torch::Tensor &branch_labels, const torch::Tensor &points,
    const torch::Tensor &direction, float max_dist, float angle) {
  TORCH_CHECK(
      points.ndimension() == 2 && points.size(1) == 2,
      "Invalid argument points: should have a shape of (N, 2) instead of",
      points.sizes());
  const std::size_t N = points.size(0);
  TORCH_CHECK(direction.ndimension() == 2 &&
                  (std::size_t)direction.size(0) == N && direction.size(1) == 2,
              "Invalid argument direction: should have a shape of (", N,
              ", 2) instead of", direction.sizes());

  auto points_acc = points.accessor<int, 2>();
  auto direction_acc = direction.accessor<float, 2>();
  auto branch_labels_acc = branch_labels.accessor<int, 2>();
  std::vector<std::pair<uint, IntPoint>> out(N);

  auto branch = torch::empty({(long)N}, torch::kInt);
  auto intercept = torch::empty({(long)N, 2}, torch::kInt);
  auto branch_acc = branch.accessor<int, 1>();
  auto intercept_acc = intercept.accessor<int, 2>();

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i++) {
    auto start = IntPoint(points_acc[i][0], points_acc[i][1]);
    auto dir = Point(direction_acc[i][0], direction_acc[i][1]);
    const auto &[b, p] =
        track_nearest_branch(start, dir, angle, max_dist, branch_labels_acc);
    branch_acc[i] = b - 1;
    intercept_acc[i][0] = p.y;
    intercept_acc[i][1] = p.x;
  }

  return {branch, intercept};
}

/**
 * @brief Split a curve into contiguous curves.
 *
 * All points of the returned contiguous curves are adjacent (horizontally,
 * vertically or diagonally).
 *
 * @param curve The curve to split.
 *
 * @return A list of contiguous curves.
 *
 */
std::list<SizePair> split_contiguous_curves(const CurveYX &curve) {
  if (curve.size() < 2) return {{0, curve.size()}};

  std::list<SizePair> curvesBoundaries;
  std::size_t start = 0;
  for (auto it = curve.begin() + 1; it != curve.end(); it++) {
    auto const &&diff = *it - *(it - 1);
    if (abs(diff.x) > 1 || abs(diff.y) > 1) {
      curvesBoundaries.push_back(
          SizePair{start, (std::size_t)(it - curve.begin())});
      start = it - curve.begin();
    }
  }
  curvesBoundaries.push_back(SizePair{start, curve.size()});

  return curvesBoundaries;
}

/*
 * @brief Draw the branches on a tensor.
 *
 * This method draws the branches on a tensor. The branches are represented by
 * a list of curves and a list of nodes. The branches are drawn as a line
 * between the nodes.
 *
 * @param branchCurves A list of uint tensor of shape (N, 2) representing the
 * branch curves.
 * @param shape The shape of the output tensor.
 * @param nodeCoords A list of nodes coordinates.
 * @param branchList A list of branches.
 *
 * @return A tensor with the branches drawn.
 */
torch::Tensor draw_branches_labels(
    const std::vector<torch::Tensor> &branchCurves, const torch::Tensor &out,
    const torch::Tensor &nodeCoords, const torch::Tensor &branchList) {
  const std::size_t B = branchCurves.size();
  auto branchesLabels = out.accessor<int, 2>();

  for (std::size_t b = 0; b < B; b++) {
    const auto &curveT = branchCurves[b];
    TORCH_CHECK(curveT.ndimension() == 2 && curveT.size(1) == 2,
                "Invalid argument branchCurves: branch ", b,
                " should have a shape of (N, 2) instead of ", curveT.sizes());
    const auto &curve = curveT.accessor<int, 2>();
    const std::size_t N = curve.size(0);
    for (std::size_t i = 0; i < N; i++)
      branchesLabels[curve[i][0]][curve[i][1]] = b + 1;
  }

  if (nodeCoords.numel() == 0 || branchList.numel() == 0) return out;

  TORCH_CHECK(
      branchList.ndimension() == 2 && branchList.size(1) == 2,
      "Invalid argument branchList: should have a shape of (B, 2) instead of",
      branchList.sizes());
  TORCH_CHECK(
      nodeCoords.ndimension() == 2 && nodeCoords.size(1) == 2 &&
          nodeCoords.size(0) >= branchList.max().item<int>(),
      "Invalid argument branchList: should have a shape of (B, 2) instead of",
      branchList.sizes());
  auto branchList_acc = branchList.accessor<int, 2>();
  auto nodeCoords_acc = nodeCoords.accessor<int, 2>();

  for (std::size_t b = 0; b < B; b++) {
    const IntPoint start = {nodeCoords_acc[branchList_acc[b][0]][0],
                            nodeCoords_acc[branchList_acc[b][0]][1]};
    const IntPoint end = {nodeCoords_acc[branchList_acc[b][1]][0],
                          nodeCoords_acc[branchList_acc[b][1]][1]};
    const auto &curve = branchCurves[b].accessor<int, 2>();
    if (curve.size(0) == 0) {
      RayIterator ray(start, end - start);
      int i = ray.stepTo(end);
      while (i-- != 0) {
        const IntPoint &p = ++ray;
        branchesLabels[p.y][p.x] = b + 1;
      }
    } else {
      const IntPoint p1 = {curve[0][0], curve[0][1]};
      const IntPoint p2 = {curve[curve.size(0) - 1][0],
                           curve[curve.size(0) - 1][1]};
      if (p1 != start) {
        RayIterator ray(start, p1 - start);
        int i = ray.stepTo(p1);
        while (i-- != 0) {
          const IntPoint &p = ++ray;
          branchesLabels[p.y][p.x] = b + 1;
        }
      }
      if (p2 != end) {
        RayIterator ray(end, p2 - end);
        int i = ray.stepTo(p2);
        while (i-- != 0) {
          const IntPoint &p = ++ray;
          branchesLabels[p.y][p.x] = b + 1;
        }
      }
    }
  }

  return out;
}
