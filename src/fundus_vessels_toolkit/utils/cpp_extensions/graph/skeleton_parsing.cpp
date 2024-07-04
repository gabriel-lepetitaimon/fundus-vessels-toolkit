#include "skeleton.h"

/********************************************************************************************
 *                                 DETECT SKELETON NODES                                    *
 *******************************************************************************************/

enum SkeletonRank {
    NONE = 0,
    ENDPOINT = 1,
    BRANCH = 2,
    JUNCTION3 = 3,
    JUNCTION4 = 4,
    HOLLOW_CROSS = 5,
};

/**
 * @brief Detect junction and endpoints base on neighborhood.
 *
 *
 * @param neighbors Value of the 8 neighbors of the pixel. The top-left neighbor is the most significant bit.
 * @return int 1 if it is an endpoint, 3 or 4 if it is a junction, 2 otherwise.
 */
SkeletonRank detect_skeleton_rank(const bool pixelValue, const uint8_t neighbors) {
    if (!pixelValue) {
        return hit(neighbors, 0b01010101) ? SkeletonRank::HOLLOW_CROSS : SkeletonRank::NONE;
    }

    // -- Single endpoints --
    if (neighbors == 0) return SkeletonRank::ENDPOINT;

    for (int s = 0; s < 8; s++) {
        const uint8_t neighbors_rolled = roll_neighbors(neighbors, s);

        // -- Endpoints --
        if (hit_and_miss(neighbors_rolled, 0b01000000, 0b00011111)) return SkeletonRank::ENDPOINT;

        // -- 3 branches junctions --
        // Y shape
        if (hit(neighbors_rolled, 0b10100100)) {
            if (miss(neighbors_rolled, 0b01010001) || miss(neighbors_rolled, 0b01001010))
                return SkeletonRank::JUNCTION3;
        }

        // T shape
        if (hit_and_miss(neighbors_rolled, 0b00010101, 0b01001010)) return SkeletonRank::JUNCTION3;
    }

    // -- 4 branches junctions --
    if (hit_and_miss(neighbors, 0b10101010, 0b01010101) || hit_and_miss(neighbors, 0b01010101, 0b10101010) ||
        hit(neighbors, 0b00011100))
        return SkeletonRank::JUNCTION4;

    return SkeletonRank::BRANCH;
}

/**
 * @brief Fix the skeleton of a binary image.
 *
 * @param skeleton Binary 2d image representing the skeleton.
 * @param fix_hollow If true, fill the hollow crosses of the skeleton.
 * @param remove_single_endpoints If true, remove terminal branches made of a single point.
 * @return torch::Tensor The fix skeleton were endpoints are set to 1, junctions are set to 2 or 3 and the rest of the
 * skeleton is set to 2.
 *
 */
torch::Tensor detect_skeleton_nodes(torch::Tensor skeleton, bool fix_hollow, bool remove_single_endpoints) {
    skeleton = torch::constant_pad_nd(skeleton, {1, 1, 1, 1}, 0);
    auto skeleton_accessor = skeleton.accessor<bool, 2>();
    int H = skeleton.size(0), W = skeleton.size(1);

    torch::Tensor parsed_skeleton = torch::zeros_like(skeleton, torch::kInt);
    auto parsed_accessor = parsed_skeleton.accessor<int, 2>();

    std::vector<IntPoint> hollow_crosses;
    std::vector<IntPoint> endpoints;

#pragma omp parallel for collapse(2) reduction(merge : endpoints, hollow_crosses)
    for (int y = 0; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            const uint8_t neighbors = get_neighborhood(skeleton_accessor, y, x);
            const SkeletonRank point = detect_skeleton_rank(skeleton_accessor[y][x], neighbors);
            if (point == SkeletonRank::HOLLOW_CROSS) {
                if (fix_hollow) hollow_crosses.push_back({y, x});
            } else if (remove_single_endpoints && point == SkeletonRank::ENDPOINT) {
                endpoints.push_back({y, x});
            }
            parsed_accessor[y][x] = point;
        }
    }

    // Fill Hollow Crosses and decrement their neighbors
    for (const IntPoint& p : hollow_crosses) {
        parsed_accessor[p.y][p.x] = SkeletonRank::BRANCH;
        for (const PointWithID& n : NEIGHBORHOOD) {
            if (parsed_accessor[p.y + n.y][p.x + n.x] > 0) parsed_accessor[p.y + n.y][p.x + n.x]--;
        }
    }

    // Detect skeleton point again on hollow crosses
    for (const IntPoint& p : hollow_crosses) {
        const uint8_t neighbors = get_neighborhood(parsed_accessor, p.y, p.x);
        parsed_accessor[p.y][p.x] = detect_skeleton_rank(parsed_accessor[p.y][p.x], neighbors);
    }

// Remove single endpoints
#pragma omp parallel for
    for (const IntPoint& p : endpoints) {
        bool singleEndpoint = true;
        for (const PointWithID& n : NEIGHBORHOOD) {
            int n_rank = parsed_accessor[p.y + n.y][p.x + n.x];
            if (n_rank >= SkeletonRank::JUNCTION3) {
                parsed_accessor[p.y + n.y][p.x + n.x] = n_rank - 1;
                singleEndpoint = true;
                break;
            } else if (n_rank > SkeletonRank::ENDPOINT) {
                singleEndpoint = false;
                break;
            }
        }
        if (singleEndpoint) parsed_accessor[p.y][p.x] = SkeletonRank::NONE;
    }

    return parsed_skeleton.index({torch::indexing::Slice(1, -1), torch::indexing::Slice(1, -1)});
}

/********************************************************************************************
 *                                      SKELETON PARSING                                    *
 *******************************************************************************************/

enum PixelRole {
    BACKGROUND = 0,
    BRANCH_ROLE = 1,
    NODE = 2,
};

inline bool is_node(int pixelRole) { return pixelRole >= PixelRole::NODE; }

/**
 * @brief Track the next neighbors of a pixel in a skeleton.
 *
 * @param skeleton 2d image representing the skeleton with identified junctions and branches.
 * @param current Coordinates of the pixel.
 * @param ignore_neighbor Id of the neighbor to ignore.
 * @param ignore_branches_close_to_node If true, ignore branches close to a node.
 * @return std::pair<std::vector<PointWithID>, std::vector<PointWithID>>
 *     - The coordinates of neighbor nodes.
 *     - The coordinates of neighbor branches.
 */
std::pair<std::vector<IntPoint>, std::vector<PointWithID>> get_node_neighbors(Tensor2DAccessor<int> skeleton,
                                                                              IntPoint current, IntPoint shape) {
    std::vector<IntPoint> nodes;
    std::vector<PointWithID> branches;
    std::array<bool, 8> is_node = {false, false, false, false, false, false, false, false};

    for (auto it = CLOSE_NEIGHBORHOOD.begin(), end = CLOSE_NEIGHBORHOOD.begin() + 4; it != end; ++it) {
        const IntPoint n = *it + current;
        int n_id = it->id;

        if (!n.is_inside(shape)) continue;

        int pixelRole = skeleton[n.y][n.x];
        if (pixelRole == PixelRole::BACKGROUND)
            continue;
        else if (pixelRole == PixelRole::BRANCH_ROLE) {
            branches.push_back({n.y, n.x, n_id});
        } else {
            nodes.push_back({n.y, n.x});
            is_node[n_id] = true;
        }
    }

    for (auto it = CLOSE_NEIGHBORHOOD.begin() + 4, end = CLOSE_NEIGHBORHOOD.end(); it != end; ++it) {
        const IntPoint n = *it + current;
        int n_id = it->id;

        if (!n.is_inside(shape) || is_node[prev_neighbor_id(n_id)] || is_node[next_neighbor_id(n_id)]) continue;

        int pixelRole = skeleton[n.y][n.x];
        if (pixelRole == PixelRole::BACKGROUND)
            continue;
        else if (pixelRole == PixelRole::BRANCH_ROLE) {
            branches.push_back({n.y, n.x, n_id});
        } else
            nodes.push_back({n.y, n.x});
    }
    return {nodes, branches};
}

/**
 * @brief Track the next neighbors of a pixel in a skeleton.
 *
 * @param skeleton 2d image representing the skeleton with identified junctions and branches.
 * @param current Coordinates of the pixel.
 * @param ignore_neighbor Id of the neighbor to ignore.
 * @param ignore_branches_close_to_node If true, ignore branches close to a node.
 * @return std::pair<std::vector<PointWithID>, std::vector<PointWithID>>
 *     - The coordinates of neighbor nodes.
 *     - The coordinates of neighbor branches.
 */
std::pair<PointWithID, int> track_branch_neighbors(Tensor2DAccessor<int> skeleton, PointWithID current,
                                                   IntPoint shape) {
    PointWithID nextBranchPoint = {0, 0, -1};
    for (const int& n_id : TRACK_NEXT_NEIGHBORS[current.id]) {
        const IntPoint n = NEIGHBORHOOD[n_id] + current;
        if (!n.is_inside(shape)) continue;

        int pixelRole = skeleton[n.y][n.x];
        if (pixelRole == PixelRole::NODE) {
            return {{n.y, n.x, n_id}, pixelRole};
        } else if (nextBranchPoint.id == -1 && pixelRole == PixelRole::BRANCH_ROLE) {
            nextBranchPoint = {n.y, n.x, n_id};
        }
    }
    if (nextBranchPoint.id == -1) {
        return {nextBranchPoint, PixelRole::BACKGROUND};
    } else {
        return {nextBranchPoint, PixelRole::BRANCH_ROLE};
    }
}

/**
 * @brief Parse a skeleton image into a graph of branches and nodes.
 *
 * @param skeleton 2d image representing the skeleton with identified junctions and branches.
 *
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 *     - The labeled map of branches and nodes
 *     - The coordinates of the nodes
 *     - The edge list. Each edge is composed of the id of the two nodes and the id of the branch connecting them.
 */
std::tuple<EdgeList, std::vector<CurveYX>, std::vector<IntPoint>> parse_skeleton_to_graph(torch::Tensor& skeletonMap) {
    // The skeletonMap will be modified with the labelled skeleton.
    auto const& skeleton = skeletonMap.clone();
    auto skel_acc = skeleton.accessor<int, 2>();
    auto label_acc = skeletonMap.accessor<int, 2>();

    IntPoint shape = {(int)skeletonMap.size(0), (int)skeletonMap.size(1)};
    std::vector<CurveYX> branches_curves;
    std::vector<PointWithID> nodes;
    int nNodes = 0;

// -- Search junctions and endpoints --
#pragma omp parallel for collapse(2) reduction(merge : nodes)
    for (int y = 1; y < shape.y - 1; y++) {
        for (int x = 1; x < shape.x - 1; x++) {
            const int pixelRole = skel_acc[y][x];
            // Erase skeleton from the skeletonMap for later annotations
            if (pixelRole == PixelRole::BACKGROUND) continue;
            if (pixelRole == PixelRole::BRANCH_ROLE) {
                label_acc[y][x] = 0;
                continue;
            }

            int node_id;
#pragma omp atomic capture
            {
                node_id = nNodes;
                nNodes++;
            }
            // Junctions
            nodes.push_back({y, x, node_id});
            label_acc[y][x] = -node_id - 1;
        }
    }

    // -- Search branches --
    // int nBranches = 0;
    EdgeList edge_list;
    edge_list.reserve(nNodes);
    std::vector<IntPoint> nodes_coordinates(nNodes);

    for (const PointWithID& node : nodes) {
        // Save node coordinates
        nodes_coordinates[node.id] = node;

        // -- Find next nodes and branches --
        auto const& [next_nodes, next_branches] = get_node_neighbors(skel_acc, {node.y, node.x}, shape);

        // Remember connection with nodes next to the current node
        for (const IntPoint& n : next_nodes) {
            int neighbor_node_id = -label_acc[n.y][n.x] - 1;
            if (node.id < neighbor_node_id) {
                edge_list.push_back({node.id, neighbor_node_id, (int)edge_list.size()});
                branches_curves.push_back({{node.y, node.x}, {n.y, n.x}});
            }
        }

        // Track branches not yet labelled
        for (const PointWithID& n : next_branches) {
            if (label_acc[n.y][n.x] == 0) {
                const int branch_id = edge_list.size();
                CurveYX branch_pixels;
                branch_pixels.reserve(64);
                branch_pixels.push_back({node.y, node.x});

                PointWithID tracker = n;
                int tracker_role = PixelRole::BRANCH_ROLE;

                while (label_acc[tracker.y][tracker.x] == 0) {
                    // Label the branch
                    label_acc[tracker.y][tracker.x] = branch_id + 1;
                    branch_pixels.push_back(tracker);

                    // Track next nodes and branches
                    std::tie(tracker, tracker_role) = track_branch_neighbors(skel_acc, tracker, shape);

                    if (tracker_role != PixelRole::BRANCH_ROLE) break;  // If node is found stop the tracking
                }
                // Save the branch curve
                branch_pixels.push_back({tracker.y, tracker.x});
                branches_curves.push_back(branch_pixels);

                if (tracker_role == PixelRole::NODE) {
                    // Save connectivity
                    const int other_node_id = -label_acc[tracker.y][tracker.x] - 1;
                    edge_list.push_back({node.id, other_node_id, branch_id});
                } else {
                }
            }
        }
    }

    return {edge_list, branches_curves, nodes_coordinates};
}