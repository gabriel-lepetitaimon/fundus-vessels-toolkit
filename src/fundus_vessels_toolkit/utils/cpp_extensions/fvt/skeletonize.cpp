#include "skeleton.h"

constexpr bool is_simple_point(const uint8_t& neighborhood) {
    // A point is simple if its removal does not change the topology of the object.
    if (count_neighbors(neighborhood) <= 1) return false;  // Endpoints are not simple points

    // Find first neighbor (the first bit set in the neighborhood)
    uint8_t step = 0b10000000;
    while (step > 0 && (neighborhood & step) == 0) step >>= 1;

    // Propagate through the neighbors with the 4/8 connectivity rule until no new neighbors are found
    // 0 — 1 — 2
    // | /   \ |
    // 7       3
    // | \   / |
    // 6 — 5 — 4
    uint8_t next_step = step;
    do {
        step = next_step;
        if (step & 0b10000000) next_step |= 0b01000001;  // 0 -> 1, 7
        if (step & 0b01000000) next_step |= 0b10110001;  // 1 -> 0, 2, 3, 7
        if (step & 0b00100000) next_step |= 0b01010000;  // 2 -> 1, 3
        if (step & 0b00010000) next_step |= 0b01101100;  // 3 -> 1, 2, 4, 5
        if (step & 0b00001000) next_step |= 0b00010100;  // 4 -> 3, 5
        if (step & 0b00000100) next_step |= 0b00011011;  // 5 -> 3, 4, 6, 7
        if (step & 0b00000010) next_step |= 0b00000101;  // 6 -> 5, 7
        if (step & 0b00000001) next_step |= 0b11000110;  // 7 -> 0, 1, 5, 6
        next_step &= neighborhood;                       // Keep only the neighbors that are present
    } while (next_step != step);

    // Check if all neighbors are connected (i.e., if the propagated step equals the original neighborhood)
    return step == neighborhood;
}

constexpr std::array<bool, 256> generate_simple_point_lookup() {
    std::array<bool, 256> lookup{};
    for (uint16_t i = 0; i < 256; ++i) {
        lookup[i] = is_simple_point(static_cast<uint8_t>(i));
    }
    return lookup;
}
static constexpr std::array<bool, 256> IS_SIMPLE_POINT_LOOKUP = generate_simple_point_lookup();

template <typename T>
uint8_t get_neighborhood(const Tensor2DAcc<T>& z, int y, int x, std::function<bool(const T&, const T&)> is_same) {
    uint8_t neighbors = 0;
    const uint8_t& a = z[y][x];
    if (is_same(a, z[y - 1][x - 1])) neighbors |= 0b10000000;
    if (is_same(a, z[y - 1][x - 0])) neighbors |= 0b01000000;
    if (is_same(a, z[y - 1][x + 1])) neighbors |= 0b00100000;
    if (is_same(a, z[y - 0][x + 1])) neighbors |= 0b00010000;
    if (is_same(a, z[y + 1][x + 1])) neighbors |= 0b00001000;
    if (is_same(a, z[y + 1][x + 0])) neighbors |= 0b00000100;
    if (is_same(a, z[y + 1][x - 1])) neighbors |= 0b00000010;
    if (is_same(a, z[y + 0][x - 1])) neighbors |= 0b00000001;
    return neighbors;
}

void connected_components(const torch::Tensor& segMap, std::function<bool(const uint8_t&)> is_not_null,
                          std::vector<std::vector<IntPoint>>& outComponents) {
    TORCH_CHECK_VALUE(segMap.dim() == 2, "segMap must be a 2D tensor");
    TORCH_CHECK_VALUE(segMap.scalar_type() == torch::kUInt8, "segMap must be a uint8 tensor");

    const int H = segMap.size(0), W = segMap.size(1);
    auto seg = segMap.accessor<uint8_t, 2>();

    std::vector<std::vector<IntPoint>> components;
    std::vector<IntPair> componentAdjacency;
    torch::Tensor ccMap = torch::zeros({H, W}, torch::dtype(torch::kInt));
    auto cc = ccMap.accessor<int, 2>();

    // --- Lambda functions: Assign to new component ---
    auto assignNew = [&](const IntPoint& p) {
        cc[p.y][p.x] = components.size();
        components.emplace_back();
        components.back().push_back(p);
    };
    // --- Lambda function: Assign to existing component ---
    auto assignExisting = [&](const IntPoint& p, int compIdx) {
        cc[p.y][p.x] = compIdx;
        components[compIdx].push_back(p);
    };

    // === Process top row and left column first to avoid out-of-bounds checks in the main loop ===
    if (is_not_null(seg[0][0])) assignNew(IntPoint(0, 0));
    for (int x = 1; x < W; ++x) {
        if (is_not_null(seg[0][x])) {
            const auto& leftCC = cc[0][x - 1];
            if (leftCC == 0)
                assignNew(IntPoint(0, x));
            else
                assignExisting(IntPoint(0, x), leftCC);
        }
    }
    for (int y = 1; y < H; ++y) {
        if (is_not_null(seg[y][0])) {
            const auto& topCC = cc[y - 1][0];
            if (topCC == 0)
                assignNew(IntPoint(y, 0));
            else
                assignExisting(IntPoint(y, 0), topCC);
        }
    }

    // === Main loop: Process the rest of the image ===
    for (int y = 1; y < H; ++y) {
        for (int x = 1; x < W; ++x) {
            if (is_not_null(seg[y][x])) {
                IntPoint p(y, x);
                const auto& leftCC = cc[y][x - 1];
                const auto& topCC = cc[y - 1][x];
                if (leftCC == 0 && topCC == 0) {
                    assignNew(p);
                } else {
                    assignExisting(p, leftCC != 0 ? leftCC : topCC);
                    if (leftCC != 0 && topCC != 0 && leftCC != topCC)
                        componentAdjacency.emplace_back(IntPair{leftCC, topCC});
                }
            }
        }
    }

    // === Merge connected components based on adjacency information ===
    // --- Compute the graph of connected components ---
    std::vector<std::vector<int>> ccGraph(components.size());
    for (const auto& [a, b] : componentAdjacency) {
        ccGraph[a].push_back(b);
        ccGraph[b].push_back(a);
    }

    // --- Merge connected components using DFS ---
    std::vector<bool> visited(components.size(), false);
    std::function<void(int)> dfs = [&](int node) {
        visited[node] = true;
        outComponents.back().insert(outComponents.back().end(), components[node].begin(), components[node].end());
        for (int neighbor : ccGraph[node]) {
            if (!visited[neighbor]) dfs(neighbor);
        }
    };

    for (std::size_t i = 0; i < components.size(); ++i) {
        outComponents.emplace_back();
        if (!visited[i]) dfs(i);
    }
}

// -------------------------------------------------------------------------------------------------------------------
template <typename T>
torch::Tensor _skeletonize(const torch::Tensor& segMap, std::function<bool(const T&)> is_true,
                           std::function<bool(const T&, const T&)> is_same) {
    /*
    Skeletonize a binary segmentation map using the Zhang-Suen thinning algorithm.

    Parameters:
    - segMap: A binary tensor of shape (H, W) representing the segmentation map.

    Returns:
    - skeletonMap: A binary tensor of shape (H, W) representing the skeletonized map.
    */
    TORCH_CHECK_VALUE(segMap.dim() == 2, "segMap must be a 2D tensor");

    const int H = segMap.size(0), W = segMap.size(1);
    auto workSegMap = torch::zeros({H + 2, W + 2}, segMap.dtype());
    workSegMap.slice(0, 1, H + 1).slice(1, 1, W + 1) = segMap;
    auto seg_acc = workSegMap.accessor<T, 2>();

    auto skelMap = torch::zeros({H, W}, torch::dtype(torch::kBool));
    auto skel_acc = skelMap.accessor<bool, 2>();

    std::list<IntPoint> candidates;

    bool has_changed;
    do {
        has_changed = false;
        for (auto [dy, dx] : std::array<std::pair<int, int>, 4>{{{-1, 0}, {+1, 0}, {0, +1}, {0, -1}}}) {
            candidates.clear();
#pragma omp parallel for collapse(2) reduction(merge : candidates)
            for (int y = 1; y < H - 1; ++y) {
                for (int x = 1; x < W - 1; ++x) {
                    if (is_true(seg_acc[y][x]) && !skel_acc[y - 1][x - 1] &&
                        !is_same(seg_acc[y][x], seg_acc[y + dy][x + dx])) {
                        if (IS_SIMPLE_POINT_LOOKUP[get_neighborhood(seg_acc, y, x, is_same)])
                            candidates.emplace_back(y, x);
                        else
                            skel_acc[y - 1][x - 1] = true;
                    }
                }
            }

            for (const auto& p : candidates) {
                if (IS_SIMPLE_POINT_LOOKUP[get_neighborhood(seg_acc, p.y, p.x, is_same)]) {
                    seg_acc[p.y][p.x] = false;
                    has_changed = true;
                } else
                    skel_acc[p.y - 1][p.x - 1] = true;
            }
        }
    } while (has_changed);

    return skelMap;
}

torch::Tensor skeletonize(const torch::Tensor& segMap) {
    if (segMap.scalar_type() == torch::kUInt8) {
        return _skeletonize<uint8_t>(
            segMap, [](const uint8_t& v) { return v != 0; }, [](const uint8_t& a, const uint8_t& b) { return a == b; });
    } else if (segMap.scalar_type() == torch::kBool) {
        return _skeletonize<bool>(
            segMap, [](const bool& v) { return v; }, [](const bool& a, const bool& b) { return a == b; });
    }

    TORCH_CHECK_VALUE(false, "Unsupported tensor type for skeletonization. Supported types are uint8 and bool.");
}

enum class AVType : uint8_t { BKG = 0, ART = 1, VEI = 2, BOTH = 3, UNK = 4 };
bool is_av_same(const uint8_t& a, const uint8_t& b) {
    // Check if two values are the same, considering 1 and 2 as equivalent (artery and vein)
    const AVType& _a = static_cast<AVType>(a);
    const AVType& _b = static_cast<AVType>(b);
    switch (_a) {
        case AVType::BKG:
            return _b == AVType::BKG;
        case AVType::ART:
        case AVType::VEI:
            return _b == _a || _b >= AVType::BOTH;
        default:
            return _b != AVType::BKG;
    }
}

/**
 * @brief Skeletonize a binary segmentation map using the Zhang-Suen thinning algorithm.
 *
 * @param segMap: A binary tensor of shape (H, W) representing the segmentation map.
 *
 * @return: A binary tensor of shape (H, W) representing the skeletonized map.
 **/
torch::Tensor skeletonize_av(const torch::Tensor& segMap) {
    TORCH_CHECK_VALUE(segMap.scalar_type() == torch::kUInt8, "segMap must be a uint8 tensor");

    return _skeletonize<uint8_t>(segMap, [](const uint8_t& v) { return v != 0; }, &is_av_same);
}

/**
 * @brief Shrink the BOTH label in a binary segmentation map by dilating the ART and VEI labels around it,
 * yet keeping the overall topology.
 *
 * @param segMap: A binary tensor of shape (H, W) representing the segmentation map. This function modifies
 * the input tensor in place.
 */
void dilate_av_labels(torch::Tensor& segMap) {
    std::size_t H = segMap.size(0), W = segMap.size(1);
    auto seg = segMap.accessor<uint8_t, 2>();

    // === Compute connected components of the BOTH labels ===
    std::vector<std::vector<IntPoint>> both_components;
    connected_components(
        segMap, [](const uint8_t& v) { return v == static_cast<uint8_t>(AVType::BOTH); }, both_components);

// === For each CC dilate ART and VEI labels ===
#pragma omp parallel for
    for (const auto& component : both_components) {
        if (component.empty()) continue;
        if (component.size() == 1) {
            const IntPoint& p = component.front();
            if (!IS_SIMPLE_POINT_LOOKUP[get_neighborhood<uint8_t>(seg, p.y, p.x, &is_av_same)]) continue;
            // If the point is simple, assign it to the neighboring ART or VEI label
            for (const auto& neighbor : NEIGHBORHOOD) {
                IntPoint neighbor_p = p + neighbor;
                if (!neighbor_p.is_inside(H, W)) continue;
                uint8_t neighbor_label = seg[neighbor_p.y][neighbor_p.x];
                if (neighbor_label == static_cast<uint8_t>(AVType::ART) ||
                    neighbor_label == static_cast<uint8_t>(AVType::VEI)) {
                    seg[p.y][p.x] = neighbor_label;
                    break;
                }
            }
            continue;
        }

        // --- Compute barycenter ---
        Point barycenter(-0.5, -0.5);
        for (const auto& p : component) barycenter += p;
        barycenter /= component.size();
        IntPoint pBarycenter = barycenter.toInt();
        bool is_barycenter_inside = seg[pBarycenter.y][pBarycenter.x] == static_cast<uint8_t>(AVType::BOTH);

        // --- Identify boundaries and their label ---
        struct BoundaryPoint {
            IntPoint p;
            double angle;
            AVType label;
        };
        std::vector<BoundaryPoint> boundary_points;
        for (const auto& p : component) {
            int nearBkg = 0, nearArt = 0, nearVei = 0;
            for (const auto& neighbor : NEIGHBORHOOD) {
                IntPoint neighbor_p = p + neighbor;
                if (!neighbor_p.is_inside(H, W)) continue;
                switch (seg[neighbor_p.y][neighbor_p.x]) {
                    case static_cast<uint8_t>(AVType::BKG):
                        nearBkg++;
                        break;
                    case static_cast<uint8_t>(AVType::ART):
                        nearArt++;
                        break;
                    case static_cast<uint8_t>(AVType::VEI):
                        nearVei++;
                        break;
                }
            }
            AVType label = AVType::BOTH;
            // If the point is only adjacent to artery and vein and a minority of background pixels...
            if ((!nearArt || !nearVei) && nearBkg * 2 < (nearBkg + nearArt + nearVei))
                label = nearArt ? AVType::ART : AVType::VEI;  // assign the corresponding label
            boundary_points.emplace_back(BoundaryPoint{p, (p - barycenter).angle(), label});
        }

        // --- Find key boundary points where the label changes ---
        std::sort(boundary_points.begin(), boundary_points.end(),
                  [](const BoundaryPoint& a, const BoundaryPoint& b) { return a.angle < b.angle; });

        struct KeyPoint {
            std::size_t index;
            AVType label;
        };
        std::vector<KeyPoint> keypoints;
        // Find the first boundary point that is not BOTH to start the search for keypoints
        std::size_t B = boundary_points.size(), startOffset = 0;
        for (std::size_t i = 0; i < B; ++i) {
            if (boundary_points[i].label != AVType::BOTH) {
                startOffset = i;
                break;
            }
        }
        if (startOffset == B) {
            // If all points are BOTH, we arbitrarily assign the VEI label to all the connected components
            keypoints.emplace_back(KeyPoint{0, AVType::VEI});
        } else {
            // Otherwise find the midpoint of the segments of BOTH labels
            int lastStart = -1;
            for (std::size_t i = 0; i < boundary_points.size(); ++i) {
                const auto& current = boundary_points[(i + startOffset) % B];
                if (current.label == AVType::BOTH) {
                    if (lastStart == -1) lastStart = i;
                    continue;
                } else if (lastStart != -1) {
                    std::size_t mid = (lastStart + i) / 2;
                    keypoints.emplace_back(KeyPoint{(mid - startOffset) % B, current.label});
                    lastStart = -1;
                }
            }
            if (keypoints.size() > 1) {
                // startOffset prevent wrapping around issue, but may shift the keypoints, so we need to rotate them to
                // start from the minimum angle
                std::size_t minKeyIdx = 0;
                for (std::size_t i = 1; i < keypoints.size(); ++i) {
                    if (boundary_points[keypoints[i].index].angle < boundary_points[keypoints[minKeyIdx].index].angle)
                        minKeyIdx = i;
                }
                std::rotate(keypoints.begin(), keypoints.begin() + minKeyIdx, keypoints.end());
            }
        }

        // --- Recolor the connected component given the key points ---
        if (keypoints.empty()) continue;  // No key points found, skip this component
        if (keypoints.size() == 1) {
            // If there's only one key point, assign its label to the entire component
            AVType label = keypoints.front().label;
            for (const auto& p : component) seg[p.y][p.x] = static_cast<uint8_t>(label);
        } else {
            // Otherwise, assign labels based on the quadrants defined by the key points
            for (const auto& p : component) {
                double angle = (p - barycenter).angle();
                AVType label = keypoints.back().label;  // Default to the last keypoint's label
                for (auto keypoint = keypoints.rbegin(); keypoint != keypoints.rend(); ++keypoint) {
                    if (angle > boundary_points[keypoint->index].angle) {
                        label = keypoint->label;
                        break;  // Stop at the first keypoint with a smaller angle
                    }
                }
                seg[p.y][p.x] = static_cast<uint8_t>(label);
            }
            // Keep the barycenter as BOTH if there are more than 2 keypoints to maintain connectivity
            if (is_barycenter_inside && keypoints.size() > 2) {
                seg[pBarycenter.y][pBarycenter.x] = static_cast<uint8_t>(AVType::BOTH);
            }
        }
    }
}
