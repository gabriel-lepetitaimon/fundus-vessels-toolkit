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

// -------------------------------------------------------------------------------------------------------------------
torch::Tensor skeletonize(const torch::Tensor& segMap) {
    /*
    Skeletonize a binary segmentation map using the Zhang-Suen thinning algorithm.

    Parameters:
    - segMap: A binary tensor of shape (H, W) representing the segmentation map.

    Returns:
    - skeletonMap: A binary tensor of shape (H, W) representing the skeletonized map.
    */

    TORCH_CHECK_VALUE(segMap.dim() == 2, "segMap must be a 2D tensor");
    TORCH_CHECK_VALUE(segMap.scalar_type() == torch::kBool, "segMap must be a boolean tensor");

    const int H = segMap.size(0);
    const int W = segMap.size(1);
    auto workSegMap = segMap.clone();
    auto seg_acc = workSegMap.accessor<bool, 2>();
    auto skelMap = torch::zeros_like(segMap);
    auto skel_acc = skelMap.accessor<bool, 2>();
    std::vector<IntPoint> candidates;

    bool has_changed;
    do {
        has_changed = false;
        for (auto [dy, dx] : std::array<std::pair<int, int>, 4>{{{-1, 0}, {+1, 0}, {0, +1}, {0, -1}}}) {
            candidates.clear();
#pragma omp parallel for collapse(2) reduction(merge : candidates)
            for (int y = 1; y < H - 1; ++y) {
                for (int x = 1; x < W - 1; ++x) {
                    if (seg_acc[y][x] && !seg_acc[y + dy][x + dx] && !skel_acc[y][x]) {
                        if (IS_SIMPLE_POINT_LOOKUP[get_neighborhood(seg_acc, y, x)])
                            candidates.emplace_back(y, x);
                        else
                            skel_acc[y][x] = true;
                    }
                }
            }

            for (const auto& p : candidates) {
                if (IS_SIMPLE_POINT_LOOKUP[get_neighborhood(seg_acc, p.y, p.x)]) {
                    seg_acc[p.y][p.x] = false;
                    has_changed = true;
                } else {
                    skel_acc[p.y][p.x] = true;
                }
            }
        }
    } while (has_changed);

    return skelMap;
}

enum class AVType : uint8_t { BKG = 0, ART = 1, VEI = 2, BOTH = 3, UNK = 4 };
bool is_av_same(uint8_t a, uint8_t b) {
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
uint8_t get_neighborhood_av(const Tensor2DAcc<uint8_t>& z, int y, int x) {
    uint8_t neighbors = 0;
    const uint8_t& a = z[y][x];
    if (is_av_same(a, z[y - 1][x - 1])) neighbors |= 0b10000000;
    if (is_av_same(a, z[y - 1][x - 0])) neighbors |= 0b01000000;
    if (is_av_same(a, z[y - 1][x + 1])) neighbors |= 0b00100000;
    if (is_av_same(a, z[y - 0][x + 1])) neighbors |= 0b00010000;
    if (is_av_same(a, z[y + 1][x + 1])) neighbors |= 0b00001000;
    if (is_av_same(a, z[y + 1][x + 0])) neighbors |= 0b00000100;
    if (is_av_same(a, z[y + 1][x - 1])) neighbors |= 0b00000010;
    if (is_av_same(a, z[y + 0][x - 1])) neighbors |= 0b00000001;
    return neighbors;
}

torch::Tensor skeletonize_av(const torch::Tensor& segMap) {
    /*
    Skeletonize a binary segmentation map using the Zhang-Suen thinning algorithm.

    Parameters:
    - segMap: A binary tensor of shape (H, W) representing the segmentation map.

    Returns:
    - skeletonMap: A binary tensor of shape (H, W) representing the skeletonized map.
    */

    TORCH_CHECK_VALUE(segMap.dim() == 2, "segMap must be a 2D tensor");
    TORCH_CHECK_VALUE(segMap.scalar_type() == torch::kUInt8, "segMap must be a uint8 tensor");

    const int H = segMap.size(0);
    const int W = segMap.size(1);
    auto workSegMap = segMap.clone();
    auto seg_acc = workSegMap.accessor<uint8_t, 2>();
    auto skelMap = torch::zeros_like(segMap, torch::dtype(torch::kBool));
    auto skel_acc = skelMap.accessor<bool, 2>();
    std::vector<IntPoint> candidates;

    bool has_changed;
    do {
        has_changed = false;
        for (auto [dy, dx] : std::array<std::pair<int, int>, 4>{{{-1, 0}, {+1, 0}, {0, +1}, {0, -1}}}) {
            candidates.clear();
#pragma omp parallel for collapse(2) reduction(merge : candidates)
            for (int y = 1; y < H - 1; ++y) {
                for (int x = 1; x < W - 1; ++x) {
                    const uint8_t& seg = seg_acc[y][x];
                    if (seg != 0 && !skel_acc[y][x] && !is_av_same(seg, seg_acc[y + dy][x + dx])) {
                        if (IS_SIMPLE_POINT_LOOKUP[get_neighborhood_av(seg_acc, y, x)])
                            candidates.emplace_back(y, x);
                        else
                            skel_acc[y][x] = true;
                    }
                }
            }

            for (const auto& p : candidates) {
                if (IS_SIMPLE_POINT_LOOKUP[get_neighborhood_av(seg_acc, p.y, p.x)]) {
                    seg_acc[p.y][p.x] = 0;
                    has_changed = true;
                } else {
                    skel_acc[p.y][p.x] = true;
                }
            }
        }
    } while (has_changed);

    return skelMap;
}