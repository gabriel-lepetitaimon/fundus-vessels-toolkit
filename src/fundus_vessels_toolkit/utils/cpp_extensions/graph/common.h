#ifndef COMMON_H
#define COMMON_H

#include <omp.h>
#include <torch/extension.h>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

/*******************************************************************************************************************
 *             === MATH ===
 *******************************************************************************************************************/
using IntPair = std::array<int, 2>;
using FloatPair = std::array<float, 2>;
using SizePair = std::array<std::size_t, 2>;

// === MATRIX ===
template <typename T, unsigned int D1 = 2, unsigned int D2 = D1>
using Matrix2 = std::array<std::array<T, D2>, D1>;

// === Point ===
struct Point;

struct IntPoint {
    int y;
    int x;

    IntPoint(int y = 0, int x = 0);
    IntPoint(IntPair yx);
    static IntPoint Invalid() { return IntPoint(INT_MIN, INT_MAX); }
    IntPoint& operator=(const IntPoint& p);

    IntPoint operator+(const IntPoint& p) const;
    Point operator+(const Point& p) const;
    IntPoint operator-(const IntPoint& p) const;
    Point operator-(const Point& p) const;
    IntPoint operator-() const;
    IntPoint operator*(int f) const;
    Point operator*(double f) const;
    Point operator/(double f) const;
    bool operator==(const IntPoint& p) const;
    bool operator!=(const IntPoint& p) const;

    bool is_inside(int H, int W) const;
    bool is_inside(const IntPoint& p) const;
    inline bool is_valid() const { return y != INT_MIN && x != INT_MAX; }
    bool is_null();

    IntPair toIntPair() const;

    friend std::ostream& operator<<(std::ostream& os, const IntPoint& p) {
        os << "(" << p.y << ", " << p.x << ")";
        return os;
    }
};

struct Point {
    double y;
    double x;

    // default + parameterized constructor
    Point(double y = 0, double x = 0);

    Point(IntPair yx);
    Point(FloatPair yx);
    Point(IntPoint yx);

    // assignment operator modifies object, therefore non-const
    Point& operator=(const Point& p);
    Point& operator+=(const Point& p);
    Point& operator-=(const Point& p);
    Point operator+(const Point& p) const;
    Point operator+(const IntPoint& p) const;
    Point operator-(const Point& p) const;
    Point operator-(const IntPoint& p) const;
    Point operator-() const;
    Point operator*(double f) const;
    Point operator/(double f) const;
    bool operator==(const Point& p) const;
    bool operator==(const IntPoint& p) const;
    bool operator!=(const Point& p) const;
    bool operator!=(const IntPoint& p) const;

    Point normalize() const;
    double dot(const Point& p) const;
    double squaredNorm() const;
    double norm() const;
    double angle() const;
    double angle(const Point& p) const;

    bool is_inside(double H, double W) const;
    bool is_inside(const Point& p) const;
    bool is_null() const;

    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(" << p.y << ", " << p.x << ")";
        return os;
    }

    IntPoint toInt() const;
    IntPair toIntPair() const;
    FloatPair toFloatPair() const;
};

#pragma omp declare reduction(merge : std::vector<Point> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction( \
        merge : std::vector<std::vector<Point>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp declare reduction( \
        merge : std::vector<IntPoint> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction( \
        merge : std::vector<std::vector<IntPoint>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

using CurveYX = std::vector<IntPoint>;
using Vector = Point;

struct PointWithID : IntPoint {
    int id;

    PointWithID(int y = 0, int x = 0, int id = 0);
    PointWithID(const IntPoint& p, int id = 0);
    PointWithID& operator=(const PointWithID& p);

    bool operator==(const PointWithID& p) const;
    bool operator!=(const PointWithID& p) const;
    bool isSamePoint(const IntPoint& p) const;

    IntPoint point() const;

    friend std::ostream& operator<<(std::ostream& os, const PointWithID& p) {
        os << "(" << p.y << ", " << p.x << ", id=" << p.id << ")";
        return os;
    }
};
#pragma omp declare reduction( \
        merge : std::vector<PointWithID> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

float distance(const Point& p1, const Point& p2);
float distance(const IntPoint& p1, const IntPoint& p2);
float distanceSqr(const Point& p1, const Point& p2);
float distanceSqr(const IntPoint& p1, const IntPoint& p2);

// === Gaussian ===
float gaussian(float x, float sigma = 1, float mu = 0);
std::vector<float> gaussianHalfKernel1D(float sigma = 1, int size = -1);
static const std::vector<float> GAUSSIAN_HALF_STD1 = gaussianHalfKernel1D(1, 4);
static const std::vector<float> GAUSSIAN_HALF_STD2 = gaussianHalfKernel1D(2, 7);
static const std::vector<float> GAUSSIAN_HALF_STD3 = gaussianHalfKernel1D(3, 10);

// === Average Filters ===
std::vector<float> movingAvg(const std::vector<float>& x, std::size_t size, const std::vector<int>& evaluateAtID);
std::vector<float> movingAvg(const std::vector<float>& x, const std::vector<float>& kernel);

template <typename T>
std::vector<T> medianFilter(const std::vector<T>& x, int halfSize = 1) {
    std::vector<T> y;
    y.reserve(x.size() - 2 * halfSize);

    for (int i = halfSize; i < (int)(x.size() - halfSize); i++) {
        std::map<T, int> count;
        for (int j = -halfSize; j <= halfSize; j++) count[x[i + j]]++;
        T median;
        for (auto [value, c] : count) {
            if (c > count[median]) median = value;
        }
        y.push_back(median);
    }
    return y;
}

// === Math utilities ===
inline int sign(int x) { return (x > 0) - (x < 0); }

std::vector<int> linspace_int(int start, int end, u_int n, bool endpoint = true);
inline int linspace_int(int i, int start, int end, int n, bool endpoint = true) {
    float step = (float)(end - start) / (endpoint ? (n - 1) : n);
    return (int)floor(start + i * step);
}
std::vector<double> linspace(double start, double end, u_int n, bool endpoint = true);
inline double linspace(int i, double start, double end, int n, bool endpoint = true) {
    double step = (end - start) / (endpoint ? (n - 1) : n);
    return start + i * step;
}

static const float SQRT2 = sqrt(2);

// === Graph ===

struct Edge {
    int start;
    int end;
    int id;

    Edge(int start = 0, int end = 0, int id = 0);
    Edge& operator=(const Edge& e);

    bool operator==(const Edge& e) const;
    bool operator!=(const Edge& e) const;
    bool operator<(const Edge& e) const;
};

using EdgeList = std::vector<Edge>;
using GraphAdjList = std::vector<std::set<Edge>>;
#pragma omp declare reduction(merge : std::vector<Edge> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

GraphAdjList edge_list_to_adjlist(const std::vector<IntPair>& edges, int N = -1, bool directed = false);
GraphAdjList edge_list_to_adjlist(const EdgeList& edges, int N = -1, bool directed = false);

/*******************************************************************************************************************
 *             === TORCH ===
 *******************************************************************************************************************/
template <typename T>
using Tensor2DAccessor = at::TensorAccessor<T, 2UL, at::DefaultPtrTraits, signed long>;

torch::Tensor vector_to_tensor(const std::vector<int>& vec);
torch::Tensor vector_to_tensor(const std::vector<float>& vec);
torch::Tensor vector_to_tensor(const std::vector<Point>& vec);
torch::Tensor vector_to_tensor(const std::vector<IntPoint>& vec);
torch::Tensor vector_to_tensor(const std::vector<IntPair>& vec);
torch::Tensor vector_to_tensor(const std::vector<FloatPair>& vec);
torch::Tensor vector_to_tensor(const std::vector<std::vector<IntPair>>& vec);
torch::Tensor vector_to_tensor(const std::vector<std::array<IntPoint, 2>>& vec);
torch::Tensor edge_list_to_tensor(const EdgeList& vec);

template <typename T>
std::vector<torch::Tensor> vectors_to_tensors(const std::vector<std::vector<T>>& vec) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(vec.size());
    for (const auto& v : vec) tensors.push_back(vector_to_tensor(v));
    return tensors;
}

CurveYX tensor_to_curve(const torch::Tensor& tensor);
std::vector<CurveYX> tensors_to_curves(const std::vector<torch::Tensor>& tensors);
std::vector<IntPair> tensor_to_vectorIntPair(const torch::Tensor& tensor);

/*******************************************************************************************************************
 *             === NEIGHBORS ===
 *******************************************************************************************************************/
/**
 * @brief Structure representing neighborhood coordinates: (y, x) and the id of the neighbor.
 *
 * The id of the neighbor is in the following order:
 *    0 1 2
 *    7 . 3
 *    6 5 4
 *
 */
static const std::array<PointWithID, 8> NEIGHBORHOOD = {
    {{-1, -1, 0}, {-1, 0, 1}, {-1, 1, 2}, {0, 1, 3}, {1, 1, 4}, {1, 0, 5}, {1, -1, 6}, {0, -1, 7}}};

/**
 * @brief Structure representing neighborhood coordinates: (y, x) and the id of the neighbor. The close neighbors
 * (horizontal and vertical) are listed first.
 *
 * The neighbors are listed in the following order:
 *    0 4 1
 *    7 . 5
 *    3 6 2
 *
 */
static const std::array<PointWithID, 8> CLOSE_NEIGHBORHOOD = {
    {{-1, -1, 0}, {-1, 1, 2}, {1, 1, 4}, {1, -1, 6}, {-1, 0, 1}, {0, 1, 3}, {1, 0, 5}, {0, -1, 7}}};

/**
 * @brief Array containing the possible next neighbors id given the previous neighbor id in a tracking process.
 *
 * When tracking a branch the neighbor pixels (x) adjacent to the previous pixel (p) are by construction not part of the
 * branch. The most likely next pixels are the ones in the opposite direction of the previous pixel (p). (The fourth and
 * fifth neighbors should theoretically never be the next pixel if the skeleton is valid.) x 4 1
 *  ->  p c 0   (c is the current pixel and p is the previous pixel,
 *      x 5 2    in this example the neighbor id of c relative to p is 3.)
 */
static const std::array<std::array<int, 5>, 8> TRACK_NEXT_NEIGHBORS = {{
    {0, 1, 7, 2, 6},  // neighbor index of c relative to p is 0 (previous pixel is bottom-right)
    {1, 2, 0, 3, 7},  // neighbor index of c relative to p is 1 (previous pixel is bottom)
    {2, 3, 1, 4, 0},  // neighbor index of c relative to p is 2 (previous pixel is bottom-left)
    {3, 4, 2, 5, 1},  // neighbor index of c relative to p is 3 (previous pixel is left)
    {4, 5, 3, 6, 2},  // neighbor index of c relative to p is 4 (previous pixel is top-left)
    {5, 6, 4, 7, 3},  // neighbor index of c relative to p is 5 (previous pixel is top)
    {6, 7, 5, 0, 4},  // neighbor index of c relative to p is 6 (previous pixel is top-right)
    {7, 0, 6, 1, 5}   // neighbor index of c relative to p is 7 (previous pixel is right)
}};

/**
 * @brief Array containing the possible next neighbors id given the previous neighbor id in a tracking process.
 *
 * When tracking a branch the neighbor pixels (x) adjacent to the previous pixel (p) are by construction not part of the
 * branch. The closest pixels (vertical and horizontal) are listed first (The fourth and fifth neighbors should
 * theoretically never be the next pixel if the skeleton is valid.) x 1 3
 *  ->  p c 0   (c is the current pixel and p is the previous pixel,
 *      x 2 4    in this example the neighbor id of c relative to p is 3.)
 *
 *      p x 4
 *  ->  x c 1   (c is the current pixel and p is the previous pixel,
 *      5 2 3    in this example the neighbor id of c relative to p is 4.)
 */
static const std::array<std::array<int, 5>, 8> TRACK_CLOSE_NEIGHBORS = {{
    {7, 1, 0, 6, 2},  // neighbor index of c relative to p is 0 (previous pixel is bottom-right)
    {1, 7, 3, 0, 2},  // neighbor index of c relative to p is 1 (previous pixel is bottom)
    {1, 3, 2, 0, 4},  // neighbor index of c relative to p is 2 (previous pixel is bottom-left)
    {3, 1, 5, 2, 4},  // neighbor index of c relative to p is 3 (previous pixel is left)
    {3, 5, 4, 2, 6},  // neighbor index of c relative to p is 4 (previous pixel is top-left)
    {5, 3, 7, 4, 6},  // neighbor index of c relative to p is 5 (previous pixel is top)
    {5, 7, 6, 4, 0},  // neighbor index of c relative to p is 6 (previous pixel is top-right)
    {7, 5, 1, 6, 0}   // neighbor index of c relative to p is 7 (previous pixel is right)
}};

inline int opposite_neighbor_id(int n) { return n < 4 ? n + 4 : n - 4; }

inline int next_neighbor_id(int n) { return n < 7 ? n + 1 : 0; }

inline int prev_neighbor_id(int n) { return n > 0 ? n - 1 : 7; }

inline bool hit(uint8_t neighborhood, uint8_t positive_mask) { return (neighborhood & positive_mask) == positive_mask; }

inline bool miss(uint8_t neighborhood, uint8_t negative_mask) { return (neighborhood & negative_mask) == 0; }

inline bool hit_and_miss(uint8_t neighborhood, uint8_t positive_mask, uint8_t negative_mask) {
    return (neighborhood & positive_mask) == positive_mask && (neighborhood & negative_mask) == 0;
}

uint8_t count_neighbors(uint8_t neighborhood);

uint8_t roll_neighbors(uint8_t neighborhood, uint8_t n);

/**
 * @brief Read the value of the 8 neighbors of a pixel in a binary image.
 *
 * The neighbors are read in the following order:
 *  1 2 3    0x01 0x02 0x04
 *  8 . 4 -> 0x80  .   0x08
 *  7 6 5    0x40 0x20 0x10
 *
 * Assume that z contains a pixel at the position (y, x).
 * The neighbors outside the image are considered to be 0.
 *
 * @param z Binary image.
 * @param y Row of the pixel.
 * @param x Column of the pixel.
 * @return unsigned short Value of the 8 neighbors of the pixel. The top-left neighbor is the most significant bit.
 */
template <typename T>
uint8_t get_neighborhood_safe(const Tensor2DAccessor<T>& z, int y, int x) {
    uint8_t neighbors = 0;
    if (y > 0) {
        if (x > 0 && z[y - 1][x - 1] > 0) neighbors |= 0b10000000;
        if (z[y - 1][x] > 0) neighbors |= 0b01000000;
        if (x < z.size(1) - 1 && z[y - 1][x + 1] > 0) neighbors |= 0b00100000;
    }

    if (x < z.size(1) - 1 && z[y][x + 1] > 0) neighbors |= 0b00010000;

    if (y < z.size(0) - 1) {
        if (x < z.size(1) - 1 && z[y + 1][x + 1] > 0) neighbors |= 0b00010000;
        if (z[y + 1][x] > 0) neighbors |= 0b00001000;
        if (x > 0 && z[y + 1][x - 1] > 0) neighbors |= 0b00000010;
    }

    if (x > 0 && z[y][x - 1] > 0) neighbors |= 0b00000001;
    return neighbors;
}

/**
 * @brief Read the value of the 8 neighbors of a pixel in a binary image.
 *
 * The neighbors are read in the following order:
 *  1 2 3    0x01 0x02 0x04
 *  8 . 4 -> 0x80  .   0x08
 *  7 6 5    0x40 0x20 0x10
 *
 * Assume that z contains a pixel at the position (y, x) and that the pixel is not at the border of the image.
 *
 * @param z Binary image.
 * @param y Row of the pixel.
 * @param x Column of the pixel.
 * @return unsigned short Value of the 8 neighbors of the pixel. The top-left neighbor is the most significant bit.
 */
template <typename T>
uint8_t get_neighborhood(const Tensor2DAccessor<T>& z, int y, int x) {
    uint8_t neighbors = 0;
    neighbors |= z[y - 1][x - 1] > 0 ? 0b10000000 : 0;
    neighbors |= z[y - 1][x] > 0 ? 0b01000000 : 0;
    neighbors |= z[y - 1][x + 1] > 0 ? 0b00100000 : 0;
    neighbors |= z[y][x + 1] > 0 ? 0b00010000 : 0;
    neighbors |= z[y + 1][x + 1] > 0 ? 0b00001000 : 0;
    neighbors |= z[y + 1][x] > 0 ? 0b00000100 : 0;
    neighbors |= z[y + 1][x - 1] > 0 ? 0b00000010 : 0;
    neighbors |= z[y][x - 1] > 0 ? 0b00000001 : 0;
    return neighbors;
}

#endif  // COMMON_H