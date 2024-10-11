#include "common.h"

#include <cmath>

// === IntPoint ===
IntPoint::IntPoint(int y, int x) : y(y), x(x) {}
IntPoint::IntPoint(IntPair yx) : y(yx[0]), x(yx[1]) {}

IntPoint& IntPoint::operator=(const IntPoint& p) {
    this->x = p.x;
    this->y = p.y;
    return *this;
}

IntPoint IntPoint::operator+(const IntPoint& p) const { return IntPoint(p.y + y, p.x + x); }
Point IntPoint::operator+(const Point& p) const { return Point(y + p.y, x + p.x); }
IntPoint IntPoint::operator-(const IntPoint& p) const { return IntPoint(y - p.y, x - p.x); }
Point IntPoint::operator-(const Point& p) const { return Point(y - p.y, x - p.x); }
IntPoint IntPoint::operator-() const { return IntPoint(-y, -x); }
IntPoint IntPoint::operator*(int f) const { return IntPoint(y * f, x * f); }
Point IntPoint::operator*(double f) const { return Point(y * f, x * f); }
Point IntPoint::operator/(double f) const { return Point(y / f, x / f); }

bool IntPoint::operator==(const IntPoint& p) const { return (x == p.x && y == p.y); }
bool IntPoint::operator!=(const IntPoint& p) const { return (x != p.x || y != p.y); }
bool IntPoint::is_inside(int h, int w) const { return (x >= 0 && x < w && y >= 0 && y < h); }
bool IntPoint::is_inside(const IntPoint& p) const { return (x >= 0 && x < p.x && y >= 0 && y < p.y); }

IntPair IntPoint::toIntPair() const { return {y, x}; }

// === Point ===
Point::Point(double y, double x) : y(y), x(x) {}
Point::Point(const IntPair& yx) : y(yx[0]), x(yx[1]) {}
Point::Point(const FloatPair& yx) : y(yx[0]), x(yx[1]) {}
Point::Point(const IntPoint& yx) : y(yx.y), x(yx.x) {}
Point::Point(const at::TensorAccessor<float, 1UL, at::DefaultPtrTraits, signed long>& yx) : y(yx[0]), x(yx[1]) {}

// assignment operator modifies object, therefore non-const
Point& Point::operator=(const Point& p) {
    this->x = p.x;
    this->y = p.y;
    return *this;
}
Point& Point::operator+=(const Point& p) {
    this->x += p.x;
    this->y += p.y;
    return *this;
}
Point& Point::operator-=(const Point& p) {
    this->x -= p.x;
    this->y -= p.y;
    return *this;
}

Point Point::operator+(const Point& p) const { return Point(p.y + y, p.x + x); }
Point Point::operator+(const IntPoint& p) const { return Point(p.y + y, p.x + x); }
Point Point::operator-(const Point& p) const { return Point(y - p.y, x - p.x); }
Point Point::operator-(const IntPoint& p) const { return Point(y - p.y, x - p.x); }
Point Point::operator-() const { return Point(-y, -x); }
Point Point::operator*(double s) const { return Point(y * s, x * s); }
Point Point::operator/(double s) const { return Point(y / s, x / s); }

bool Point::operator==(const Point& p) const { return (x == p.x && y == p.y); }
bool Point::operator==(const IntPoint& p) const { return (x == p.x && y == p.y); }
bool Point::operator!=(const Point& p) const { return (x != p.x || y != p.y); }
bool Point::operator!=(const IntPoint& p) const { return (x != p.x || y != p.y); }

Point Point::normalize() const {
    double n = norm();
    if (n != 0) return Point(y / n, x / n);
    return Point(0, 0);
}

double Point::dot(const Point& p) const { return y * p.y + x * p.x; }
double Point::cosSim(const Point& p) const { return dot(p) / (norm() * p.norm()); }
double Point::cross(const Point& p) const { return y * p.x - x * p.y; }
double Point::squaredNorm() const { return y * y + x * x; }
Point Point::positiveCoordinates() const { return Point(std::abs(y), std::abs(x)); }
Point Point::rot90() const { return Point(-x, y); }
Point Point::rot270() const { return Point(x, -y); }
double Point::norm() const { return sqrt(y * y + x * x); }
double Point::angle() const { return atan2(y, x); }
double Point::angle(const Point& p) const { return acos(dot(p) / (norm() * p.norm())); }
/// @brief Rotate the point by an angle in radians. Positive rotation is clockwise.
/// @param angle
/// @return The rotated point.
Point Point::rotate(double angle) const { return rotate(Point(sin(angle), cos(angle))); }
/// @brief Rotate the point by the angle from the positive x-axis to the u vector.
/// @param u A unitary vector.
/// @return The rotated point.
Point Point::rotate(const Point& u) const { return Point(y * u.x + x * u.y, x * u.x - y * u.y); }
/// @brief Rotate the point by the angle from the u vector to the positive x-axis (effectively using u as the new
/// base vector for the x-axis).
/// @param u A unitary vector.
/// @return The rotated point.
Point Point::rotate_neg(const Point& u) const { return Point(y * u.x - x * u.y, x * u.x + y * u.y); }

bool Point::is_inside(double h, double w) const { return (x >= 0 && x < w && y >= 0 && y < h); }
bool Point::is_inside(const Point& p) const { return (x >= 0 && x < p.x && y >= 0 && y < p.y); }
bool Point::is_null() const { return (x == 0 && y == 0); }

IntPoint Point::toInt() const { return IntPoint((int)round(y), (int)round(x)); }
IntPair Point::toIntPair() const { return {(int)round(y), (int)round(x)}; }
FloatPair Point::toFloatPair() const { return {(float)y, (float)x}; }

// === PointWithID ===
PointWithID::PointWithID(int y, int x, int id) : IntPoint(y, x), id(id) {}

PointWithID::PointWithID(const IntPoint& p, int id) : IntPoint(p), id(id) {}

PointWithID& PointWithID::operator=(const PointWithID& p) {
    x = p.x;
    y = p.y;
    id = p.id;
    return *this;
}

bool PointWithID::operator==(const PointWithID& p) const { return (x == p.x && y == p.y && id == p.id); }

bool PointWithID::operator!=(const PointWithID& p) const { return (x != p.x || y != p.y || id != p.id); }

bool PointWithID::isSamePoint(const IntPoint& p) const { return (x == p.x && y == p.y); }

IntPoint PointWithID::point() const { return IntPoint(y, x); }

// === Gaussian ===
float gaussian(float x, float sigma, float mu) { return exp(-pow(x - mu, 2) / (2 * pow(sigma, 2))); }

std::vector<float> gaussianHalfKernel1D(float sigma, int size) {
    if (size < 0) {
        if (sigma == floor(sigma)) {
            switch ((int)sigma) {
                case 1:
                    return GAUSSIAN_HALF_STD1;
                case 2:
                    return GAUSSIAN_HALF_STD2;
                case 3:
                    return GAUSSIAN_HALF_STD3;
            }
        }
        size = ceil(3 * sigma) + 1;
    }

    std::vector<float> kernel;
    kernel.reserve(size);
    for (int i = 0; i < size; i++) kernel.push_back(gaussian(i, sigma));
    return kernel;
}

std::vector<float> movingAvg(const std::vector<float>& x, const std::vector<float>& halfKernel) {
    std::vector<float> xSmooth;
    xSmooth.reserve(x.size());
    const std::size_t xSize = x.size();
    const std::size_t K = halfKernel.size();

    for (std::size_t i = 0; i < x.size(); i++) {
        float v = 0;
        float sumWeights = 0;
        for (std::size_t j = 1; j < K && i >= j; j++) {
            const float w = halfKernel[j];
            v += w * x[i - j];
            sumWeights += w;
        }
        for (std::size_t j = 0; j < K && j < xSize - i; j++) {
            std::size_t idx = i + j;
            const float w = halfKernel[j];
            v += w * x[idx];
            sumWeights += w;
        }
        if (sumWeights != 0) {
            xSmooth.push_back(v / sumWeights);
        } else {
            xSmooth.push_back(x[i]);
        }
    }
    return xSmooth;
}

// === Moving Average ===
std::vector<float> movingAvg(const std::vector<float>& x, std::size_t size, const std::vector<int>& evaluateAtID) {
    const std::size_t xSize = x.size();
    bool evaluateAll = evaluateAtID.size() == 0;
    const std::size_t ySize = evaluateAll ? xSize : evaluateAtID.size();

    std::vector<float> y(ySize);

    for (std::size_t yI = 0; yI < ySize; yI++) {
        const std::size_t i = evaluateAll ? yI : evaluateAtID[yI];
        float sum = 0;
        int count = 0;
        for (std::size_t j = (i >= size ? i - size : 0); j <= (i + size < xSize ? i + size : xSize - 1); j++) {
            sum += x[j];
            count++;
        }
        y[yI] = sum / count;
    }
    return y;
}

// === Math functions ===
float distance(const Point& p1, const Point& p2) { return sqrt(pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2)); }
float distance(const IntPoint& p1, const IntPoint& p2) { return sqrt(pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2)); }
float distanceSqr(const Point& p1, const Point& p2) { return pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2); }
float distanceSqr(const IntPoint& p1, const IntPoint& p2) { return pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2); }

std::vector<std::size_t> arange(const std::size_t& start, const std::size_t& end, const std::size_t& step) {
    std::vector<std::size_t> vec;
    vec.reserve((end - start) / step);
    for (std::size_t i = start; i < end; i += step) vec.push_back(i);
    return vec;
}

std::vector<int> linspace_int(int start, int end, u_int n, bool endpoint) {
    if (n == 1) return {start};

    std::vector<int> vec;
    vec.reserve(n);

    float step = (float)(end - start) / (endpoint ? (n - 1) : n);
    for (u_int i = 0; i < n; i++) vec.push_back((int)floor(start + i * step));
    return vec;
}
std::vector<double> linspace(double start, double end, u_int n, bool endpoint) {
    if (n == 1) return {start};

    std::vector<double> vec;
    vec.reserve(n);

    double step = (end - start) / (endpoint ? (n - 1) : n);
    for (u_int i = 0; i < n; i++) vec.push_back(start + i * step);
    return vec;
}

std::vector<int> quantize_triband(const std::vector<float>& x, float low, float high, std::size_t medianHalfSize) {
    std::vector<int> y;
    y.reserve(x.size());
    for (const auto& v : x) {
        if (v < low)
            y.push_back(-1);
        else if (v > high)
            y.push_back(1);
        else
            y.push_back(0);
    }
    if (medianHalfSize == 0 or y.size() < medianHalfSize * 2) return y;
    return medianFilter(y, medianHalfSize);
}

/*******************************************************************************************************************
 *             === TORCH ===
 *******************************************************************************************************************/
// === FROM_VECTOR ===
torch::Tensor vector_to_tensor(const std::vector<int>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size()}, torch::kInt32);
    auto accessor = tensor.accessor<int, 1>();
    for (int i = 0; i < (int)vec.size(); i++) accessor[i] = vec[i];
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<float>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size()}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 1>();
    for (int i = 0; i < (int)vec.size(); i++) accessor[i] = vec[i];
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<double>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size()}, torch::kFloat64);
    auto accessor = tensor.accessor<double, 1>();
    for (int i = 0; i < (int)vec.size(); i++) accessor[i] = vec[i];
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<std::size_t>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size()}, torch::kInt64);
    auto accessor = tensor.accessor<int64_t, 1>();
    for (int i = 0; i < (int)vec.size(); i++) accessor[i] = vec[i];
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<IntPair>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size(), 2}, torch::kInt32);
    auto accessor = tensor.accessor<int, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i][0];
        accessor[i][1] = vec[i][1];
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<UIntPair>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size(), 2}, torch::kInt32);
    auto accessor = tensor.accessor<int, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i][0];
        accessor[i][1] = vec[i][1];
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<FloatPair>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size(), 2}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i][0];
        accessor[i][1] = vec[i][1];
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<std::array<IntPair, 2>>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size(), 2, 2}, torch::kInt32);
    auto accessor = tensor.accessor<int, 3>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0][0] = vec[i][0][0];
        accessor[i][0][1] = vec[i][0][1];
        accessor[i][1][0] = vec[i][1][0];
        accessor[i][1][1] = vec[i][1][1];
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<Point>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size(), 2}, torch::kDouble);
    auto accessor = tensor.accessor<double, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i].y;
        accessor[i][1] = vec[i].x;
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<IntPoint>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size(), 2}, torch::kInt32);
    auto accessor = tensor.accessor<int, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i].y;
        accessor[i][1] = vec[i].x;
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<std::array<IntPoint, 2>>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size(), 2, 2}, torch::kInt32);
    auto accessor = tensor.accessor<int, 3>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0][0] = vec[i][0].y;
        accessor[i][0][1] = vec[i][0].x;
        accessor[i][1][0] = vec[i][1].y;
        accessor[i][1][1] = vec[i][1].x;
    }
    return tensor;
}

torch::Tensor remove_rows(const torch::Tensor& tensor, std::vector<int> rows) {
    if (rows.empty()) return tensor;
    std::sort(rows.begin(), rows.end());
    std::vector<long> shape(tensor.dim());
    shape[0] = tensor.size(0) - rows.size();
    rows.push_back(tensor.size(0));
    for (int i = 1; i < tensor.dim(); i++) shape[i] = tensor.size(i);

    torch::Tensor new_tensor = torch::empty(shape, tensor.options());
    auto const iniSlice = torch::indexing::Slice(0, rows[0], 1);
    new_tensor.index({iniSlice}) = tensor.index({iniSlice});
    for (std::size_t i = 1; i < rows.size(); i++) {
        new_tensor.index_put_({torch::indexing::Slice(rows[i - 1] + 1 - i, rows[i] - i, 1)},
                              tensor.index({torch::indexing::Slice(rows[i - 1] + 1, rows[i], 1)}));
    }
    return new_tensor;
}

CurveYX tensor_to_curve(const torch::Tensor& tensor) {
    auto accessor = tensor.accessor<int, 2>();
    CurveYX curveYX;
    curveYX.reserve(tensor.size(0));
    for (int i = 0; i < tensor.size(0); i++) curveYX.push_back({accessor[i][0], accessor[i][1]});
    return curveYX;
}

std::vector<CurveYX> tensors_to_curves(const std::vector<torch::Tensor>& tensors) {
    std::vector<CurveYX> curves;
    curves.reserve(tensors.size());
    for (const auto& tensor : tensors) curves.push_back(tensor_to_curve(tensor));
    return curves;
}

std::vector<IntPair> tensor_to_vectorIntPair(const torch::Tensor& tensor) {
    auto accessor = tensor.accessor<int, 2>();
    std::vector<IntPair> vec;
    vec.reserve(tensor.size(0));
    for (std::size_t i = 0; i < (std::size_t)tensor.size(0); i++) vec.push_back({accessor[i][0], accessor[i][1]});
    return vec;
}

PointList tensor_to_pointList(const torch::Tensor& tensor) {
    auto accessor = tensor.accessor<float, 2>();
    PointList curve;
    curve.reserve(tensor.size(0));
    for (std::size_t i = 0; i < (std::size_t)tensor.size(0); i++) curve.push_back({accessor[i][0], accessor[i][1]});
    return curve;
}

Scalars tensor_to_scalars(const torch::Tensor& tensor) {
    auto accessor = tensor.accessor<float, 1>();
    Scalars vec;
    vec.reserve(tensor.size(0));
    for (std::size_t i = 0; i < (std::size_t)tensor.size(0); i++) vec.push_back(accessor[i]);
    return vec;
}

/*******************************************************************************************************************
 *             === GRAPH ===
 *******************************************************************************************************************/
Edge::Edge(int start, int end, int id) : start(start), end(end), id(id) {}
Edge& Edge::operator=(const Edge& e) {
    this->start = e.start;
    this->end = e.end;
    this->id = e.id;
    return *this;
}

int Edge::other(int node) const { return (node == start) ? end : start; }

bool Edge::operator==(const Edge& e) const { return (start == e.start && end == e.end && id == e.id); }
bool Edge::operator!=(const Edge& e) const { return (start != e.start || end != e.end || id != e.id); }
bool Edge::operator<(const Edge& e) const { return id < e.id; }

GraphAdjList edge_list_to_adjlist(const std::vector<IntPair>& edges, int N, bool directed) {
    if (N < 0) {
        N = 0;
        for (const IntPair& e : edges) N = std::max(N, std::max(e[0], e[1]));
        N++;
    }

    GraphAdjList graph(N);
    int i = 0;
    for (const IntPair& e : edges) {
        const Edge edge(e[0], e[1], i);
        graph[edge.start].insert(edge);
        if (!directed) graph[edge.end].insert(edge);
        i++;
    }
    return graph;
}

GraphAdjList edge_list_to_adjlist(const EdgeList& edges, int N, bool directed) {
    if (N < 0) {
        N = 0;
        for (const Edge& e : edges) N = std::max(N, std::max(e.start, e.end));
        N++;
    }

    GraphAdjList graph(N);
    for (const Edge& e : edges) {
        graph[e.start].insert(e);
        if (!directed) graph[e.end].insert(e);
    }
    return graph;
}

GraphAdjList edge_list_to_adjlist(const Tensor2DAcc<int>& edges, int N, bool directed) {
    const std::size_t E = (std::size_t)edges.size(0);
    if (N < 0) {
        N = 0;
        for (std::size_t i = 0; i < E; i++) N = std::max(N, std::max(edges[i][0], edges[i][1]));
        N++;
    }

    GraphAdjList graph(N);
    int i = 0;
    for (std::size_t j = 0; j < E; j++) {
        const Edge edge(edges[j][0], edges[j][1], i);
        graph[edge.start].insert(edge);
        if (!directed) graph[edge.end].insert(edge);
        i++;
    }
    return graph;
}

torch::Tensor edge_list_to_tensor(const EdgeList& edge_list) {
    torch::Tensor branches_list_tensor = torch::empty({(int)edge_list.size(), 2}, torch::kInt32);
    auto branches_list_acc = branches_list_tensor.accessor<int32_t, 2>();
    for (const auto& v : edge_list) {
        branches_list_acc[v.id][0] = v.start;
        branches_list_acc[v.id][1] = v.end;
    }
    return branches_list_tensor;
}

/*******************************************************************************************************************
 *             === NEIGHBORS ===
 *******************************************************************************************************************/
uint8_t count_neighbors(uint8_t neighborhood) {
    static const uint8_t NIBBLE_LOOKUP[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

    return NIBBLE_LOOKUP[neighborhood & 0x0F] + NIBBLE_LOOKUP[neighborhood >> 4];
}

uint8_t roll_neighbors(uint8_t neighborhood, uint8_t n) { return (neighborhood << n) | (neighborhood >> (8 - n)); }
