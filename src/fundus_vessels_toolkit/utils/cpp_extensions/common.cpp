#include "common.h"

#include <cmath>

// === FROM_VECTOR ===
torch::Tensor vector_to_tensor(const std::vector<int>& vec) {
    torch::Tensor tensor = torch::zeros({(long)vec.size()}, torch::kInt32);
    auto accessor = tensor.accessor<int, 1>();
    for (int i = 0; i < (int)vec.size(); i++) accessor[i] = vec[i];
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<float>& vec) {
    torch::Tensor tensor = torch::zeros({(long)vec.size()}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 1>();
    for (int i = 0; i < (int)vec.size(); i++) accessor[i] = vec[i];
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<IntPair>& vec) {
    torch::Tensor tensor = torch::zeros({(long)vec.size(), 2}, torch::kInt32);
    auto accessor = tensor.accessor<int, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i][0];
        accessor[i][1] = vec[i][1];
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<FloatPair>& vec) {
    torch::Tensor tensor = torch::zeros({(long)vec.size(), 2}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i][0];
        accessor[i][1] = vec[i][1];
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<std::array<IntPair, 2>>& vec) {
    torch::Tensor tensor = torch::zeros({(long)vec.size(), 2, 2}, torch::kInt32);
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
    torch::Tensor tensor = torch::zeros({(long)vec.size(), 2}, torch::kDouble);
    auto accessor = tensor.accessor<double, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i].y;
        accessor[i][1] = vec[i].x;
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<IntPoint>& vec) {
    torch::Tensor tensor = torch::zeros({(long)vec.size(), 2}, torch::kInt32);
    auto accessor = tensor.accessor<int, 2>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0] = vec[i].y;
        accessor[i][1] = vec[i].x;
    }
    return tensor;
}

torch::Tensor vector_to_tensor(const std::vector<std::array<IntPoint, 2>>& vec) {
    torch::Tensor tensor = torch::zeros({(long)vec.size(), 2, 2}, torch::kInt32);
    auto accessor = tensor.accessor<int, 3>();
    for (int i = 0; i < (int)vec.size(); i++) {
        accessor[i][0][0] = vec[i][0].y;
        accessor[i][0][1] = vec[i][0].x;
        accessor[i][1][0] = vec[i][1].y;
        accessor[i][1][1] = vec[i][1].x;
    }
    return tensor;
}

CurveYX tensor_to_curveYX(const torch::Tensor& tensor) {
    auto accessor = tensor.accessor<int, 2>();
    CurveYX curveYX;
    curveYX.reserve(tensor.size(0));
    for (int i = 0; i < tensor.size(0); i++) curveYX.push_back({accessor[i][0], accessor[i][1]});
    return curveYX;
}

std::vector<IntPair> tensor_to_vectorIntPair(const torch::Tensor& tensor) {
    auto accessor = tensor.accessor<int, 2>();
    std::vector<IntPair> vec;
    vec.reserve(tensor.size(0));
    for (int i = 0; i < tensor.size(0); i++) vec.push_back({accessor[i][0], accessor[i][1]});
    return vec;
}

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
Point::Point(IntPair yx) : y(yx[0]), x(yx[1]) {}
Point::Point(FloatPair yx) : y(yx[0]), x(yx[1]) {}
Point::Point(IntPoint yx) : y(yx.y), x(yx.x) {}

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
double Point::squaredNorm() const { return y * y + x * x; }
double Point::norm() const { return sqrt(y * y + x * x); }
double Point::angle() const { return atan2(y, x); }
double Point::angle(const Point& p) const { return acos(dot(p) / (norm() * p.norm())); }

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
std::vector<float> movingAvg(const std::vector<float>& x, int size) {
    if (size < 0) size = x.size();

    std::vector<float> y;
    y.reserve(x.size());
    for (int i = 0; i < (int)x.size(); i++) {
        float sum = 0;
        int count = 0;
        for (int j = i - size; j <= i + size; j++) {
            if (j >= 0 && j < (int)x.size()) {
                sum += x[j];
                count++;
            }
        }
        y.push_back(sum / count);
    }
    return y;
}

// === Math functions ===
float distance(const Point& p1, const Point& p2) { return sqrt(pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2)); }
float distance(const IntPoint& p1, const IntPoint& p2) { return sqrt(pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2)); }
float distanceSqr(const Point& p1, const Point& p2) { return pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2); }
float distanceSqr(const IntPoint& p1, const IntPoint& p2) { return pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2); }

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

// === Graph ===
Edge::Edge(int start, int end, int id) : start(start), end(end), id(id) {}
Edge& Edge::operator=(const Edge& e) {
    this->start = e.start;
    this->end = e.end;
    this->id = e.id;
    return *this;
}

bool Edge::operator==(const Edge& e) const { return (start == e.start && end == e.end && id == e.id); }
bool Edge::operator!=(const Edge& e) const { return (start != e.start || end != e.end || id != e.id); }
bool Edge::operator<(const Edge& e) const {
    if (start != e.start) return start < e.start;
    if (end != e.end) return end < e.end;
    return id < e.id;
}

GraphAdjList edge_list_to_adjlist(const std::vector<IntPair>& edges, int N, bool directed) {
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

/*******************************************************************************************************************
 *             === NEIGHBORS ===
 *******************************************************************************************************************/
uint8_t count_neighbors(uint8_t neighborhood) {
    static const uint8_t NIBBLE_LOOKUP[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

    return NIBBLE_LOOKUP[neighborhood & 0x0F] + NIBBLE_LOOKUP[neighborhood >> 4];
}

uint8_t roll_neighbors(uint8_t neighborhood, uint8_t n) { return (neighborhood << n) | (neighborhood >> (8 - n)); }
