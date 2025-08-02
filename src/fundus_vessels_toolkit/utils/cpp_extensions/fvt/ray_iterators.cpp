#include "ray_iterators.h"

std::array<IntPoint, 2> track_nearest_edges(const IntPoint& start, const Point& direction,
                                            const Tensor2DAcc<bool>& segmentation, int max_iter) {
    if (direction.is_null()) return {{IntPoint::Invalid(), IntPoint::Invalid()}};
    const int H = segmentation.size(0), W = segmentation.size(1);

    RayIterator dIter, rIter;  // Direct iterator, reverse iterator
    bool dSafe = true, rSafe = true;
    if (direction.x > 0) {
        if (direction.y > 0) {                // Direct:  West
            if (direction.x > direction.y) {  // SWW
                float delta = direction.y / direction.x;
                dIter = RayIterator(start, delta, Octant::SEE, &Incrementors::SEE);
                rIter = RayIterator(start, delta, Octant::NWW, &Incrementors::NWW);
            } else {  // SSW
                float delta = direction.x / direction.y;
                dIter = RayIterator(start, delta, Octant::SSE, &Incrementors::SSE);
                rIter = RayIterator(start, delta, Octant::NNW, &Incrementors::NNW);
            }
            dSafe = start.y + max_iter < H && start.x + max_iter < W;
            rSafe = start.y >= max_iter && start.x >= max_iter;
        } else {                               // Direct:  East
            if (direction.x > -direction.y) {  // SEE
                float delta = -direction.y / direction.x;
                dIter = RayIterator(start, delta, Octant::NEE, &Incrementors::NEE);
                rIter = RayIterator(start, delta, Octant::SWW, &Incrementors::SWW);
            } else {  // SSE
                float delta = -direction.x / direction.y;
                dIter = RayIterator(start, delta, Octant::NNE, &Incrementors::NNE);
                rIter = RayIterator(start, delta, Octant::SSW, &Incrementors::SSW);
            }
            dSafe = start.y + max_iter < H && start.x >= max_iter;
            rSafe = start.y >= max_iter && start.x + max_iter < W;
        }
    } else {                                   // Direct: North
        if (direction.y > 0) {                 // Direct:  West
            if (-direction.x > direction.y) {  // NWW
                float delta = -direction.y / direction.x;
                dIter = RayIterator(start, delta, Octant::SWW, &Incrementors::SWW);
                rIter = RayIterator(start, delta, Octant::NEE, &Incrementors::NEE);
            } else {  // NNW
                float delta = -direction.x / direction.y;
                dIter = RayIterator(start, delta, Octant::SSW, &Incrementors::SSW);
                rIter = RayIterator(start, delta, Octant::NNE, &Incrementors::NNE);
            }
            dSafe = start.y >= max_iter && start.x + max_iter < W;
            rSafe = start.y + max_iter < H && start.x >= max_iter;
        } else {                              // Direct:  East
            if (direction.x < direction.y) {  // NEE
                float delta = direction.y / direction.x;
                dIter = RayIterator(start, delta, Octant::NWW, &Incrementors::NWW);
                rIter = RayIterator(start, delta, Octant::SEE, &Incrementors::SEE);
            } else {  // NNE
                float delta = direction.x / direction.y;
                dIter = RayIterator(start, delta, Octant::NNW, &Incrementors::NNW);
                rIter = RayIterator(start, delta, Octant::SSE, &Incrementors::SSE);
            }
            dSafe = start.y >= max_iter && start.x >= max_iter;
            rSafe = start.y + max_iter < H && start.x + max_iter < W;
        }
    }

    IntPoint dBound = IntPoint::Invalid(), rBound = IntPoint::Invalid();

    int i = 0;
    IntPoint lastP = start;
    do {
        const IntPoint& p = ++dIter;
        if (!segmentation[p.y][p.x]) {
            dBound = lastP;
            break;
        } else if (!dSafe && !p.is_inside(H, W))
            break;
        lastP = p;
    } while (++i != max_iter);

    i = 0;
    lastP = start;
    do {
        const IntPoint& p = ++rIter;
        if (!segmentation[p.y][p.x]) {
            rBound = lastP;
            break;
        } else if (!rSafe && !p.is_inside(H, W))
            break;
        lastP = p;
    } while (++i != max_iter);

    return {dBound, rBound};
}

std::pair<int, IntPoint> track_nearest_branch(const IntPoint& start, const Point& direction, float angle,
                                              float max_distance, const Tensor2DAcc<int>& branchMap) {
    if (direction.is_null()) return {0, IntPoint::Invalid()};
    const int H = branchMap.size(0), W = branchMap.size(1);

    ConeIterator cIter(start, direction, angle);
    float leftRayDelta = cIter.leftRay().delta();
    int max_it = max_distance / sqrt(1 + leftRayDelta * leftRayDelta) + 1;

    float bestDist = max_distance + 1e-3;
    std::pair<int, IntPoint> best = {0, IntPoint::Invalid()};

    while (!cIter.iter() || cIter.height() < max_it) {
        const IntPoint& p = *cIter;
        if (p.is_inside(H, W)) {
            const int& branchID = branchMap[p.y][p.x];
            if (branchID > 0) {
                float dist = distance(start, p);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = {branchID, p};
                    // Reduce the maximum number of iterations to when the central axis of
                    // the cone will reach the current best distance.
                    IntPoint maxCentralP = (direction * bestDist).ceil();
                    max_it = std::max(abs(maxCentralP.x), abs(maxCentralP.y));
                }
            }
        }
    }
    return best;
}

void draw_line(IntPoint start, IntPoint end, Tensor2DAcc<int>& tensor, int value, int H, int W) {
    if (start == end) {
        if (start.is_inside(H, W)) tensor[start.y][start.x] = value;
        return;
    }
    bool safe = start.is_inside(1, 1, H - 1, W - 1) && end.is_inside(1, 1, H - 1, W - 1);
    RayIterator ray(start, end - start);
    int i = ray.stepTo(end);
    while (i-- > 0) {
        const IntPoint& p = ++ray;
        if (safe || p.is_inside(H, W)) tensor[p.y][p.x] = value;
    }
}

void draw_line(IntPoint start, IntPoint end, Tensor2DAcc<int>& tensor, int value) {
    const auto& size = tensor.sizes();
    draw_line(start, end, tensor, value, size[0], size[1]);
}

/**********************************************************************************************************************
 *            === INCREMENTORS ===
 **********************************************************************************************************************/

Incrementor Incrementors::SEE = {[](IntPoint& p, int primary, int secondary) {
                                     p.x += primary;
                                     p.y += secondary;
                                 },
                                 [](IntPoint& p) { p.x++; }, [](IntPoint& p, int value) { p.y += value; }};

Incrementor Incrementors::SSE = {[](IntPoint& p, int primary, int secondary) {
                                     p.y += primary;
                                     p.x += secondary;
                                 },
                                 [](IntPoint& p) { p.y++; }, [](IntPoint& p, int value) { p.x += value; }};

Incrementor Incrementors::SSW = {[](IntPoint& p, int primary, int secondary) {
                                     p.y += primary;
                                     p.x -= secondary;
                                 },
                                 [](IntPoint& p) { p.y++; }, [](IntPoint& p, int value) { p.x -= value; }};

Incrementor Incrementors::SWW = {[](IntPoint& p, int primary, int secondary) {
                                     p.x -= primary;
                                     p.y += secondary;
                                 },
                                 [](IntPoint& p) { p.x--; }, [](IntPoint& p, int value) { p.y += value; }};

Incrementor Incrementors::NWW = {[](IntPoint& p, int primary, int secondary) {
                                     p.x -= primary;
                                     p.y -= secondary;
                                 },
                                 [](IntPoint& p) { p.x--; }, [](IntPoint& p, int value) { p.y -= value; }};

Incrementor Incrementors::NNW = {[](IntPoint& p, int primary, int secondary) {
                                     p.y -= primary;
                                     p.x -= secondary;
                                 },
                                 [](IntPoint& p) { p.y--; }, [](IntPoint& p, int value) { p.x -= value; }};

Incrementor Incrementors::NNE = {[](IntPoint& p, int primary, int secondary) {
                                     p.y -= primary;
                                     p.x += secondary;
                                 },
                                 [](IntPoint& p) { p.y--; }, [](IntPoint& p, int value) { p.x += value; }};

Incrementor Incrementors::NEE = {[](IntPoint& p, int primary, int secondary) {
                                     p.x += primary;
                                     p.y -= secondary;
                                 },
                                 [](IntPoint& p) { p.x++; }, [](IntPoint& p, int value) { p.y -= value; }};

Incrementor* Incrementors::get(Octant octant) {
    switch (octant) {
        case Octant::SEE:
            return &SEE;
        case Octant::SSE:
            return &SSE;
        case Octant::SSW:
            return &SSW;
        case Octant::SWW:
            return &SWW;
        case Octant::NWW:
            return &NWW;
        case Octant::NNW:
            return &NNW;
        case Octant::NNE:
            return &NNE;
        case Octant::NEE:
            return &NEE;
        default:
            throw std::invalid_argument("Invalid octant");
    }
}

/******************************************************************************************************************
 *            === RAY ITERATORS ===
 **********************************************************************************************************************/
RayIterator::RayIterator() : point(IntPoint::Invalid()) {}

RayIterator::RayIterator(const IntPoint& start, Point direction) : point(start), _error(0) {
    if (direction.is_null()) {
        point = IntPoint::Invalid();
        _delta = 0;
        _octant = Octant::SEE;
        _incrementor = &Incrementors::SEE;  // Default incrementor
        return;
    }

    if (direction.x > 0) {
        if (direction.y > 0) {
            if (direction.x > direction.y) {
                _delta = direction.y / direction.x;
                _octant = Octant::SEE;
                _incrementor = &Incrementors::SEE;
            } else {
                _delta = direction.x / direction.y;
                _octant = Octant::SSE;
                _incrementor = &Incrementors::SSE;
            }
        } else if (direction.y < 0) {
            if (direction.x > -direction.y) {
                _delta = -direction.y / direction.x;
                _octant = Octant::NEE;
                _incrementor = &Incrementors::NEE;
            } else {
                _delta = -direction.x / direction.y;
                _octant = Octant::NNE;
                _incrementor = &Incrementors::NNE;
            }
        } else {
            _delta = 0;
            _octant = Octant::SEE;
            _incrementor = &Incrementors::SEE;  // Default incrementor
        }
    } else {
        if (direction.y > 0) {
            if (-direction.x > direction.y) {
                _delta = -direction.y / direction.x;
                _octant = Octant::SWW;
                _incrementor = &Incrementors::SWW;
            } else {
                _delta = -direction.x / direction.y;
                _octant = Octant::SSW;
                _incrementor = &Incrementors::SSW;
            }
        } else {
            if (direction.x < direction.y) {
                _delta = direction.y / direction.x;
                _octant = Octant::NWW;
                _incrementor = &Incrementors::NWW;
            } else {
                _delta = direction.x / direction.y;
                _octant = Octant::NNW;
                _incrementor = &Incrementors::NNW;
            }
        }
    }
}

RayIterator::RayIterator(const IntPoint& start, float delta, Octant octant)
    : _delta(delta), _octant(octant), point(start), _error(0) {
    _incrementor = Incrementors::get(octant);
}

RayIterator::RayIterator(const IntPoint& start, float delta, Octant octant, Incrementor* incrementor)
    : _delta(delta), _incrementor(incrementor), _octant(octant), point(start), _error(0) {}

void RayIterator::reset(const IntPoint& start) {
    point = start;
    _error = 0;
}

void RayIterator::reset_error() { _error = 0; }

const IntPoint& RayIterator::operator*() const { return point; }
const int& RayIterator::y() const { return point.y; }
const int& RayIterator::x() const { return point.x; }
const float& RayIterator::delta() const { return _delta; }
const Octant& RayIterator::octant() const { return _octant; }
const float& RayIterator::error() const { return _error; }

bool RayIterator::operator!=(const RayIterator& other) { return point != other.point; }

const IntPoint& RayIterator::operator++() {
    iter();
    return point;
}

bool RayIterator::iter() {
    _incrementor->stepMain(point);
    _error += _delta;
    float inc = round(_error);
    if (inc > 0) {
        _error -= inc;
        _incrementor->incrSecondary(point, inc);
        return true;
    }
    return false;
}

IntPoint RayIterator::previousHalfStep() const {
    IntPoint p = point;
    _incrementor->incrSecondary(p, -1);
    return p;
}

IntPoint RayIterator::extrapolate(int step) const {
    if (step == 0) return point;

    int stepDelta = floor(step * _delta);
    IntPoint p = point;
    _incrementor->incr(p, step, stepDelta);
    return p;
}

int RayIterator::stepTo(const IntPoint& p) const {
    if (_delta == 0) return 0;
    switch (_octant) {
        case Octant::SEE:
        case Octant::NEE:
            return p.x - point.x;
        case Octant::SSE:
        case Octant::SSW:
            return p.y - point.y;
            return p.y - point.y;
        case Octant::SWW:
        case Octant::NWW:
            return point.x - p.x;
        case Octant::NNW:
        case Octant::NNE:
            return point.y - p.y;

        default:
            return 0;
    }
}

void RayIterator::skip(int step) {
    _error += step * _delta;
    int stepDelta = round(_error);
    _error -= stepDelta;
    _incrementor->incr(point, step, stepDelta);
}

CountingRayIterator::CountingRayIterator() : RayIterator() {}

CountingRayIterator::CountingRayIterator(const IntPoint& start, Point direction) : RayIterator(start, direction) {}

bool CountingRayIterator::operator!=(const RayIterator& other) {
    return RayIterator::operator!=(other) || _count != static_cast<const CountingRayIterator&>(other)._count;
}
const IntPoint& CountingRayIterator::operator++() {
    _count++;
    return RayIterator::operator++();
}

int CountingRayIterator::step() const { return _count; }

bool CountingRayIterator::iter() {
    _count++;
    return RayIterator::iter();
}
void CountingRayIterator::skip(int step) {
    _count += step;
    RayIterator::skip(step);
}

void CountingRayIterator::reset(const IntPoint& start) {
    RayIterator::reset(start);
    _count = 0;
}

/**********************************************************************************************************************
 *            === CONE ITERATOR ===
 **********************************************************************************************************************/
ConeIterator::ConeIterator(const IntPoint& start, Point direction, float angle) : _start(start) {
    _rightRay = direction.rotate(angle / 2);
    _rightRay /= _rightRay.abs().max();
    _leftRayIter = RayIterator(start, direction.rotate(-angle / 2));
    transversalIter = RayIterator(start, direction.rot270());
}

const IntPoint& ConeIterator::operator*() const { return *transversalIter; }
const int& ConeIterator::height() const { return _height; }
const IntPoint& ConeIterator::start() const { return _start; }
const RayIterator& ConeIterator::leftRay() const { return _leftRayIter; }
const Point& ConeIterator::rightRayDirection() const { return _rightRay; }

const IntPoint& ConeIterator::operator++() {
    iter();
    return *transversalIter;
}

/**
 * @brief Iterate over the pixels of a cone defined by two rays.
 *
 * @return True if the iterator has moved to a new line, false otherwise.
 */
bool ConeIterator::iter() {
    // TODO: Currently, if angle >= 45 some points are returned twice.

    if (interstice) {
        // Find the next gap to fill...
        while (!transversalIter.iter());
    } else
        // Or walk along the transversal ray...
        transversalIter.iter();

    // ... and check that it is not beyond the right ray
    const auto& p = *transversalIter;
    if (!beyondRightRay(p)) return false;

    // Otherwise, advance the left ray and check if we need to fill interstices
    if (!interstice) {
        interstice = _leftRayIter.iter();
        if (interstice) {
            // To fill interstice:
            // - Shift the current left ray pixel a half step to the right
            IntPoint shiftedP = *_leftRayIter;
            const auto& leftRayOctant = _leftRayIter.octant();
            if (leftRayOctant < Octant::NWW) {
                if (leftRayOctant < Octant::SWW)
                    shiftedP.y--;  // SWW, SSW: shift up
                else
                    shiftedP.x++;  // SSE, SEE: shift right

            } else {
                if (leftRayOctant < Octant::NNE)
                    shiftedP.y++;  // NEE, NNE: shift down
                else
                    shiftedP.x--;  // NNW, NWW: shift left
            }

            // - Place the transversal iterator at the shifted pixel and find the next
            // gap to fill
            transversalIter.reset(shiftedP);
            while (!beyondRightRay(*transversalIter)) {
                if (transversalIter.iter()) {
                    if (beyondRightRay(*transversalIter)) break;
                    return false;
                }
            }
            // - If the next gap is beyond the right ray, proceed normally
            interstice = false;
        }
    } else {
        interstice = false;
    }

    // If we don't need to fill interstices, place the transversal iterator on the
    // left ray pixel (previously advanced)
    transversalIter.reset(*_leftRayIter);
    ++_height;
    return true;  // We began a new line
}

bool ConeIterator::beyondRightRay(const IntPoint& p) { return _rightRay.cross(p - _start + Point(0.5, 0.5)) < 0; }

const RayIterator& ConeIterator::transversalRay() const { return transversalIter; }

/**********************************************************************************************************************
 *            === Triangle ITERATOR ===
 **********************************************************************************************************************/
TriangleIterator::TriangleIterator(const IntPoint& v0, const IntPoint& v1, const IntPoint& v2)
    : _v0(v0), _v1(v1), _v2(v2) {
    if (v0 == v1) {
        _ratioTransverseMain = 0;
        v2Reached = true;
        return;
    }

    Point main = Point(v1 - v0);
    _mainRay = CountingRayIterator(v0, main.normalize());

    if (v1 == v2) {
        _ratioTransverseMain = 0;
        _oppositeEdgeDir = main.normalize();
        _transversalRay = CountingRayIterator(v0, Point(1, 0));
        return;
    }

    Point transverse = Point(v2 - v1);
    Point opposite = Point(v2 - v0);
    _ratioTransverseMain = transverse.abs().max() / main.abs().max();
    _oppositeEdgeDir = opposite.normalize();
    _transversalRay = CountingRayIterator(v0, transverse.normalize());
}

const IntPoint& TriangleIterator::operator*() const { return *_transversalRay; }
const IntPoint& TriangleIterator::point() const { return *_transversalRay; }
const IntPoint& TriangleIterator::start() const { return _v0; }
const RayIterator& TriangleIterator::mainRay() const { return _mainRay; }
const RayIterator& TriangleIterator::transversalRay() const { return _transversalRay; }
const Point& TriangleIterator::oppositeEdgeDirection() const { return _oppositeEdgeDir; }
bool TriangleIterator::finished() const { return v2Reached; }
float TriangleIterator::relativeTraversalHeight() const { return (float)_transversalRay.step() / _traversalHeight; }

const IntPoint& TriangleIterator::operator++() {
    iter();
    return *_transversalRay;
}

bool TriangleIterator::iter() {
    if (v2Reached) return false;

    if (interstice) {
        // If we are filling interstices, find the next gap to fill...
        while (_transversalRay.step() + 2 < _traversalHeight) {
            if (!_transversalRay.iter()) {
                std::cout << "(" << point().y << ", " << point().x << "| " << relativeTraversalHeight() << " ) inter2"
                          << std::endl;
                return true;
            }
        }
    } else if (_transversalRay.step() + 1 < _traversalHeight) {
        // Otherwise walk along the transversal ray
        _transversalRay.iter();
        std::cout << "(" << point().y << ", " << point().x << "| " << relativeTraversalHeight() << " ) main"
                  << std::endl;
        return true;
    }

    // In either case, if we reached the opposite edge, attempt to start a new line
    if (!interstice) {
        // Check first if we reached the end of the main ray
        if (*_mainRay == _v1) {
            v2Reached = true;
            return false;  // We reached the end of the triangle
        }

        // Then advance the main ray to a new line and check if we need to fill interstices
        IntPoint shiftedP = *_mainRay;
        interstice = _mainRay.iter();
        _traversalHeight = floor(_mainRay.step() * _ratioTransverseMain) + 1;

        if (interstice) {
            // To fill interstice:
            // - Shift the current main ray pixel a "half step" in the major direction of the main ray
            switch (_mainRay.octant()) {
                case Octant::SSE:
                case Octant::SSW:
                    shiftedP.y++;  // Shift down
                    break;
                case Octant::NNW:
                case Octant::NNE:
                    shiftedP.y--;  // Shift up
                    break;
                case Octant::SEE:
                case Octant::NEE:
                    shiftedP.x++;  // Shift right
                    break;
                case Octant::SWW:
                case Octant::NWW:
                    shiftedP.x--;  // Shift left
                    break;
            }

            // - Place the transversal iterator at the shifted pixel and find the next gap to fill
            _transversalRay.reset(shiftedP);
            std::cout << "T: (" << point().y << ", " << point().x << " )" << std::endl;
            while (_transversalRay.step() + 2 < _traversalHeight) {
                if (_transversalRay.iter()) {
                    std::cout << "(" << point().y << ", " << point().x << "| " << relativeTraversalHeight()
                              << " ) inter" << std::endl;
                    return true;
                }
            }
            // - If the next gap is beyond the right ray, proceed normally
            interstice = false;
        }
    } else {
        interstice = false;
    }

    // If we don't need to fill interstices, place the transversal iterator on the
    // main ray pixel (previously advanced) and start a new line
    _transversalRay.reset(*_mainRay);
    std::cout << "(" << point().y << ", " << point().x << "| " << relativeTraversalHeight() << " ) newline"
              << std::endl;
    return true;
}

bool TriangleIterator::nextStepBeyondOppositeEdge() { return _transversalRay.step() + 1 >= _traversalHeight; }

/**********************************************************************************************************************
 *            === Simple Triangle ITERATOR ===
 **********************************************************************************************************************/

SimpleTriangleIterator::SimpleTriangleIterator(const IntPoint& v0, const IntPoint& v1, const IntPoint& v2)
    : _v0(v0), _v1(v1), _v2(v2), _point(v0) {
    if (v0 == v1) {
        width = 0;
        return;
    } else if (v1 == v2 || v0 == v2) {
        _v2 = v1;  // If two vertices are equal, treat the triangle as a line
        _edge01 = CountingRayIterator(v0, v1 - v0);
        _edge02 = RayIterator(v0, v1 - v0);
        _traversalIncr = &Incrementors::SEE;
        e01width = e02width = width = (v1 - v0).abs().max();
        return;
    }

    IntPoint e01 = v1 - v0;
    IntPoint e02 = v2 - v0;
    IntPoint e12 = v2 - v1;
    Point u01 = e01.normalize();
    Point u02 = e02.normalize();
    Point u12 = e12.normalize();

    // Compute the height direction towards the opposite vertex
    Point h = u01.cross(u02) > 0 ? u01.rot90() : u01.rot270();

    // The traversal direction is the nearest direction to h
    Octant traversalDir;
    bool flipV0V1 = false;

    traverseVertically = abs(h.y) > abs(h.x);
    if (traverseVertically) {
        traversalDir = h.y > 0 ? Octant::SSE : Octant::NNW;  // Traverse vertically
        if ((e01.x > 0) != (e02.x > 0)) flipV0V1 = true;     // Flip if v0 is between v1 and v2 horizontally
    } else {
        traversalDir = h.x > 0 ? Octant::SEE : Octant::SWW;  // Traverse horizontally
        if ((e01.y > 0) != (e02.y > 0)) flipV0V1 = true;     // Flip if v0 is between v1 and v2 vertically
    }
    _traversalIncr = Incrementors::get(traversalDir);

    // Flip v0 and v1 if needed to ensure v0 is at an extreme of the triangle
    if (flipV0V1) {
        _point = v1;          // Start at v1
        std::swap(_v0, _v1);  // Swap v0 and v1
        e01 = -e01;           // Flip the edge direction
        u01 = -u01;           // Flip the unit vector direction
        std::swap(u02, u12);  // Swap the other edges to maintain the triangle structure
        std::swap(e02, e12);
    }

    _edge01 = CountingRayIterator(_v0, u01);
    _edge02 = RayIterator(_v0, u02);

    if (traverseVertically) {
        e01width = abs(e01.x);
        e02width = abs(e02.x);
    } else {
        e01width = abs(e01.y);
        e02width = abs(e02.y);
    }

    width = std::max(e01width, e02width);
    if (e01width <= e02width)
        _edge12 = RayIterator(_v1, u12);
    else
        _edge12 = RayIterator(_v2, -u12);  // Reverse the edge direction if needed

    // Compute initial traversal length
    updateTraversalLength();
}

const IntPoint& SimpleTriangleIterator::operator*() const { return _point; }
const IntPoint& SimpleTriangleIterator::operator++() {
    iter();
    return _point;
}
const IntPoint& SimpleTriangleIterator::point() const { return _point; }

bool SimpleTriangleIterator::finished() const { return _edge01.step() > width; }

bool SimpleTriangleIterator::iter() {
    // If we reached the height, move to a new line
    if (_traversalStep >= _traversalLength) {
        // - Advance along edge 01
        _edge01.iter();
        if (finished()) {
            _point = e01width == width ? _v1 : _v2;
            return false;
        }

        updateTraversalLength();

        // - Move the current point to the main ray position
        _traversalStep = 0;
        _point = *_edge01;  // Reset the point to the main ray position

        // - Skip the first pixels of the traversal if needed
        if (_edge01.step() > e01width) {
            int skipStep;
            if (traverseVertically) {
                // Advance along edge 12 until we reached the same column as edge 01
                while (_edge01.x() != _edge12.x()) _edge12.iter();
                skipStep = abs(_edge01.y() - _edge12.y());
            } else {
                // Advance along edge 12 until we reached the same row as edge 01
                while (_edge01.y() != _edge12.y()) _edge12.iter();
                skipStep = abs(_edge01.x() - _edge12.x());
            }
            _traversalStep = skipStep;
            _traversalIncr->incr(_point, skipStep, 0);
        }
        return true;
    }
    // Otherwise, walk along the transversal ray
    _traversalStep++;
    _traversalIncr->stepMain(_point);
    return true;
}

void SimpleTriangleIterator::updateTraversalLength() {
    if (_v1 == _v2) {
        _traversalLength = 0;  // If v1 and v2 are the same, no traversal is needed
        return;
    }

    if (_edge01.step() < e02width) {
        // == Compute the traversal length from edge 01 to edge 02 ==
        IntPoint lastP = *_edge02;
        if (traverseVertically) {
            // Advance along edge 02 until we reached the same column as edge 01
            while (_edge01.x() == (++_edge02).x) lastP = *_edge02;
            _traversalLength = abs(_edge01.y() - lastP.y);
        } else {
            // Advance along edge 02 until we reached the same row as edge 01
            while (_edge01.y() == (++_edge02).y) lastP = *_edge02;
            _traversalLength = abs(_edge01.x() - lastP.x);
        }
    } else {
        // == Compute the traversal length from edge 01 to edge 21 ==
        if (traverseVertically) {
            // Advance along edge 21 until we reached the same column as edge 01
            while (_edge01.x() != _edge12.x()) _edge12.iter();
            _traversalLength = abs(_edge01.y() - _edge12.y());
        } else {
            // Advance along edge 21 until we reached the same row as edge 01
            while (_edge01.y() != _edge12.y()) _edge12.iter();
            _traversalLength = abs(_edge01.x() - _edge12.x());
        }
    }
}

float SimpleTriangleIterator::relativeHeight() const {
    if (_traversalLength == 0) return 0.0f;
    return static_cast<float>(_traversalStep) / static_cast<float>(_traversalLength);
}