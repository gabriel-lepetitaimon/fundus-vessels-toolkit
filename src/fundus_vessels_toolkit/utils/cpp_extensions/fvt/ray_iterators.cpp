#include "ray_iterators.h"

std::array<IntPoint, 2> track_nearest_edges(const IntPoint& start, const Point& direction,
                                            const Tensor2DAcc<bool>& segmentation, int max_iter) {
    if (direction.is_null()) return {{IntPoint::Invalid(), IntPoint::Invalid()}};
    const int H = segmentation.size(0), W = segmentation.size(1);

    RayIterator dIter = RayIterator(start, direction),  // Direct iterator
        rIter = dIter.oppositeRay();                    // reverse iterator
    bool dSafe = dIter.extrapolate(max_iter).is_inside(H, W), rSafe = rIter.extrapolate(max_iter).is_inside(H, W);

    IntPoint dBound = IntPoint::Invalid(), rBound = IntPoint::Invalid();

    IntPoint lastP = start;
    while ((++dIter).step() != max_iter) {
        const IntPoint& p = dIter.point();
        if (!segmentation[p.y][p.x]) {
            dBound = lastP;
            break;
        } else if (!dSafe && !p.is_inside(H, W))
            break;
        lastP = p;
    }

    lastP = start;
    while ((++rIter).step() != max_iter) {
        const IntPoint& p = rIter.point();
        if (!segmentation[p.y][p.x]) {
            rBound = lastP;
            break;
        } else if (!rSafe && !p.is_inside(H, W))
            break;
        lastP = p;
    }

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
    for (const auto& p : Line(start, end)) {
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
                                 [](IntPoint& p) { p.x++; }, [](IntPoint& p, int value) { p.y += value; },
                                 [](const IntPoint& start, const IntPoint& p) { return p.x - start.x; }};

Incrementor Incrementors::SSE = {[](IntPoint& p, int primary, int secondary) {
                                     p.y += primary;
                                     p.x += secondary;
                                 },
                                 [](IntPoint& p) { p.y++; }, [](IntPoint& p, int value) { p.x += value; },
                                 [](const IntPoint& start, const IntPoint& p) { return p.y - start.y; }};

Incrementor Incrementors::SSW = {[](IntPoint& p, int primary, int secondary) {
                                     p.y += primary;
                                     p.x -= secondary;
                                 },
                                 [](IntPoint& p) { p.y++; }, [](IntPoint& p, int value) { p.x -= value; },
                                 [](const IntPoint& start, const IntPoint& p) { return p.y - start.y; }};

Incrementor Incrementors::SWW = {[](IntPoint& p, int primary, int secondary) {
                                     p.x -= primary;
                                     p.y += secondary;
                                 },
                                 [](IntPoint& p) { p.x--; }, [](IntPoint& p, int value) { p.y += value; },
                                 [](const IntPoint& start, const IntPoint& p) { return start.x - p.x; }};

Incrementor Incrementors::NWW = {[](IntPoint& p, int primary, int secondary) {
                                     p.x -= primary;
                                     p.y -= secondary;
                                 },
                                 [](IntPoint& p) { p.x--; }, [](IntPoint& p, int value) { p.y -= value; },
                                 [](const IntPoint& start, const IntPoint& p) { return start.x - p.x; }};

Incrementor Incrementors::NNW = {[](IntPoint& p, int primary, int secondary) {
                                     p.y -= primary;
                                     p.x -= secondary;
                                 },
                                 [](IntPoint& p) { p.y--; }, [](IntPoint& p, int value) { p.x -= value; },
                                 [](const IntPoint& start, const IntPoint& p) { return start.y - p.y; }};

Incrementor Incrementors::NNE = {[](IntPoint& p, int primary, int secondary) {
                                     p.y -= primary;
                                     p.x += secondary;
                                 },
                                 [](IntPoint& p) { p.y--; }, [](IntPoint& p, int value) { p.x += value; },
                                 [](const IntPoint& start, const IntPoint& p) { return start.y - p.y; }};

Incrementor Incrementors::NEE = {[](IntPoint& p, int primary, int secondary) {
                                     p.x += primary;
                                     p.y -= secondary;
                                 },
                                 [](IntPoint& p) { p.x++; }, [](IntPoint& p, int value) { p.y -= value; },
                                 [](const IntPoint& start, const IntPoint& p) { return p.x - start.x; }};

Incrementor* Incrementors::get(Octant octant) {
    const std::array<Incrementor*, 8> incrementors = {&SEE, &SSE, &SSW, &SWW, &NWW, &NNW, &NNE, &NEE};
    return incrementors[static_cast<int>(octant) % 8];
}

/***********************************************************************************************************************
 *            === RAY ITERATOR ===
 **********************************************************************************************************************/
RayIterator::RayIterator() : _start(IntPoint::Invalid()), _point(IntPoint::Invalid()) {}

RayIterator::RayIterator(const IntPoint& start, Point direction) : _start(start), _point(start), _error(0) {
    if (direction.is_null()) {
        _point = IntPoint::Invalid();
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
    : _start(start), _delta(delta), _octant(octant), _point(start), _error(0) {
    _incrementor = Incrementors::get(octant);
}

RayIterator::RayIterator(const IntPoint& start, float delta, Octant octant, Incrementor* incrementor)
    : _start(start), _delta(delta), _incrementor(incrementor), _octant(octant), _point(start), _error(0) {}

RayIterator RayIterator::oppositeRay() const { return RayIterator(_point, _delta, oppositeOctant(_octant)); }

bool RayIterator::next() {
    _incrementor->stepMain(_point);
    _error += _delta;
    float inc = round(_error);
    if (inc > 0) {
        _error -= inc;
        _incrementor->incrSecondary(_point, inc);
        return true;
    }
    return false;
}

RayIterator& RayIterator::operator++() {
    next();
    return *this;
}
RayIterator RayIterator::operator++(int) {
    RayIterator temp = *this;
    next();
    return temp;
}

const IntPoint& RayIterator::operator*() const { return _point; }
const IntPoint* RayIterator::operator->() const { return &_point; }
bool RayIterator::operator==(const RayIterator& other) const {
    return _point == other._point && _octant == other._octant && _delta == other._delta && _start == other._start;
}
bool RayIterator::operator!=(const RayIterator& other) const {
    return _point != other._point || _octant != other._octant || _delta != other._delta || _start != other._start;
}

const IntPoint& RayIterator::point() const { return _point; }
const int& RayIterator::y() const { return _point.y; }
const int& RayIterator::x() const { return _point.x; }
const float& RayIterator::delta() const { return _delta; }
const Octant& RayIterator::octant() const { return _octant; }
const float& RayIterator::error() const { return _error; }
int RayIterator::step() const { return _incrementor->stepsBetween(_start, _point); }

void RayIterator::reset() {
    _point = _start;
    _error = 0;
}
void RayIterator::reset(const IntPoint& start) {
    _start = start;
    _point = start;
    _error = 0;
}

void RayIterator::reset_error() { _error = 0; }
IntPoint RayIterator::previousHalfStep() const {
    IntPoint p = _point;
    _incrementor->incrSecondary(p, -1);
    return p;
}

IntPoint RayIterator::extrapolate(int step) const {
    if (step == 0) return _point;

    int stepDelta = floor(step * _delta);
    IntPoint p = _point;
    _incrementor->incr(p, step, stepDelta);
    return p;
}

int RayIterator::stepsCountTo(const IntPoint& p) const {
    if (_delta == 0) return 0;
    return _incrementor->stepsBetween(_point, p);
}

RayIterator& RayIterator::skip(int step) {
    _error += step * _delta;
    int stepDelta = round(_error);
    _error -= stepDelta;
    _incrementor->incr(_point, step, stepDelta);
    return *this;
}

/***********************************************************************************************************************
 *            === LINE ===
 **********************************************************************************************************************/
Line::Line(const IntPoint& p0, const IntPoint& p1, bool skipLast, bool skipFirst)
    : _p0(p0), _p1(p1), _direction((p1 - p0).normalize()) {
    if (p0 == p1) {
        _delta = 1;
        _octant = Octant::SEE;  // Default octant
        _length = 0;
        return;
    }

    _direction = (p1 - p0).normalize();
    const auto& it = RayIterator(p0, _direction);
    _delta = it.delta();
    _octant = it.octant();
    _length = it.stepsCountTo(p1);

    if (skipFirst && _length > 0) {
        _p0 = it.extrapolate(1);
        _length--;
    }
    if (skipLast && _length > 0) _length--;
}

RayIterator Line::begin() const { return RayIterator(_p0, _delta, _octant); }
RayIterator Line::end() const { return RayIterator(_p0, _delta, _octant).skip(_length + 1); }

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
        while (!transversalIter.next());
    } else
        // Or walk along the transversal ray...
        transversalIter.next();

    // ... and check that it is not beyond the right ray
    const auto& p = *transversalIter;
    if (!beyondRightRay(p)) return false;

    // Otherwise, advance the left ray and check if we need to fill interstices
    if (!interstice) {
        interstice = _leftRayIter.next();
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
                if (transversalIter.next()) {
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
    : _v0(v0), _v1(v1), _v2(v2), _point(v0) {
    if (v0 == v1) {
        width = 0;
        return;
    } else if (v1 == v2 || v0 == v2) {
        _v2 = v1;  // If two vertices are equal, treat the triangle as a line
        _edge01 = RayIterator(v0, v1 - v0);
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

    _edge01 = RayIterator(_v0, u01);
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

const IntPoint& TriangleIterator::operator*() const { return _point; }
const IntPoint& TriangleIterator::operator++() {
    iter();
    return _point;
}
const IntPoint& TriangleIterator::point() const { return _point; }

bool TriangleIterator::finished() const { return _edge01.step() > width; }

bool TriangleIterator::iter() {
    // If we reached the height, move to a new line
    if (_traversalStep >= _traversalLength) {
        // - Advance along edge 01
        _edge01.next();
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
                while (_edge01.x() != _edge12.x()) _edge12.next();
                skipStep = abs(_edge01.y() - _edge12.y());
            } else {
                // Advance along edge 12 until we reached the same row as edge 01
                while (_edge01.y() != _edge12.y()) _edge12.next();
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

void TriangleIterator::updateTraversalLength() {
    if (_v1 == _v2) {
        _traversalLength = 0;  // If v1 and v2 are the same, no traversal is needed
        return;
    }

    if (_edge01.step() < e02width) {
        // == Compute the traversal length from edge 01 to edge 02 ==
        IntPoint lastP = *_edge02;
        if (traverseVertically) {
            // Advance along edge 02 until we reached the same column as edge 01
            while (_edge01.x() == (++_edge02).x()) lastP = *_edge02;
            _traversalLength = abs(_edge01.y() - lastP.y);
        } else {
            // Advance along edge 02 until we reached the same row as edge 01
            while (_edge01.y() == (++_edge02).y()) lastP = *_edge02;
            _traversalLength = abs(_edge01.x() - lastP.x);
        }
    } else {
        // == Compute the traversal length from edge 01 to edge 21 ==
        if (traverseVertically) {
            // Advance along edge 21 until we reached the same column as edge 01
            while (_edge01.x() != _edge12.x()) _edge12.next();
            _traversalLength = abs(_edge01.y() - _edge12.y());
        } else {
            // Advance along edge 21 until we reached the same row as edge 01
            while (_edge01.y() != _edge12.y()) _edge12.next();
            _traversalLength = abs(_edge01.x() - _edge12.x());
        }
    }
}

float TriangleIterator::relativeHeight() const {
    if (_traversalLength == 0) return 0.0f;
    return static_cast<float>(_traversalStep) / static_cast<float>(_traversalLength);
}