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
                dIter = RayIterator(start, delta, &RayIterator::iterSWW);
                rIter = RayIterator(start, delta, &RayIterator::iterNEE);
            } else {  // SSW
                float delta = direction.x / direction.y;
                dIter = RayIterator(start, delta, &RayIterator::iterSSW);
                rIter = RayIterator(start, delta, &RayIterator::iterNNE);
            }
            dSafe = start.y + max_iter < H && start.x + max_iter < W;
            rSafe = start.y >= max_iter && start.x >= max_iter;
        } else {                               // Direct:  East
            if (direction.x > -direction.y) {  // SEE
                float delta = -direction.y / direction.x;
                dIter = RayIterator(start, delta, &RayIterator::iterNWW);
                rIter = RayIterator(start, delta, &RayIterator::iterSEE);
            } else {  // SSE
                float delta = -direction.x / direction.y;
                dIter = RayIterator(start, delta, &RayIterator::iterNNW);
                rIter = RayIterator(start, delta, &RayIterator::iterSSE);
            }
            dSafe = start.y + max_iter < H && start.x >= max_iter;
            rSafe = start.y >= max_iter && start.x + max_iter < W;
        }
    } else {                                   // Direct: North
        if (direction.y > 0) {                 // Direct:  West
            if (-direction.x > direction.y) {  // NWW
                float delta = -direction.y / direction.x;
                dIter = RayIterator(start, delta, &RayIterator::iterSEE);
                rIter = RayIterator(start, delta, &RayIterator::iterNWW);
            } else {  // NNW
                float delta = -direction.x / direction.y;
                dIter = RayIterator(start, delta, &RayIterator::iterSSE);
                rIter = RayIterator(start, delta, &RayIterator::iterNNW);
            }
            dSafe = start.y >= max_iter && start.x + max_iter < W;
            rSafe = start.y + max_iter < H && start.x >= max_iter;
        } else {                              // Direct:  East
            if (direction.x < direction.y) {  // NEE
                float delta = direction.y / direction.x;
                dIter = RayIterator(start, delta, &RayIterator::iterNEE);
                rIter = RayIterator(start, delta, &RayIterator::iterSWW);
            } else {  // NNE
                float delta = direction.x / direction.y;
                dIter = RayIterator(start, delta, &RayIterator::iterNNE);
                rIter = RayIterator(start, delta, &RayIterator::iterSSW);
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
 *            === RAY ITERATORS ===
 **********************************************************************************************************************/
RayIterator::RayIterator() : point(IntPoint::Invalid()) {}

RayIterator::RayIterator(const IntPoint& start, Point direction) : point(start), error(0) {
    if (direction.is_null()) {
        point = IntPoint::Invalid();
        _delta = 0;
        _iter = &RayIterator::iterSWW;
        _octant = Octant::SWW;
        return;
    }

    if (direction.x > 0) {
        if (direction.y > 0) {
            if (direction.x > direction.y) {
                _iter = &RayIterator::iterSWW;
                _delta = direction.y / direction.x;
                _octant = Octant::SWW;
            } else {
                _iter = &RayIterator::iterSSW;
                _delta = direction.x / direction.y;
                _octant = Octant::SSW;
            }
        } else {
            if (direction.x > -direction.y) {
                _iter = &RayIterator::iterNWW;
                _delta = -direction.y / direction.x;
                _octant = Octant::NWW;
            } else {
                _iter = &RayIterator::iterNNW;
                _delta = -direction.x / direction.y;
                _octant = Octant::NNW;
            }
        }
    } else {
        if (direction.y > 0) {
            if (-direction.x > direction.y) {
                _iter = &RayIterator::iterSEE;
                _delta = -direction.y / direction.x;
                _octant = Octant::SEE;
            } else {
                _iter = &RayIterator::iterSSE;
                _delta = -direction.x / direction.y;
                _octant = Octant::SSE;
            }
        } else {
            if (direction.x < direction.y) {
                _iter = &RayIterator::iterNEE;
                _delta = direction.y / direction.x;
                _octant = Octant::NEE;
            } else {
                _iter = &RayIterator::iterNNE;
                _delta = direction.x / direction.y;
                _octant = Octant::NNE;
            }
        }
    }
}

RayIterator::RayIterator(const IntPoint& start, float delta, bool (RayIterator::*iter)())
    : _delta(delta), _iter(iter), point(start), error(0) {}

void RayIterator::reset(const IntPoint& start) {
    point = start;
    error = 0;
}

void RayIterator::reset_error() { error = 0; }

const IntPoint& RayIterator::operator*() const { return point; }
const int& RayIterator::y() const { return point.y; }
const int& RayIterator::x() const { return point.x; }
const float& RayIterator::delta() const { return _delta; }
const Octant& RayIterator::octant() const { return _octant; }

bool RayIterator::operator!=(const RayIterator& other) { return point != other.point; }

const IntPoint& RayIterator::operator++() {
    (this->*_iter)();
    return point;
}

bool RayIterator::iter() { return (this->*_iter)(); }

bool RayIterator::iterSWW() {
    point.x++;
    error += _delta;
    if (error > 0.5) {
        point.y++;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterSSW() {
    point.y++;
    error += _delta;
    if (error > 0.5) {
        point.x++;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterSSE() {
    point.y++;
    error += _delta;
    if (error > 0.5) {
        point.x--;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterSEE() {
    point.x--;
    error += _delta;
    if (error > 0.5) {
        point.y++;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterNEE() {
    point.x--;
    error += _delta;
    if (error > 0.5) {
        point.y--;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterNNE() {
    point.y--;
    error += _delta;
    if (error > 0.5) {
        point.x--;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterNNW() {
    point.y--;
    error += _delta;
    if (error > 0.5) {
        point.x++;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterNWW() {
    point.x++;
    error += _delta;
    if (error > 0.5) {
        point.y--;
        error -= 1;
        return true;
    }
    return false;
}

IntPoint RayIterator::previousHalfStep() const {
    switch (_octant) {
        case Octant::SWW:
            return IntPoint(point.y - 1, point.x);
        case Octant::SSW:
            return IntPoint(point.y - 1, point.x);
        case Octant::SSE:
            return IntPoint(point.y, point.x + 1);
        case Octant::SEE:
            return IntPoint(point.y, point.x + 1);
        case Octant::NEE:
            return IntPoint(point.y + 1, point.x);
        case Octant::NNE:
            return IntPoint(point.y + 1, point.x);
        case Octant::NNW:
            return IntPoint(point.y, point.x - 1);
        case Octant::NWW:
            return IntPoint(point.y, point.x - 1);
        default:
            return IntPoint::Invalid();
    }
}

IntPoint RayIterator::extrapolate(int step) const {
    if (step == 0) return point;

    int stepDelta = floor(step * _delta);
    switch (_octant) {
        case Octant::SWW:
            return IntPoint(point.y + stepDelta, point.x + step);
        case Octant::SSW:
            return IntPoint(point.y + step, point.x + stepDelta);
        case Octant::SSE:
            return IntPoint(point.y + step, point.x - stepDelta);
        case Octant::SEE:
            return IntPoint(point.y + stepDelta, point.x - step);
        case Octant::NEE:
            return IntPoint(point.y - stepDelta, point.x - step);
        case Octant::NNE:
            return IntPoint(point.y - step, point.x - stepDelta);
        case Octant::NNW:
            return IntPoint(point.y - step, point.x + stepDelta);
        case Octant::NWW:
            return IntPoint(point.y - stepDelta, point.x + step);
        default:
            return IntPoint::Invalid();
    }
}

int RayIterator::stepTo(const IntPoint& p) const {
    if (_delta == 0) return 0;
    switch (_octant) {
        case Octant::SWW:
        case Octant::NWW:
            return p.x - point.x;
        case Octant::SSW:
        case Octant::SSE:
            return p.y - point.y;
            return p.y - point.y;
        case Octant::SEE:
        case Octant::NEE:
            return point.x - p.x;
        case Octant::NNE:
        case Octant::NNW:
            return point.y - p.y;

        default:
            return 0;
    }
}

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
            if (leftRayOctant < Octant::NEE) {
                if (leftRayOctant < Octant::SEE)
                    shiftedP.y--;  // SWW, SSW: shift up
                else
                    shiftedP.x++;  // SSE, SEE: shift right

            } else {
                if (leftRayOctant < Octant::NNW)
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