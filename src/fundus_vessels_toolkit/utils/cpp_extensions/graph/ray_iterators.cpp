#include "ray_iterators.h"

RayIterator::RayIterator() : point(IntPoint::Invalid()) {}

RayIterator::RayIterator(const IntPoint& start, Point direction) : point(start), error(0) {
    if (direction.is_null()) {
        point = IntPoint::Invalid();
        delta = 0;
        _iter = &RayIterator::iterSWW;
        _octant = Octant::SWW;
        return;
    }

    if (direction.x > 0) {
        if (direction.y > 0) {
            if (direction.x > direction.y) {
                _iter = &RayIterator::iterSWW;
                delta = direction.y / direction.x;
                _octant = Octant::SWW;
            } else {
                _iter = &RayIterator::iterSSW;
                delta = direction.x / direction.y;
                _octant = Octant::SSW;
            }
        } else {
            if (direction.x > -direction.y) {
                _iter = &RayIterator::iterNWW;
                delta = -direction.y / direction.x;
                _octant = Octant::NWW;
            } else {
                _iter = &RayIterator::iterNNW;
                delta = -direction.x / direction.y;
                _octant = Octant::NNW;
            }
        }
    } else {
        if (direction.y > 0) {
            if (-direction.x > direction.y) {
                _iter = &RayIterator::iterSEE;
                delta = -direction.y / direction.x;
                _octant = Octant::SEE;
            } else {
                _iter = &RayIterator::iterSSE;
                delta = -direction.x / direction.y;
                _octant = Octant::SSE;
            }
        } else {
            if (direction.x < direction.y) {
                _iter = &RayIterator::iterNEE;
                delta = direction.y / direction.x;
                _octant = Octant::NEE;
            } else {
                _iter = &RayIterator::iterNNE;
                delta = direction.x / direction.y;
                _octant = Octant::NNE;
            }
        }
    }
}

RayIterator::RayIterator(const IntPoint& start, float delta, bool (RayIterator::*iter)())
    : delta(delta), _iter(iter), point(start), error(0) {}

void RayIterator::reset(const IntPoint& start) {
    point = start;
    error = 0;
}

void RayIterator::reset_error() { error = 0; }

const IntPoint& RayIterator::operator*() const { return point; }
const int& RayIterator::y() const { return point.y; }
const int& RayIterator::x() const { return point.x; }
const Octant& RayIterator::octant() const { return _octant; }

bool RayIterator::operator!=(const RayIterator& other) { return point != other.point; }

const IntPoint& RayIterator::operator++() {
    (this->*_iter)();
    return point;
}

bool RayIterator::iter() { return (this->*_iter)(); }

bool RayIterator::iterSWW() {
    point.x++;
    error += delta;
    if (error > 0.5) {
        point.y++;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterSSW() {
    point.y++;
    error += delta;
    if (error > 0.5) {
        point.x++;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterSSE() {
    point.y++;
    error += delta;
    if (error > 0.5) {
        point.x--;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterSEE() {
    point.x--;
    error += delta;
    if (error > 0.5) {
        point.y++;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterNEE() {
    point.x--;
    error += delta;
    if (error > 0.5) {
        point.y--;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterNNE() {
    point.y--;
    error += delta;
    if (error > 0.5) {
        point.x--;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterNNW() {
    point.y--;
    error += delta;
    if (error > 0.5) {
        point.x++;
        error -= 1;
        return true;
    }
    return false;
}

bool RayIterator::iterNWW() {
    point.x++;
    error += delta;
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

ConeIterator::ConeIterator(const IntPoint& start, Point direction, float angle) : start(start) {
    rightRay = direction.rotate(angle / 2);
    leftRayIter = RayIterator(start, direction.rotate(-angle / 2));
    transversalIter = RayIterator(start, direction.rot270());
}

const IntPoint& ConeIterator::operator*() const { return *transversalIter; }
const int& ConeIterator::height() const { return _height; }

const IntPoint& ConeIterator::operator++() {
    // TODO: Currently, if angle >= 45 some points are returned twice.

    if (interstice) {
        // Find the next gap to fill...
        while (!transversalIter.iter());
    } else
        // Or walk along the transversal ray...
        transversalIter.iter();

    // ... and check that it is not beyond the right ray
    const auto& p = *transversalIter;
    if (!beyondRightRay(p)) return p;

    // Otherwise, advance the left ray and check if we need to fill interstices
    if (!interstice) {
        interstice = leftRayIter.iter();
        if (interstice) {
            // To fill interstice:
            // - Shift the current left ray pixel a half step to the right
            IntPoint shiftedP = *leftRayIter;
            const auto& leftRayOctant = leftRayIter.octant();
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

            // - Place the transversal iterator at the shifted pixel and find the next gap to fill
            transversalIter.reset(shiftedP);
            while (!beyondRightRay(*transversalIter)) {
                if (transversalIter.iter()) {
                    if (beyondRightRay(*transversalIter)) break;
                    return *transversalIter;
                }
            }
            // - If the next gap is beyond the right ray, proceed normally
            interstice = false;
        }
    } else {
        interstice = false;
    }

    // If we don't need to fill interstices, place the transversal iterator on the left ray pixel
    const auto& pLeft = *leftRayIter;
    transversalIter.reset(pLeft);
    ++_height;
    return pLeft;
}

bool ConeIterator::beyondRightRay(const IntPoint& p) { return rightRay.cross(p - start + Point(0.5, 0.5)) < 0; }