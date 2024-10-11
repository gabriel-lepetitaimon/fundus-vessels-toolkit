#ifndef RAY_ITERATORS_H
#define RAY_ITERATORS_H

#include "common.h"

enum class Octant { SWW = 0, SSW = 1, SSE = 2, SEE = 3, NEE = 4, NNE = 5, NNW = 6, NWW = 7 };

class RayIterator {
   public:
    RayIterator();
    RayIterator(const IntPoint& start, Point direction);
    RayIterator(const IntPoint& start, float delta, bool (RayIterator::*iter)());
    void reset(const IntPoint& start);
    void reset_error();

    const IntPoint& operator*() const;
    const int& y() const;
    const int& x() const;
    const Octant& octant() const;

    bool operator!=(const RayIterator& other);
    const IntPoint& operator++();

    bool iterSWW();
    bool iterSSW();
    bool iterSSE();
    bool iterSEE();
    bool iterNEE();
    bool iterNNE();
    bool iterNNW();
    bool iterNWW();

    bool iter();
    IntPoint previousHalfStep() const;

   private:
    float delta;
    bool (RayIterator::*_iter)();
    Octant _octant;

    IntPoint point;
    float error;
};

class ConeIterator {
   public:
    ConeIterator(const IntPoint& start, Point direction, float angle);
    const IntPoint& operator*() const;

    const IntPoint& operator++();
    const int& height() const;

   protected:
    bool beyondRightRay(const IntPoint& p);

   private:
    IntPoint start;
    Point rightRay;
    RayIterator leftRayIter, transversalIter;
    bool interstice = false;
    int _height = 0;
};

#endif  // RAY_ITERATORS_H