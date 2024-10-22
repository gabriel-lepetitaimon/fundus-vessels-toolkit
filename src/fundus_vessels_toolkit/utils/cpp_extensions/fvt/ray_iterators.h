#ifndef RAY_ITERATORS_H
#define RAY_ITERATORS_H

#include "common.h"

std::array<IntPoint, 2> track_nearest_edges(
    const IntPoint& start, const Point& direction,
    const Tensor2DAcc<bool>& segmentation, int max_distance = 40);

std::pair<int, IntPoint> track_nearest_branch(
    const IntPoint& start, const Point& direction, float angle, float max_dist,
    const Tensor2DAcc<int>& branchMap);
/**********************************************************************************************************************
 *            === RAY ITERATORS ===
 **********************************************************************************************************************/

enum class Octant {
  SWW = 0,
  SSW = 1,
  SSE = 2,
  SEE = 3,
  NEE = 4,
  NNE = 5,
  NNW = 6,
  NWW = 7
};

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
  const float& delta() const;
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
  IntPoint extrapolate(int step) const;
  int stepTo(const IntPoint& p) const;

 private:
  float _delta;
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
  bool iter();
  const int& height() const;

  const IntPoint& start() const;
  const RayIterator& leftRay() const;
  const Point& rightRayDirection() const;

 protected:
  bool beyondRightRay(const IntPoint& p);

 private:
  IntPoint _start;
  Point _rightRay;
  RayIterator _leftRayIter, transversalIter;
  bool interstice = false;
  int _height = 0;
};

#endif  // RAY_ITERATORS_H