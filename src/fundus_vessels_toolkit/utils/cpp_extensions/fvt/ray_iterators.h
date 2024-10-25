#ifndef RAY_ITERATORS_H
#define RAY_ITERATORS_H

#include "common.h"

/**
 * @brief Track the first not-segmented pixels on two semi-infinite lines
 * defined by a start point and a direction. The first line is tracked in the
 * direction of the direction vector, while the second line is tracked in the
 * opposite direction.
 *
 * @param start The start point of the line.
 * @param direction The direction of the line.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the
 * binary segmentation.
 * @param max_distance The maximum distance to track.
 *
 * @return The first point of the line for which the segmentation is false. If
 * no such point is found, return IntPoint::Invalid().
 */
std::array<IntPoint, 2> track_nearest_edges(const IntPoint& start, const Point& direction,
                                            const Tensor2DAcc<bool>& segmentation, int max_distance = 40);

/**
 * @brief Track the nearest non-zero pixel on a cone defined by a start point, a
 * direction and an angle.
 *
 * @param start The tip of the cone.
 * @param direction The direction of the cone.
 * @param angle The angle of the cone.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing a
 * semantic segmentation.
 * @param max_distance The maximum distance to track.
 *
 * @return The first point (label and position) both inside the cone and the
 * segmentation. If no such point is found, return {0, IntPoint::Invalid()}.
 */
std::pair<int, IntPoint> track_nearest_branch(const IntPoint& start, const Point& direction, float angle,
                                              float max_dist, const Tensor2DAcc<int>& branchMap);

/**
 * @brief Draw a line between two points on a tensor.
 *
 * @param start The start point of the line.
 * @param end The end point of the line.
 * @param tensor The tensor on which to draw the line.
 * @param value The value to draw on the tensor.
 */
void draw_line(IntPoint start, IntPoint end, Tensor2DAcc<int>& tensor, int value, int H, int W);
void draw_line(IntPoint start, IntPoint end, Tensor2DAcc<int>& tensor, int value);

/**********************************************************************************************************************
 *            === RAY ITERATORS ===
 **********************************************************************************************************************/

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