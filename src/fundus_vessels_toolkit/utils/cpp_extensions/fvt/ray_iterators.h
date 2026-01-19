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
 *            === RAY ITERATOR ===
 **********************************************************************************************************************/

enum class Octant { SEE = 0, SSE = 1, SSW = 2, SWW = 3, NWW = 4, NNW = 5, NNE = 6, NEE = 7 };

inline bool isPositiveVertically(Octant octant) { return octant >= Octant::NWW; }
inline bool isPositiveHorizontally(Octant octant) { return octant <= Octant::SSE || octant >= Octant::NNE; }
inline Octant oppositeOctant(Octant octant) { return static_cast<Octant>((static_cast<int>(octant) + 4) % 8); }

struct Incrementor {
    void (*incr)(IntPoint& p, int primary, int secondary);
    void (*stepMain)(IntPoint& p);
    void (*incrSecondary)(IntPoint& p, int value);
    int (*stepsBetween)(const IntPoint& start, const IntPoint& p);
};

class Incrementors {
   public:
    static Incrementor SEE, SSE, SSW, SWW, NWW, NNW, NNE, NEE;
    static Incrementor* get(Octant octant);
};

class RayIterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const IntPoint;
    using difference_type = std::ptrdiff_t;
    using pointer = const IntPoint*;
    using reference = const IntPoint&;

    RayIterator();
    RayIterator(const IntPoint& start, Point direction);
    RayIterator(const IntPoint& start, float delta, Octant octant = Octant::SEE);
    RayIterator(const IntPoint& start, float delta, Octant octant, Incrementor* incrementor);
    RayIterator oppositeRay() const;

    bool next();
    RayIterator& operator++();
    RayIterator operator++(int);

    reference operator*() const;
    pointer operator->() const;
    bool operator==(const RayIterator& other) const;
    bool operator!=(const RayIterator& other) const;

    const IntPoint& point() const;
    const int& y() const;
    const int& x() const;
    const float& delta() const;
    const Octant& octant() const;
    const float& error() const;
    int step() const;

    IntPoint previousHalfStep() const;
    IntPoint extrapolate(int step) const;
    RayIterator& skip(int step);
    int stepsCountTo(const IntPoint& p) const;
    void reset();
    void reset(const IntPoint& start);
    void reset_error();

   protected:
    IntPoint _start;
    float _delta;
    Incrementor* _incrementor;
    Octant _octant;

    IntPoint _point;
    float _error;
};

class Line {
   public:
    Line(const IntPoint& p0, const IntPoint& p1, bool skipLast = false, bool skipFirst = false);

    RayIterator begin() const;
    RayIterator end() const;

    const IntPoint& p0() const;
    const IntPoint& p1() const;
    const Point& dir() const;
    const int& length() const;

   protected:
    IntPoint _p0, _p1;
    Point _direction;

    float _delta;
    Octant _octant;
    int _length;
};

/**********************************************************************************************************************
 *            === COMPOSITE ITERATORS ===
 **********************************************************************************************************************/
class ConeIterator {
   public:
    ConeIterator(const IntPoint& start, Point direction, float angle);
    const IntPoint& operator*() const;

    const IntPoint& operator++();
    bool iter();
    const int& height() const;

    const IntPoint& start() const;
    const RayIterator& leftRay() const;
    const RayIterator& transversalRay() const;
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

class TriangleIterator {
   public:
    TriangleIterator(const IntPoint& v0, const IntPoint& v1, const IntPoint& v2);
    const IntPoint& operator*() const;
    const IntPoint& operator++();
    const IntPoint& point() const;

    bool iter();
    bool finished() const;

    float relativeHeight() const;

   private:
    void updateTraversalLength();

    IntPoint _v0, _v1, _v2, _point;
    bool traverseVertically;

    RayIterator _edge01, _edge02, _edge12;
    Incrementor* _traversalIncr;
    int _traversalStep = 0, _traversalLength = 0;

    int width, e01width, e02width;
};

#endif  // RAY_ITERATORS_H