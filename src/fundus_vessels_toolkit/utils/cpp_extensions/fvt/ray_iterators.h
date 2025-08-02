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

enum class Octant { SEE = 0, SSE = 1, SSW = 2, SWW = 3, NWW = 4, NNW = 5, NNE = 6, NEE = 7 };

inline bool isPositiveVertically(Octant octant) { return octant >= Octant::NWW; }
inline bool isPositiveHorizontally(Octant octant) { return octant <= Octant::SSE || octant >= Octant::NNE; }

struct Incrementor {
    void (*incr)(IntPoint& p, int primary, int secondary);
    void (*stepMain)(IntPoint& p);
    void (*incrSecondary)(IntPoint& p, int value);
};

class Incrementors {
   public:
    static Incrementor SEE, SSE, SSW, SWW, NWW, NNW, NNE, NEE;
    static Incrementor* get(Octant octant);
};

class RayIterator {
   public:
    RayIterator();
    RayIterator(const IntPoint& start, Point direction);
    RayIterator(const IntPoint& start, float delta, Octant octant = Octant::SEE);
    RayIterator(const IntPoint& start, float delta, Octant octant, Incrementor* incrementor);
    void reset(const IntPoint& start);
    void reset_error();

    const IntPoint& operator*() const;
    const int& y() const;
    const int& x() const;
    const float& delta() const;
    const Octant& octant() const;
    const float& error() const;

    bool operator!=(const RayIterator& other);
    const IntPoint& operator++();

    bool iter();
    IntPoint previousHalfStep() const;
    IntPoint extrapolate(int step) const;
    void skip(int step);
    int stepTo(const IntPoint& p) const;

   private:
    float _delta;
    Incrementor* _incrementor;
    Octant _octant;

    IntPoint point;
    float _error;
};

class CountingRayIterator : public RayIterator {
   public:
    CountingRayIterator();
    CountingRayIterator(const IntPoint& start, Point direction);

    bool operator!=(const RayIterator& other);
    const IntPoint& operator++();

    bool iter();
    void skip(int step);
    void reset(const IntPoint& start);
    int step() const;

   private:
    int _count = 0;
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

    const IntPoint& start() const;
    const RayIterator& mainRay() const;
    const RayIterator& transversalRay() const;
    const Point& oppositeEdgeDirection() const;

    float relativeTraversalHeight() const;

   protected:
    bool nextStepBeyondOppositeEdge();

   private:
    IntPoint _v0, _v1, _v2;
    Point _oppositeEdgeDir;
    CountingRayIterator _mainRay, _transversalRay;
    int _traversalHeight = 1;
    float _ratioTransverseMain = 0;
    bool interstice = false, v2Reached = false;
};

class SimpleTriangleIterator {
   public:
    SimpleTriangleIterator(const IntPoint& v0, const IntPoint& v1, const IntPoint& v2);
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

    CountingRayIterator _edge01;
    RayIterator _edge02, _edge12;
    Incrementor* _traversalIncr;
    int _traversalStep = 0, _traversalLength = 0;

    int width, e01width, e02width;
};

#endif  // RAY_ITERATORS_H