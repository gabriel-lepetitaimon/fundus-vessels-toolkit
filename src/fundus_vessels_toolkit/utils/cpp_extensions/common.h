#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <tuple>
#include <iostream>

#include <omp.h>
#include <torch/extension.h>

template <typename T>
using Tensor2DAccessor = at::TensorAccessor<T, 2UL, at::DefaultPtrTraits, signed long>;

using IntPair = std::array<int, 2>;

struct Point
{
    int y;
    int x;

    // default + parameterized constructor
    Point(int y=0, int x=0) 
        : y(y), x(x)
    {
    }

    Point(IntPair yx) 
        : y(yx[0]), x(yx[1])
    {
    }

    // assignment operator modifies object, therefore non-const
    Point& operator=(const Point& p)
    {
        x=p.x;
        y=p.y;
        return *this;
    }

    Point operator+(const Point& p) const
    {
        return Point(p.y+y, p.x+x);
    }

    Point operator-(const Point& p) const
    {
        return Point(y-p.y, x-p.x);
    }

    bool operator==(const Point& p) const
    {
        return (x == p.x && y == p.y);
    }

    bool operator!=(const Point& p) const
    {
        return (x != p.x || y != p.y);
    }

    bool is_inside(int H, int W) const
    {
        return (y >= 0 && y < H && x >= 0 && x < W);
    }
    bool is_inside(const Point& size) const
    {
        return (y >= 0 && y < size.y && x >= 0 && x < size.x);
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& p)
    {
        os << "(" << p.y << ", " << p.x << ")";
        return os;
    }

    IntPair toIntPair() const
    {
        return {y, x};
    }
};

float distance(const Point& p1, const Point& p2){
    return sqrt(pow(p1.y - p2.y, 2) + pow(p1.x - p2.x, 2));
}

#pragma omp declare reduction(merge : std::vector<Point> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction(merge : std::vector<std::vector<Point>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

struct PointWithID : Point {
    int id;

    PointWithID(int y=0, int x=0, int id=0) 
        : Point(y, x), id(id)
    {
    }

    PointWithID(const Point& p, int id=0) 
        : Point(p), id(id)
    {
    }

    PointWithID& operator=(const PointWithID& p)
    {
        x=p.x;
        y=p.y;
        id=p.id;
        return *this;
    }

    bool operator==(const PointWithID& p) const
    {
        return (x == p.x && y == p.y && id == p.id);
    }

    bool operator!=(const PointWithID& p) const
    {
        return (x != p.x || y != p.y || id != p.id);
    }

    bool isSamePoint(const Point& p) const
    {
        return (x == p.x && y == p.y);
    }
    
    Point point() const
    {
        return Point(y, x);
    }

    friend std::ostream& operator<<(std::ostream& os, const PointWithID& p)
    {
        os << "(" << p.y << ", " << p.x << ", id=" << p.id <<")";
        return os;
    }
};

#pragma omp declare reduction(merge : std::vector<PointWithID> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))


#endif // COMMON_H