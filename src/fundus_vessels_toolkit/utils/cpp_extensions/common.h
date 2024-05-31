#include <vector>
#include <tuple>
#include <iostream>

#include <omp.h>
#include <torch/extension.h>

template <typename T>
using Tensor2DAccessor = at::TensorAccessor<T, 2UL, at::DefaultPtrTraits, signed long>;

typedef struct {
    int y;
    int x;
} Point;

#pragma omp declare reduction(merge : std::vector<Point> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

typedef struct {
    int y;
    int x;
    int id;
} PointWithID;

#pragma omp declare reduction(merge : std::vector<PointWithID> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
