#distutils: language=c++
#cython: language_level=3, wraparound=False, boundscheck=False, cdivision=True, nonecheck=False

cimport cython
from cython.parallel import prange
from cython.operator import dereference as deref, preincrement as inc
import numpy as np
cimport numpy as np
np.import_array()

from libcpp cimport bool
from libcpp.list cimport list as cpplist

ctypedef np.npy_int INT
ctypedef np.npy_uint8 UINT8
ctypedef np.npy_bool BOOL


cdef struct Point:
    int y
    int x


def fix_hollow_cross(np.ndarray[UINT8, ndim=2, cast=True] skeleton):
    """
    Search for hollow crosses in the skeleton and remove them.

    Parameters
    ----------
    skeleton : np.ndarray[UINT8, ndim=2, cast=True]
        Skeleton image to label. Must be a uint8 image with 0 for background and positive values for skeleton pixels.
        The algorithm assumes that endpoints are encoded 1, branches 2 and junctions 3+. 
            (Encoding the connectivity of the pixel to its neighbors.)
    
    Returns
    -------
    np.ndarray[UINT32, ndim=2]
        Branch labels map. Each branch is labeled with a unique positive integer. Background pixels are labeled 0.
    
    np.ndarray[INT32, ndim=2]
    """
    cdef:
        int H = skeleton.shape[0]
        int W = skeleton.shape[1]


    # Find hollow crosses
    for y in range(1:H-1):
        for x in range(1:W-1):
            if skeleton[y, x] == 0 
                and skeleton[y-1, x] == 1 
                and skeleton[y+1, x] == 1 
                and skeleton[y, x-1] == 1 
                and skeleton[y, x+1] == 1:
                
                # Remove hollow cross
                skeleton[y-1, x] -= 1
                skeleton[y+1, x] -= 1
                skeleton[y, x-1] -= 1
                skeleton[y, x+1] -= 1

                # Add center pixel to the skeleton
                skeleton[y, x] = 1


cdef bool fit_or_miss(int neighbors[8] neigbors, np.ndarray[UINT8, ndim=3] masks):
    for mask in masks:
        for n in range(7):
            if mask[n] == 1: # Should be > 0
                if neighbors[n] <= 0:
                    break
            if mask[n] == 2: # Should be == 0
                if neighbors[n] > 0:
                    break
        else:
            return True
    return False


# ---------------------------------------------------------
# Create 3 branch junctions
cdef cpplist[int[8]] JONCTIONS_3

# Y vertical
JONCTIONS_3.push_back([1, 0, 1,
                       0,    0,
                       2, 1, 2])
JONCTIONS_3.push_back([1, 0, 1,
                      2,    2,
                      0, 1, 0])
# Y Diagonal
JONCTIONS_3[2, :] = [0, 1, 0,
                     2,    1,
                     1, 2, 0]
JONCTIONS_3[3, :] = [2, 1, 0,
                     0,    1,
                     1, 0, 2]
# T Vertical
JONCTIONS_3[4, :] = [2, 0, 2,
                     1,    1,
                     0, 1, 0]
# T Diagonal
JONCTIONS_3[5, :] = [1, 2, 0,
                     0,    2,
                     1, 0, 1]

# Rotate the masks
for i in range(6, 24, 6):
    for j in range(6):
        JONCTIONS_3[i + j, 0] = JONCTIONS_3[j, 2]
        JONCTIONS_3[i + j, 1] = JONCTIONS_3[j, 3]
        JONCTIONS_3[i + j, 2] = JONCTIONS_3[j, 4]
        JONCTIONS_3[i + j, 3] = JONCTIONS_3[j, 5]
        JONCTIONS_3[i + j, 4] = JONCTIONS_3[j, 6]
        JONCTIONS_3[i + j, 5] = JONCTIONS_3[j, 7]
        JONCTIONS_3[i + j, 6] = JONCTIONS_3[j, 0]
        JONCTIONS_3[i + j, 7] = JONCTIONS_3[j, 1]
    

cdef int[8] get_neighbors(np.ndarray[UINT8, ndim=2, cast=True] skeleton, Point yx):
    cdef int neighbors[8]
    neighbors[0] = skeleton[yx.y-1, yx.x-1]
    neighbors[1] = skeleton[yx.y-1, yx.x]
    neighbors[2] = skeleton[yx.y-1, yx.x+1]
    neighbors[3] = skeleton[yx.y, yx.x-1]
    neighbors[4] = skeleton[yx.y, yx.x+1]
    neighbors[5] = skeleton[yx.y+1, yx.x-1]
    neighbors[6] = skeleton[yx.y+1, yx.x]
    neighbors[7] = skeleton[yx.y+1, yx.x+1]
    return neighbors

cdef int masks_4[3][3][3] = [
    [[1, 0, 1],
     [0, 1, 0],
     [1, 0, 1]],
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],
    [[2, 2, 2],
     [2, 1, 1],
     [2, 1, 1]],
]