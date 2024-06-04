#ifndef NEIGHBORS_H
#define NEIGHBORS_H

#include <stdint.h>

#include "common.h"


/**
 * @brief Structure representing neighborhood coordinates: (y, x) and the id of the neighbor.
 * 
 * The id of the neighbor is in the following order:
 *    0 1 2
 *    7 . 3
 *    6 5 4
 * 
*/
static const std::array<PointWithID, 8> NEIGHBORHOOD =
  {{
    {-1, -1, 0}, {-1, 0, 1}, {-1, 1, 2},
    {0, 1, 3},
    {1, 1, 4}, {1, 0, 5}, {1, -1, 6},
    {0, -1, 7}
  }};

/**
 * @brief Structure representing neighborhood coordinates: (y, x) and the id of the neighbor. The close neighbors (horizontal and vertical) are listed first.
 * 
 * The neighbors are listed in the following order:
 *    0 4 1
 *    7 . 5
 *    3 6 2
 * 
*/
static const std::array<PointWithID, 8> CLOSE_NEIGHBORHOOD =
{{
  {-1, -1, 0}, {-1, 1, 2}, {1, 1, 4}, {1, -1, 6},
  {-1,  0, 1}, { 0, 1, 3}, {1, 0, 5}, {0, -1, 7}
}};


/**
 * @brief Array containing the possible next neighbors id given the previous neighbor id in a tracking process.
 * 
 * When tracking a branch the neighbor pixels (x) adjacent to the previous pixel (p) are by construction not part of the branch. The most likely next pixels are the ones in the opposite direction of the previous pixel (p).
 * (The fourth and fifth neighbors should theoretically never be the next pixel if the skeleton is valid.)
 *      x 4 1
 *  ->  p c 0   (c is the current pixel and p is the previous pixel,
 *      x 5 2    in this example the neighbor id of c relative to p is 3.)
*/ 
static const std::array<std::array<int, 5>, 8> TRACK_NEXT_NEIGHBORS = {{
  {0, 1, 7, 2, 6}, // neighbor index of c relative to p is 0 (previous pixel is bottom-right)
  {1, 2, 0, 3, 7}, // neighbor index of c relative to p is 1 (previous pixel is bottom)
  {2, 3, 1, 4, 0}, // neighbor index of c relative to p is 2 (previous pixel is bottom-left)
  {3, 4, 2, 5, 1}, // neighbor index of c relative to p is 3 (previous pixel is left)
  {4, 5, 3, 6, 2}, // neighbor index of c relative to p is 4 (previous pixel is top-left)
  {5, 6, 4, 7, 3}, // neighbor index of c relative to p is 5 (previous pixel is top)
  {6, 7, 5, 0, 4}, // neighbor index of c relative to p is 6 (previous pixel is top-right)
  {7, 0, 6, 1, 5}  // neighbor index of c relative to p is 7 (previous pixel is right)
}};

/**
 * @brief Array containing the possible next neighbors id given the previous neighbor id in a tracking process.
 * 
 * When tracking a branch the neighbor pixels (x) adjacent to the previous pixel (p) are by construction not part of the branch. The closest pixels (vertical and horizontal) are listed first
 * (The fourth and fifth neighbors should theoretically never be the next pixel if the skeleton is valid.)
 *      x 1 3
 *  ->  p c 0   (c is the current pixel and p is the previous pixel,
 *      x 2 4    in this example the neighbor id of c relative to p is 3.)
 * 
 *      p x 4
 *  ->  x c 1   (c is the current pixel and p is the previous pixel,
 *      5 2 3    in this example the neighbor id of c relative to p is 4.)
*/ 
static const std::array<std::array<int, 5>, 8> TRACK_CLOSE_NEIGHBORS = {{
  {7, 1, 0, 6, 2}, // neighbor index of c relative to p is 0 (previous pixel is bottom-right)
  {1, 7, 3, 0, 2}, // neighbor index of c relative to p is 1 (previous pixel is bottom)
  {1, 3, 2, 0, 4}, // neighbor index of c relative to p is 2 (previous pixel is bottom-left)
  {3, 1, 5, 2, 4}, // neighbor index of c relative to p is 3 (previous pixel is left)
  {3, 5, 4, 2, 6}, // neighbor index of c relative to p is 4 (previous pixel is top-left)
  {5, 3, 7, 4, 6}, // neighbor index of c relative to p is 5 (previous pixel is top)
  {5, 7, 6, 4, 0}, // neighbor index of c relative to p is 6 (previous pixel is top-right)
  {7, 5, 1, 6, 0}  // neighbor index of c relative to p is 7 (previous pixel is right)
}};

inline int opposite_neighbor_id(int n) {
  return n<4 ? n+4 : n-4;
}

inline int next_neighbor_id(int n) {
  return n<7 ? n+1 : 0;
}

inline int prev_neighbor_id(int n) {
  return n>0 ? n-1 : 7;
}

uint8_t count_neighbors (uint8_t neighborhood) {
  static const uint8_t NIBBLE_LOOKUP [16] =
  {
    0, 1, 1, 2, 1, 2, 2, 3, 
    1, 2, 2, 3, 2, 3, 3, 4
  };

  return NIBBLE_LOOKUP[neighborhood & 0x0F] + NIBBLE_LOOKUP[neighborhood >> 4];
}

uint8_t roll_neighbors(uint8_t neighborhood, uint8_t n) {
  return (neighborhood<<n) | (neighborhood>>(8-n));
}

bool hit(uint8_t neighborhood, uint8_t positive_mask) {
    return (neighborhood & positive_mask) == positive_mask;
}

bool miss(uint8_t neighborhood, uint8_t negative_mask) {
    return (neighborhood & negative_mask) == 0;
}

bool hit_and_miss(uint8_t neighborhood, uint8_t positive_mask, uint8_t negative_mask) {
  return (neighborhood & positive_mask) == positive_mask && (neighborhood & negative_mask) == 0;
}



/**
 * @brief Read the value of the 8 neighbors of a pixel in a binary image.
 * 
 * The neighbors are read in the following order:
 *  1 2 3    0x01 0x02 0x04
 *  8 . 4 -> 0x80  .   0x08
 *  7 6 5    0x40 0x20 0x10
 * 
 * Assume that z contains a pixel at the position (y, x). 
 * The neighbors outside the image are considered to be 0.
 * 
 * @param z Binary image.
 * @param y Row of the pixel.
 * @param x Column of the pixel.
 * @return unsigned short Value of the 8 neighbors of the pixel. The top-left neighbor is the most significant bit.
*/
template <typename T>
uint8_t get_neighborhood_safe(const Tensor2DAccessor<T>& z, int y, int x) {
    uint8_t neighbors = 0;
    if (y > 0){
        if (x > 0 && z[y - 1][x - 1] > 0)
            neighbors |= 0b10000000;
        if (z[y - 1][x] > 0)
            neighbors |= 0b01000000;
        if (x < z.size(1) - 1 && z[y - 1][x + 1] > 0)
            neighbors |= 0b00100000;
    }

    if (x < z.size(1) - 1 && z[y][x + 1] > 0)
        neighbors |= 0b00010000;

    if (y < z.size(0) - 1){
        if (x < z.size(1) - 1 && z[y + 1][x + 1] > 0)
            neighbors |= 0b00010000;
        if (z[y + 1][x] > 0)
            neighbors |= 0b00001000;
        if (x > 0 && z[y + 1][x - 1] > 0)
            neighbors |= 0b00000010;
    }

    if (x > 0 && z[y][x - 1] > 0)
        neighbors |= 0b00000001;
    return neighbors;
}


/**
 * @brief Read the value of the 8 neighbors of a pixel in a binary image.
 * 
 * The neighbors are read in the following order:
 *  1 2 3    0x01 0x02 0x04
 *  8 . 4 -> 0x80  .   0x08
 *  7 6 5    0x40 0x20 0x10
 * 
 * Assume that z contains a pixel at the position (y, x) and that the pixel is not at the border of the image.
 * 
 * @param z Binary image.
 * @param y Row of the pixel.
 * @param x Column of the pixel.
 * @return unsigned short Value of the 8 neighbors of the pixel. The top-left neighbor is the most significant bit.
*/
template <typename T>
uint8_t get_neighborhood(const Tensor2DAccessor<T>& z, int y, int x) {
    uint8_t neighbors = 0;
    neighbors |= z[y - 1][x - 1] > 0    ? 0b10000000 : 0;
    neighbors |= z[y - 1][x] > 0        ? 0b01000000 : 0;
    neighbors |= z[y - 1][x + 1] > 0    ? 0b00100000 : 0;
    neighbors |= z[y][x + 1] > 0        ? 0b00010000 : 0;
    neighbors |= z[y + 1][x + 1] > 0    ? 0b00001000 : 0;
    neighbors |= z[y + 1][x] > 0        ? 0b00000100 : 0;
    neighbors |= z[y + 1][x - 1] > 0    ? 0b00000010 : 0;
    neighbors |= z[y][x - 1] > 0        ? 0b00000001 : 0;
    return neighbors;
}

#endif // NEIGHBORS_H