#include <stdint.h>

#include "common.h"

static const PointWithID NEIGHBORHOOD [8] =
  {
    {-1, -1, 0}, {-1, 0, 1}, {-1, 1, 2},
    {0, 1, 3},
    {1, 1, 4}, {1, 0, 5}, {1, -1, 6},
    {0, -1, 7}
  };

int opposite_neighbor_id(int n) {
  return (n + 4) % 8;
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