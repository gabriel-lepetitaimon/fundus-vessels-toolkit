#include "vector_fields.h"

torch::Tensor inverse_displacement(const torch::Tensor& disp_field, const torch::Tensor& points, const int max_iters,
                                   const float sqr_tol) {
    /*
    Inverse vector displacement at given sample coordinates using fixed-point iteration.

    Parameters:
    - disp_field: A tensor of shape (H, W, 2) representing the displacement field.
    - points: A tensor of shape (N, 2) containing the (y, x) coordinates where inversion is to be performed.

    Returns:
    - A tensor of shape (N, 2) containing the inverted displacement vectors at the sample coordinates.
    */

    // Get dimensions
    TORCH_CHECK_VALUE(disp_field.dim() == 3, "disp_field must be a 3D tensor");
    TORCH_CHECK_VALUE(disp_field.size(2) == 2, "disp_field must have 2 channels in the last dimension");
    const int H = disp_field.size(0);
    const int W = disp_field.size(1);
    auto disp_acc = disp_field.accessor<float, 3>();

    TORCH_CHECK_VALUE(points.dim() == 2, "points must be a 2D tensor");
    TORCH_CHECK_VALUE(points.size(1) == 2, "points must have 2 columns (y, x)");
    const int64_t N = points.size(0);
    auto coords_acc = points.accessor<double, 2>();

    // Prepare output tensor
    torch::Tensor output = torch::empty({N, 2}, disp_field.options());
    auto output_acc = output.accessor<float, 2>();

    // Fixed-point iteration for inversion
    // #pragma omp parallel for
    Point last_p, last_inv_disp_p;
    for (int64_t n = 0; n < N; ++n) {
        Point p{coords_acc[n][0], coords_acc[n][1]};

        const Point& diff_p = p - last_p;
        Point inv_disp_p = diff_p.squaredNorm() <= 3 ? last_inv_disp_p + diff_p
                                                     : _vec_bilinear_interpolate(disp_acc, p, IntPoint{H, W});

        for (int iter = 0; iter < max_iters; ++iter) {
            Point disp_o = _vec_bilinear_interpolate(disp_acc, p + inv_disp_p, IntPoint{H, W});
            if ((disp_o + inv_disp_p).squaredNorm() < sqr_tol) break;
            inv_disp_p = -disp_o;
        }

        output_acc[n][0] = inv_disp_p.y;
        output_acc[n][1] = inv_disp_p.x;
    }
    return output;
}

torch::Tensor vec_bilinear_interpolate(const torch::Tensor& vector_field, const torch::Tensor& points) {
    /*
    Bilinear interpolation of a vector field at given sample coordinates.

    Parameters:
    - vector_field: A tensor of shape (H, W, 2) representing the vector field.
    - points: A tensor of shape (N, 2) containing the (y, x) coordinates where interpolation is to be performed.

    Returns:
    - A tensor of shape (N, 2) containing the interpolated vectors at the sample coordinates.
    */

    // Get dimensions
    TORCH_CHECK_VALUE(vector_field.dim() == 3, "vector_field must be a 3D tensor");
    TORCH_CHECK_VALUE(vector_field.size(2) == 2, "vector_field must have 2 channels in the last dimension");
    const int H = vector_field.size(0);
    const int W = vector_field.size(1);
    auto vec_acc = vector_field.accessor<float, 3>();

    TORCH_CHECK_VALUE(points.dim() == 2, "points must be a 2D tensor");
    TORCH_CHECK_VALUE(points.size(1) == 2, "points must have 2 columns (y, x)");
    const int64_t N = points.size(0);
    auto coords_acc = points.accessor<float, 2>();

    // Prepare output tensor
    torch::Tensor output = torch::empty({N, 2}, vector_field.options());
    auto output_acc = output.accessor<float, 2>();

// Perform bilinear interpolation
#pragma omp parallel for
    for (int64_t n = 0; n < N; ++n) {
        const auto& p = _vec_bilinear_interpolate(vec_acc, Point{coords_acc[n][0], coords_acc[n][1]}, IntPoint{H, W});
        output_acc[n][0] = p.y;
        output_acc[n][1] = p.x;
    }

    return output;
}

Point _vec_bilinear_interpolate(const Tensor3DAcc<float>& vec_acc, const Point& p, const IntPoint& field_size) {
    const auto &H = field_size.y, &W = field_size.x;
    int64_t y0 = static_cast<int64_t>(std::floor(p.y));
    int64_t x0 = static_cast<int64_t>(std::floor(p.x));
    if (y0 < 0) y0 = 0;
    if (x0 < 0) x0 = 0;
    if (y0 >= H - 1) y0 = H - 2;
    if (x0 >= W - 1) x0 = W - 2;
    int64_t y1 = y0 + 1, x1 = x0 + 1;
    const float dy1 = p.y - y0, dy0 = 1 - dy1, dx1 = p.x - x0, dx0 = 1 - dx1;

    std::array<float, 2> out;

    for (int64_t c = 0; c < 2; ++c) {
        const float v0 = vec_acc[y0][x0][c] * dy0 + vec_acc[y1][x0][c] * dy1;
        const float v1 = vec_acc[y0][x1][c] * dy0 + vec_acc[y1][x1][c] * dy1;
        out[c] = v0 * dx0 + v1 * dx1;
    }
    return Point{out[0], out[1]};
}

torch::Tensor smooth_binary_mask(const torch::Tensor& mask, float sigma, const float tol) {
    // Ensure the input is a binary mask
    TORCH_CHECK_VALUE(mask.dim() == 2, "mask must be a 2D tensor");
    TORCH_CHECK_VALUE(mask.dtype() == torch::kBool, "mask must be of type uint8 or bool");

    // Create a Gaussian kernel
    std::list<std::pair<IntPoint, float>> gaussian_kernel;
    sigma = 2.0 * sigma * sigma;  // Precompute for efficiency
    float sum = 0.0f;
    int max_x = 0;
    for (int y = 1; y <= 3 * sigma; ++y) {
        for (int x = 1; x <= 3 * sigma; ++x) {
            float value = std::exp(-(x * x + y * y) / sigma);
            if (value < tol) continue;  // Skip negligible values
            gaussian_kernel.emplace_back(IntPoint{y, x}, value);
            sum += (y == 0 && x == 0) ? value : 2 * value;  // Account for symmetry
            if (x > max_x) max_x = x;
        }
    }

    // Normalize the kernel
    float inv_sum = 1.0f / (sum * 4 + 1);  // Include the center point
    for (auto& [point, value] : gaussian_kernel) value *= inv_sum;

    // Apply the Gaussian kernel to the binary mask
    torch::Tensor smoothed_mask = torch::zeros({mask.size(0) + 2 * max_x, mask.size(1) + 2 * max_x}, torch::kFloat32);
    auto mask_acc = mask.accessor<bool, 2>();
    auto smoothed_acc = smoothed_mask.accessor<float, 2>();

    for (int64_t y = 0; y < mask.size(0); ++y) {
        for (int64_t x = 0; x < mask.size(1); ++x) {
            if (!mask_acc[y][x]) continue;
            int yy = static_cast<int>(y) + max_x;
            int xx = static_cast<int>(x) + max_x;
            smoothed_acc[yy][xx] += inv_sum;  // Add the center point contribution
            for (const auto& [offset, value] : gaussian_kernel) {
                smoothed_acc[yy + offset.y][xx + offset.x] += value;
                smoothed_acc[yy - offset.y][xx + offset.x] += value;
                smoothed_acc[yy + offset.y][xx - offset.x] += value;
                smoothed_acc[yy - offset.y][xx - offset.x] += value;
            }
        }
    }

    return smoothed_mask.squeeze();
}