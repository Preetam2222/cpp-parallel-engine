#include "pe/transpose.hpp"

#include <algorithm>

namespace pe {

void transpose_naive(const float* in, float* out, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) out[j * rows + i] = in[i * cols + j];
}

namespace {
void transpose_rows(const float* in, float* out, size_t rows, size_t cols, size_t tile, size_t i0, size_t i1) {
    for (size_t ii = i0; ii < i1; ii += tile) {
        const size_t ie = std::min(ii + tile, i1);
        for (size_t jj = 0; jj < cols; jj += tile) {
            const size_t je = std::min(jj + tile, cols);
            for (size_t i = ii; i < ie; ++i)
                for (size_t j = jj; j < je; ++j) out[j * rows + i] = in[i * cols + j];
        }
    }
}
}  // namespace

void transpose_blocked(const float* in, float* out, size_t rows, size_t cols, size_t tile) {
    if (tile == 0) tile = 32;
    transpose_rows(in, out, rows, cols, tile, 0, rows);
}

void transpose_parallel(ThreadPool& pool, const float* in, float* out, size_t rows, size_t cols, size_t tile) {
    if (tile == 0) tile = 32;
    const size_t tile_rows = (rows + tile - 1) / tile;
    // Each task owns a band of `tile` input rows => disjoint output columns, no races.
    pool.parallel_for(tile_rows, 1, [&](size_t b, size_t e) {
        transpose_rows(in, out, rows, cols, tile, b * tile, std::min(e * tile, rows));
    });
}

}  // namespace pe
