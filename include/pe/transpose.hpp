// Matrix transpose: out (cols x rows, row-major) = transpose of in (rows x cols, row-major).
// The naive loop reads rows but writes columns with stride `rows` floats, which misses cache on
// nearly every write for large matrices. The blocked version works on tile x tile sub-matrices
// that fit in L1 -- the same idea as the shared-memory tile in a CUDA transpose/matmul kernel.
#pragma once
#include <cstddef>

#include "pe/thread_pool.hpp"

namespace pe {

void transpose_naive(const float* in, float* out, size_t rows, size_t cols);
void transpose_blocked(const float* in, float* out, size_t rows, size_t cols, size_t tile = 32);
void transpose_parallel(ThreadPool& pool, const float* in, float* out, size_t rows, size_t cols,
                        size_t tile = 32);

}  // namespace pe
