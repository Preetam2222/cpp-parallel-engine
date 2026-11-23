#include <gtest/gtest.h>

#include <vector>

#include "pe/transpose.hpp"

using namespace pe;

namespace {
std::vector<float> iota_matrix(size_t rows, size_t cols) {
    std::vector<float> m(rows * cols);
    for (size_t i = 0; i < m.size(); ++i) m[i] = (float)i;
    return m;
}
}  // namespace

TEST(Transpose, NaiveMatchesDefinition) {
    auto in = iota_matrix(3, 5);
    std::vector<float> out(15);
    transpose_naive(in.data(), out.data(), 3, 5);  // out is 5 x 3
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 5; ++j) EXPECT_EQ(out[j * 3 + i], in[i * 5 + j]);
}

TEST(Transpose, BlockedAndParallelMatchNaiveOnRaggedShapes) {
    ThreadPool pool(3);
    const size_t shapes[][2] = {{1, 1}, {1, 7}, {7, 1}, {33, 65}, {100, 37}, {64, 64}, {129, 31}};
    for (auto& sh : shapes) {
        size_t r = sh[0], c = sh[1];
        auto in = iota_matrix(r, c);
        std::vector<float> ref(r * c);
        transpose_naive(in.data(), ref.data(), r, c);
        for (size_t tile : {1u, 8u, 32u, 100u}) {
            std::vector<float> a(r * c, -1.0f), b(r * c, -1.0f);
            transpose_blocked(in.data(), a.data(), r, c, tile);
            transpose_parallel(pool, in.data(), b.data(), r, c, tile);
            EXPECT_EQ(a, ref) << r << "x" << c << " tile=" << tile;
            EXPECT_EQ(b, ref) << r << "x" << c << " tile=" << tile << " (parallel)";
        }
    }
}

TEST(Transpose, TransposingTwiceIsIdentity) {
    const size_t r = 77, c = 130;
    auto in = iota_matrix(r, c);
    std::vector<float> t(r * c), back(r * c);
    transpose_blocked(in.data(), t.data(), r, c);
    transpose_blocked(t.data(), back.data(), c, r);
    EXPECT_EQ(back, in);
}

TEST(Transpose, ZeroTileFallsBackToDefault) {
    auto in = iota_matrix(40, 40);
    std::vector<float> a(1600), ref(1600);
    transpose_naive(in.data(), ref.data(), 40, 40);
    transpose_blocked(in.data(), a.data(), 40, 40, 0);
    EXPECT_EQ(a, ref);
}
