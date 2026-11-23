// Scoring pipeline: for each record compute s = sqrt(x^2+y^2+z^2) * w, keep it if s > thr,
// and return the sum and count of kept scores. Implemented several ways so the effect of each
// optimisation (layout, unrolling, SIMD, threads) can be measured separately.
//
// All variants compute each per-record score with the same float operations in the same
// order (built with -ffp-contract=off), so they agree exactly on `count`. `sum` differs only
// by float rounding from summation order.
#pragma once
#include <cstddef>
#include <cstdint>

#include "pe/records.hpp"
#include "pe/thread_pool.hpp"

namespace pe {

struct Stat {
    double sum = 0.0;
    uint64_t count = 0;
};

// Baselines on AoS. `scalar` is compiled with auto-vectorisation disabled (models naive
// code); `default` is whatever the compiler does by itself at -O3.
Stat score_aos_scalar(const RecordAoS* r, size_t n, float thr);
Stat score_aos_default(const RecordAoS* r, size_t n, float thr);

// SoA variants.
Stat score_soa_scalar(const SoAView& v, float thr);    // layout change only
Stat score_soa_unrolled(const SoAView& v, float thr);  // + 4x unroll, independent accumulators
Stat score_soa_autovec(const SoAView& v, float thr);   // portable lane-array form; compiler vectorises
Stat score_soa_simd(const SoAView& v, float thr);      // AVX2 intrinsics if available, else autovec
Stat score_soa_parallel(ThreadPool& pool, const SoAView& v, float thr);  // simd + threads, deterministic

bool has_avx2();

}  // namespace pe
