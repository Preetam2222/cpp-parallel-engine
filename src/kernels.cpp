#include "pe/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define PE_X86 1
#include <immintrin.h>
#endif

// Used to model "naive scalar code": switch off auto-vectorisation for one function / loop.
#if defined(__clang__)
#define PE_NOVEC_FN
#define PE_NOVEC _Pragma("clang loop vectorize(disable) interleave(disable)")
#elif defined(__GNUC__)
#define PE_NOVEC_FN __attribute__((optimize("no-tree-vectorize")))
#define PE_NOVEC
#else
#define PE_NOVEC_FN
#define PE_NOVEC
#endif

namespace pe {
namespace {

// Flush float partial sums into a double every kBlock records: keeps rounding error ~ kBlock*eps
// instead of n*eps, while the inner loop stays pure float (SIMD friendly). Multiple of 8.
constexpr size_t kBlock = 512;

// The one definition of the per-record score. Evaluation order is fixed: ((x*x + y*y) + z*z).
inline float score(float x, float y, float z, float w) { return std::sqrt(x * x + y * y + z * z) * w; }

inline Stat tail_scalar(const SoAView& v, size_t i, float thr, Stat s) {
    for (; i < v.n; ++i) {
        float sc = score(v.x[i], v.y[i], v.z[i], v.w[i]);
        if (sc > thr) {
            s.sum += sc;
            ++s.count;
        }
    }
    return s;
}

}  // namespace

PE_NOVEC_FN Stat score_aos_scalar(const RecordAoS* r, size_t n, float thr) {
    Stat s;
    PE_NOVEC
    for (size_t i = 0; i < n; ++i) {
        float sc = score(r[i].x, r[i].y, r[i].z, r[i].w);
        if (sc > thr) {
            s.sum += sc;
            ++s.count;
        }
    }
    return s;
}

Stat score_aos_default(const RecordAoS* r, size_t n, float thr) {
    Stat s;
    for (size_t i = 0; i < n; ++i) {
        float sc = score(r[i].x, r[i].y, r[i].z, r[i].w);
        if (sc > thr) {
            s.sum += sc;
            ++s.count;
        }
    }
    return s;
}

PE_NOVEC_FN Stat score_soa_scalar(const SoAView& v, float thr) {
    Stat s;
    PE_NOVEC
    for (size_t i = 0; i < v.n; ++i) {
        float sc = score(v.x[i], v.y[i], v.z[i], v.w[i]);
        if (sc > thr) {
            s.sum += sc;
            ++s.count;
        }
    }
    return s;
}

PE_NOVEC_FN Stat score_soa_unrolled(const SoAView& v, float thr) {
    Stat r;
    const size_t n4 = v.n & ~size_t(3);
    size_t i = 0;
    while (i < n4) {
        const size_t end = std::min(n4, i + kBlock);
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;  // independent chains hide FP add latency
        uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        PE_NOVEC
        for (; i < end; i += 4) {
            float s0 = score(v.x[i], v.y[i], v.z[i], v.w[i]);
            float s1 = score(v.x[i + 1], v.y[i + 1], v.z[i + 1], v.w[i + 1]);
            float s2 = score(v.x[i + 2], v.y[i + 2], v.z[i + 2], v.w[i + 2]);
            float s3 = score(v.x[i + 3], v.y[i + 3], v.z[i + 3], v.w[i + 3]);
            if (s0 > thr) { a0 += s0; ++c0; }
            if (s1 > thr) { a1 += s1; ++c1; }
            if (s2 > thr) { a2 += s2; ++c2; }
            if (s3 > thr) { a3 += s3; ++c3; }
        }
        r.sum += (double)a0 + a1 + a2 + a3;
        r.count += (uint64_t)c0 + c1 + c2 + c3;
    }
    return tail_scalar(v, i, thr, r);
}

// Portable vector path (x86 SSE/AVX, ARM NEON, ...): `omp simd` (enabled by -fopenmp-simd, no
// OpenMP runtime needed) tells the compiler it may reassociate this one float reduction, which
// is what lets it use vector registers. Branch-free: keep is 0 or 1.
Stat score_soa_autovec(const SoAView& v, float thr) {
    Stat r;
    size_t i = 0;
    while (i < v.n) {
        const size_t end = std::min(v.n, i + kBlock);
        float acc = 0.0f;
        uint32_t cnt = 0;
#pragma omp simd reduction(+ : acc, cnt)
        for (size_t k = i; k < end; ++k) {
            float sc = score(v.x[k], v.y[k], v.z[k], v.w[k]);
            const uint32_t keep = (uint32_t)(sc > thr);
            acc += sc * (float)keep;  // exact: sc is finite, so sc*0 = 0 and sc*1 = sc
            cnt += keep;
        }
        r.sum += acc;
        r.count += cnt;
        i = end;
    }
    return r;
}

#ifdef PE_X86
__attribute__((target("avx2"))) static Stat score_soa_avx2(const SoAView& v, float thr) {
    Stat r;
    const size_t n8 = v.n & ~size_t(7);
    const __m256 vthr = _mm256_set1_ps(thr);
    size_t i = 0;
    while (i < n8) {
        const size_t end = std::min(n8, i + kBlock);
        __m256 acc = _mm256_setzero_ps();
        __m256i cnt = _mm256_setzero_si256();
        for (; i < end; i += 8) {
            __m256 x = _mm256_loadu_ps(v.x + i), y = _mm256_loadu_ps(v.y + i);
            __m256 z = _mm256_loadu_ps(v.z + i), w = _mm256_loadu_ps(v.w + i);
            __m256 sq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)),
                                      _mm256_mul_ps(z, z));
            __m256 sc = _mm256_mul_ps(_mm256_sqrt_ps(sq), w);
            __m256 m = _mm256_cmp_ps(sc, vthr, _CMP_GT_OQ);  // all-ones where sc > thr
            acc = _mm256_add_ps(acc, _mm256_and_ps(sc, m));
            cnt = _mm256_sub_epi32(cnt, _mm256_castps_si256(m));  // -(-1) = +1
        }
        alignas(32) float a[8];
        alignas(32) int32_t c[8];
        _mm256_store_ps(a, acc);
        _mm256_store_si256(reinterpret_cast<__m256i*>(c), cnt);
        for (int j = 0; j < 8; ++j) {
            r.sum += a[j];
            r.count += (uint32_t)c[j];
        }
    }
    return tail_scalar(v, i, thr, r);
}
#endif

bool has_avx2() {
#ifdef PE_X86
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

Stat score_soa_simd(const SoAView& v, float thr) {
#ifdef PE_X86
    static const bool avx2 = has_avx2();
    if (avx2) return score_soa_avx2(v, thr);
#endif
    return score_soa_autovec(v, thr);
}

Stat score_soa_parallel(ThreadPool& pool, const SoAView& v, float thr) {
    if (v.n == 0) return {};
    // ~4 chunks per thread for load balance, multiple of 8 records, at least one block each.
    size_t grain = std::max<size_t>(kBlock, v.n / (pool.size() * 4));
    grain = (grain + 7) & ~size_t(7);
    const size_t chunks = (v.n + grain - 1) / grain;
    std::vector<Stat> parts(chunks);  // one slot per chunk; summed in order => deterministic
    pool.parallel_for(v.n, grain, [&](size_t b, size_t e) { parts[b / grain] = score_soa_simd(v.slice(b, e), thr); });
    Stat total;
    for (const Stat& p : parts) {
        total.sum += p.sum;
        total.count += p.count;
    }
    return total;
}

}  // namespace pe
