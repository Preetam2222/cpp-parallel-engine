#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "pe/kernels.hpp"

using namespace pe;

namespace {

// Independent reference: same per-record float math, long double accumulation.
struct Ref {
    long double sum = 0;
    uint64_t count = 0;
};
Ref reference(const std::vector<RecordAoS>& a, float thr) {
    Ref r;
    for (const auto& rec : a) {
        float s = std::sqrt(rec.x * rec.x + rec.y * rec.y + rec.z * rec.z) * rec.w;
        if (s > thr) {
            r.sum += s;
            ++r.count;
        }
    }
    return r;
}

void expect_matches(const Stat& got, const Ref& ref, const std::string& what) {
    EXPECT_EQ(got.count, ref.count) << what;
    // All kept scores are positive, so there is no cancellation: relative error stays tiny.
    EXPECT_NEAR((double)got.sum, (double)ref.sum, 1e-4 * (double)ref.sum + 1e-12) << what;
}

struct Case {
    std::string name;
    std::function<Stat(const std::vector<RecordAoS>&, const SoA&, float)> run;
};

std::vector<Case> all_cases() {
    return {
        {"aos_scalar", [](auto& a, auto&, float t) { return score_aos_scalar(a.data(), a.size(), t); }},
        {"aos_default", [](auto& a, auto&, float t) { return score_aos_default(a.data(), a.size(), t); }},
        {"soa_scalar", [](auto&, auto& s, float t) { return score_soa_scalar(s.view(), t); }},
        {"soa_unrolled", [](auto&, auto& s, float t) { return score_soa_unrolled(s.view(), t); }},
        {"soa_autovec", [](auto&, auto& s, float t) { return score_soa_autovec(s.view(), t); }},
        {"soa_simd", [](auto&, auto& s, float t) { return score_soa_simd(s.view(), t); }},
    };
}

}  // namespace

TEST(Kernels, AllVariantsMatchReferenceOnEdgeSizes) {
    for (size_t n : {0, 1, 3, 7, 8, 9, 15, 16, 17, 255, 511, 512, 513, 1000, 4097, 100003}) {
        std::vector<RecordAoS> aos;
        SoA soa;
        make_dataset(n, 1000 + n, aos, soa);
        Ref ref = reference(aos, 0.25f);
        for (auto& c : all_cases()) expect_matches(c.run(aos, soa, 0.25f), ref, c.name + " n=" + std::to_string(n));
    }
}

TEST(Kernels, ThresholdExtremes) {
    std::vector<RecordAoS> aos;
    SoA soa;
    make_dataset(5000, 7, aos, soa);
    for (float thr : {-1.0f, 0.0f, 0.5f, 100.0f}) {
        Ref ref = reference(aos, thr);
        for (auto& c : all_cases()) expect_matches(c.run(aos, soa, thr), ref, c.name + " thr=" + std::to_string(thr));
    }
    EXPECT_EQ(score_soa_simd(soa.view(), -1.0f).count, 5000u);  // score >= 0 > -1: everything kept
    EXPECT_EQ(score_soa_simd(soa.view(), 100.0f).count, 0u);    // max score < sqrt(3): nothing kept
}

TEST(Kernels, UnalignedSlicesWork) {
    std::vector<RecordAoS> aos;
    SoA soa;
    make_dataset(2003, 5, aos, soa);
    for (size_t off : {1u, 2u, 3u, 5u, 7u}) {
        std::vector<RecordAoS> sub(aos.begin() + off, aos.end());
        Ref ref = reference(sub, 0.25f);
        SoAView v = soa.view().slice(off, soa.n);  // pointers no longer 32-byte aligned
        expect_matches(score_soa_simd(v, 0.25f), ref, "simd off=" + std::to_string(off));
        expect_matches(score_soa_autovec(v, 0.25f), ref, "autovec off=" + std::to_string(off));
        expect_matches(score_soa_unrolled(v, 0.25f), ref, "unrolled off=" + std::to_string(off));
    }
}

TEST(Kernels, ParallelMatchesReferenceForEveryThreadCount) {
    std::vector<RecordAoS> aos;
    SoA soa;
    make_dataset(300007, 11, aos, soa);
    Ref ref = reference(aos, 0.25f);
    for (size_t threads : {1u, 2u, 3u, 4u, 8u}) {
        ThreadPool pool(threads);
        expect_matches(score_soa_parallel(pool, soa.view(), 0.25f), ref, "threads=" + std::to_string(threads));
    }
}

TEST(Kernels, ParallelIsDeterministicAndHandlesTinyInputs) {
    std::vector<RecordAoS> aos;
    SoA soa;
    make_dataset(123457, 3, aos, soa);
    ThreadPool pool(4);
    Stat a = score_soa_parallel(pool, soa.view(), 0.25f);
    for (int i = 0; i < 20; ++i) {
        Stat b = score_soa_parallel(pool, soa.view(), 0.25f);
        EXPECT_EQ(a.count, b.count);
        EXPECT_EQ(a.sum, b.sum);  // bit-identical: chunk results are combined in fixed order
    }
    for (size_t n : {0u, 1u, 5u}) {
        std::vector<RecordAoS> a2;
        SoA s2;
        make_dataset(n, 9, a2, s2);
        expect_matches(score_soa_parallel(pool, s2.view(), 0.25f), reference(a2, 0.25f), "tiny n=" + std::to_string(n));
    }
}

TEST(Kernels, DatasetLayoutsHoldTheSameValues) {
    std::vector<RecordAoS> aos;
    SoA soa;
    make_dataset(1000, 42, aos, soa);
    ASSERT_EQ(aos.size(), 1000u);
    ASSERT_EQ(soa.n, 1000u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(soa.x.data()) % 64, 0u);
    for (size_t i = 0; i < 1000; ++i) {
        ASSERT_EQ(aos[i].x, soa.x[i]);
        ASSERT_EQ(aos[i].y, soa.y[i]);
        ASSERT_EQ(aos[i].z, soa.z[i]);
        ASSERT_EQ(aos[i].w, soa.w[i]);
        ASSERT_GE(aos[i].w, 0.0f);
        ASSERT_LT(aos[i].w, 1.0f);
    }
}
