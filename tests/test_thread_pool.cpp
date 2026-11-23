#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#include "pe/thread_pool.hpp"

using namespace pe;

TEST(ThreadPool, EveryIndexVisitedExactlyOnce) {
    for (size_t threads : {1u, 2u, 3u, 8u}) {
        ThreadPool pool(threads);
        EXPECT_EQ(pool.size(), threads);
        for (size_t n : {1u, 2u, 7u, 100u, 1000u, 12345u}) {
            for (size_t grain : {1u, 3u, 64u, 5000u, 100000u}) {
                std::vector<std::atomic<int>> hits(n);
                pool.parallel_for(n, grain, [&](size_t b, size_t e) {
                    ASSERT_LE(b, e);
                    ASSERT_LE(e, n);
                    ASSERT_LE(e - b, grain);
                    for (size_t i = b; i < e; ++i) hits[i].fetch_add(1);
                });
                for (size_t i = 0; i < n; ++i) ASSERT_EQ(hits[i].load(), 1) << "n=" << n << " grain=" << grain << " i=" << i;
            }
        }
    }
}

TEST(ThreadPool, EmptyRangeAndZeroGrain) {
    ThreadPool pool(3);
    bool called = false;
    pool.parallel_for(0, 10, [&](size_t, size_t) { called = true; });
    EXPECT_FALSE(called);
    std::atomic<size_t> total{0};
    pool.parallel_for(10, 0, [&](size_t b, size_t e) { total += e - b; });  // grain 0 treated as 1
    EXPECT_EQ(total.load(), 10u);
}

TEST(ThreadPool, ActuallyUsesMultipleThreads) {
    if (std::thread::hardware_concurrency() < 2) GTEST_SKIP() << "single core";
    ThreadPool pool(4);
    std::mutex m;
    std::vector<std::thread::id> ids;
    pool.parallel_for(64, 1, [&](size_t, size_t) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // let other threads grab chunks
        std::lock_guard<std::mutex> lk(m);
        ids.push_back(std::this_thread::get_id());
    });
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    EXPECT_GE(ids.size(), 2u);
}

TEST(ThreadPool, ExceptionIsRethrownAndPoolStaysUsable) {
    ThreadPool pool(4);
    EXPECT_THROW(pool.parallel_for(1000, 10,
                                   [&](size_t b, size_t) {
                                       if (b == 500) throw std::runtime_error("boom");
                                   }),
                 std::runtime_error);
    std::atomic<size_t> total{0};  // still works afterwards
    pool.parallel_for(1000, 10, [&](size_t b, size_t e) { total += e - b; });
    EXPECT_EQ(total.load(), 1000u);
}

TEST(ThreadPool, ManySuccessiveCallsStress) {
    ThreadPool pool(4);
    std::atomic<size_t> total{0};
    for (int rep = 0; rep < 2000; ++rep) pool.parallel_for(100, 7, [&](size_t b, size_t e) { total += e - b; });
    EXPECT_EQ(total.load(), 200000u);
}

TEST(ThreadPool, ConstructDestroyRepeatedly) {
    for (int i = 0; i < 50; ++i) {
        ThreadPool pool(3);
        std::atomic<int> x{0};
        pool.parallel_for(10, 1, [&](size_t, size_t) { ++x; });
        EXPECT_EQ(x.load(), 10);
    }
}
