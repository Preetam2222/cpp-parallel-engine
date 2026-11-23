#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

#include "pe/pool.hpp"

using namespace pe;

namespace {
struct Tracked {
    static int alive;
    int value;
    explicit Tracked(int v) : value(v) { ++alive; }
    ~Tracked() { --alive; }
};
int Tracked::alive = 0;

struct alignas(64) Wide {
    char data[64];
};

struct Throws {
    explicit Throws(bool t) { if (t) throw std::runtime_error("ctor"); }
};
}  // namespace

TEST(ObjectPool, AllocationsAreDistinctAndCounted) {
    ObjectPool<Tracked, 8> pool;
    std::vector<Tracked*> ps;
    std::set<Tracked*> uniq;
    for (int i = 0; i < 100; ++i) {
        ps.push_back(pool.create(i));
        uniq.insert(ps.back());
    }
    EXPECT_EQ(uniq.size(), 100u);
    EXPECT_EQ(pool.live(), 100u);
    EXPECT_EQ(Tracked::alive, 100);
    EXPECT_GE(pool.capacity(), 100u);
    EXPECT_EQ(pool.capacity() % 8, 0u);  // grows in whole blocks
    for (int i = 0; i < 100; ++i) EXPECT_EQ(ps[i]->value, i);  // no object overwrote another
    for (auto* p : ps) pool.destroy(p);
    EXPECT_EQ(pool.live(), 0u);
    EXPECT_EQ(Tracked::alive, 0);  // destructors ran
}

TEST(ObjectPool, FreedSlotIsReusedFirst) {
    ObjectPool<Tracked, 8> pool;
    Tracked* a = pool.create(1);
    Tracked* b = pool.create(2);
    pool.destroy(a);
    Tracked* c = pool.create(3);
    EXPECT_EQ(c, a);  // LIFO free list
    EXPECT_EQ(b->value, 2);
    pool.destroy(b);
    pool.destroy(c);
}

TEST(ObjectPool, FirstBlockIsContiguousAndAscending) {
    ObjectPool<Tracked, 16> pool;
    std::vector<Tracked*> ps;
    for (int i = 0; i < 16; ++i) ps.push_back(pool.create(i));
    // Slots are max(sizeof(T), sizeof(void*)) bytes: constant positive stride, no gaps or overlap.
    const std::ptrdiff_t stride = reinterpret_cast<char*>(ps[1]) - reinterpret_cast<char*>(ps[0]);
    EXPECT_EQ(stride, (std::ptrdiff_t)std::max(sizeof(Tracked), sizeof(void*)));
    for (int i = 1; i < 16; ++i)
        EXPECT_EQ(reinterpret_cast<char*>(ps[i]) - reinterpret_cast<char*>(ps[i - 1]), stride) << i;
    for (auto* p : ps) pool.destroy(p);
}

TEST(ObjectPool, HonoursOverAlignment) {
    ObjectPool<Wide, 4> pool;
    std::vector<Wide*> ps;
    for (int i = 0; i < 10; ++i) {
        ps.push_back(pool.create());
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ps.back()) % 64, 0u);
    }
    for (auto* p : ps) pool.destroy(p);
}

TEST(ObjectPool, ConstructorExceptionReturnsTheSlot) {
    ObjectPool<Throws, 4> pool;
    EXPECT_THROW(pool.create(true), std::runtime_error);
    EXPECT_EQ(pool.live(), 0u);
    Throws* ok = pool.create(false);
    EXPECT_EQ(pool.live(), 1u);
    pool.destroy(ok);
}

TEST(ObjectPool, RandomChurnKeepsValuesIntact) {
    ObjectPool<Tracked, 32> pool;
    std::vector<Tracked*> live(200, nullptr);
    uint64_t s = 1;
    auto rnd = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int step = 0; step < 20000; ++step) {
        size_t i = rnd() % live.size();
        if (live[i]) {
            ASSERT_EQ(live[i]->value, (int)i);
            pool.destroy(live[i]);
            live[i] = nullptr;
        } else {
            live[i] = pool.create((int)i);
        }
    }
    size_t n = 0;
    for (auto* p : live) if (p) { ++n; pool.destroy(p); }
    (void)n;
    EXPECT_EQ(pool.live(), 0u);
    EXPECT_EQ(Tracked::alive, 0);
}
