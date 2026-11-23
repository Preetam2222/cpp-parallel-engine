#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace pe {

inline double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Compiler barrier so benchmarked results are not optimised away.
template <class T>
inline void keep(const T& v) {
    asm volatile("" : : "g"(&v) : "memory");
}

// Median seconds per call of f over `reps` timed repetitions. Each repetition runs f enough
// times to take at least ~30 ms (one call if f is already slower than that).
template <class F>
double median_seconds(F&& f, int reps = 7) {
    double t0 = now_s();
    f();  // warm-up + calibration
    double one = std::max(now_s() - t0, 1e-9);
    int iters = std::max(1, (int)std::ceil(0.03 / one));
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        double s = now_s();
        for (int i = 0; i < iters; ++i) f();
        t.push_back((now_s() - s) / iters);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

}  // namespace pe
