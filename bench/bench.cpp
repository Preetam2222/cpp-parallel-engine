// Benchmarks for every optimisation in the engine. Every result is checked against the baseline
// before it is timed, and times are medians of several repetitions.
//
//   pe_bench                       full run (markdown tables)
//   pe_bench --quick               smaller sizes, fewer repetitions
//   pe_bench --threads N           thread count for the parallel rows (default: all cores)
//   pe_bench --variant NAME --n N [--iters K]
//                                  run ONE scoring variant K times, no timing output
//                                  (for gprof / valgrind; NAME as in the table)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "pe/kernels.hpp"
#include "pe/pool.hpp"
#include "pe/records.hpp"
#include "pe/thread_pool.hpp"
#include "pe/timing.hpp"
#include "pe/transpose.hpp"

using namespace pe;

namespace {

constexpr float kThr = 0.25f;

struct Variant {
    std::string name;
    std::function<Stat()> run;
};

std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "unknown";
}

std::vector<Variant> make_variants(const std::vector<RecordAoS>& aos, const SoA& soa, ThreadPool& pool) {
    const SoAView v = soa.view();
    return {
        {"aos_scalar", [&aos] { return score_aos_scalar(aos.data(), aos.size(), kThr); }},
        {"aos_default", [&aos] { return score_aos_default(aos.data(), aos.size(), kThr); }},
        {"soa_scalar", [v] { return score_soa_scalar(v, kThr); }},
        {"soa_unrolled", [v] { return score_soa_unrolled(v, kThr); }},
        {"soa_autovec", [v] { return score_soa_autovec(v, kThr); }},
        {"soa_simd", [v] { return score_soa_simd(v, kThr); }},
        {"soa_simd_parallel", [v, &pool] { return score_soa_parallel(pool, v, kThr); }},
    };
}

void bench_pipeline(size_t n, int reps, size_t threads) {
    std::vector<RecordAoS> aos;
    SoA soa;
    make_dataset(n, 12345, aos, soa);
    ThreadPool pool(threads);
    auto variants = make_variants(aos, soa, pool);
    const Stat ref = variants[0].run();

    std::printf("\n### Scoring pipeline, N = %zu records (SoA %.0f MB, AoS %.0f MB), %zu thread(s) in parallel row\n\n", n,
                n * 16.0 / 1e6, n * 32.0 / 1e6, pool.size());
    std::printf("| variant | time (ms) | Mrecords/s | useful GB/s | vs aos_scalar | vs aos_default |\n|---|---|---|---|---|---|\n");
    double t_scalar = 0, t_default = 0;
    for (auto& var : variants) {
        Stat s = var.run();
        if (s.count != ref.count || std::fabs(s.sum - ref.sum) > 1e-4 * ref.sum) {
            std::fprintf(stderr, "MISMATCH in %s: count %llu vs %llu, sum %.6f vs %.6f\n", var.name.c_str(),
                         (unsigned long long)s.count, (unsigned long long)ref.count, s.sum, ref.sum);
            std::exit(1);
        }
        double t = median_seconds([&] { Stat r = var.run(); keep(r); }, reps);
        if (var.name == "aos_scalar") t_scalar = t;
        if (var.name == "aos_default") t_default = t;
        std::printf("| %s | %.3f | %.0f | %.1f | %.2fx | %.2fx |\n", var.name.c_str(), t * 1e3, n / t / 1e6,
                    n * 16.0 / t / 1e9, t_scalar / t, t_default / t);
    }
}

void bench_transpose(size_t n, int reps, size_t threads) {
    std::vector<float> in(n * n), out(n * n), ref(n * n);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (float)(i % 9973);
    transpose_naive(in.data(), ref.data(), n, n);
    ThreadPool pool(threads);

    std::printf("\n### Transpose %zux%zu floats (%.0f MB in + %.0f MB out), %zu thread(s) in parallel row\n\n", n, n,
                n * n * 4.0 / 1e6, n * n * 4.0 / 1e6, pool.size());
    std::printf("| variant | time (ms) | GB/s | speedup |\n|---|---|---|---|\n");
    struct T { std::string name; std::function<void()> run; };
    std::vector<T> ts = {
        {"naive", [&] { transpose_naive(in.data(), out.data(), n, n); }},
        {"blocked tile=16", [&] { transpose_blocked(in.data(), out.data(), n, n, 16); }},
        {"blocked tile=32", [&] { transpose_blocked(in.data(), out.data(), n, n, 32); }},
        {"blocked tile=64", [&] { transpose_blocked(in.data(), out.data(), n, n, 64); }},
        {"blocked tile=32 + threads", [&] { transpose_parallel(pool, in.data(), out.data(), n, n, 32); }},
    };
    double t0 = 0;
    for (auto& t : ts) {
        std::fill(out.begin(), out.end(), 0.0f);
        t.run();
        if (std::memcmp(out.data(), ref.data(), out.size() * 4) != 0) {
            std::fprintf(stderr, "MISMATCH in transpose %s\n", t.name.c_str());
            std::exit(1);
        }
        double s = median_seconds([&] { t.run(); keep(out[0]); }, reps);
        if (t.name == "naive") t0 = s;
        std::printf("| %s | %.2f | %.1f | %.2fx |\n", t.name.c_str(), s * 1e3, 2.0 * n * n * 4 / s / 1e9, t0 / s);
    }
}

struct Node {
    uint64_t pad[3];
    double v;
};

template <class Alloc, class Free>
double churn(size_t live, size_t ops, Alloc alloc, Free release) {
    std::vector<Node*> p(live);
    uint64_t s = 99;
    auto rnd = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (size_t i = 0; i < live; ++i) p[i] = alloc((double)(rnd() & 1023));
    for (size_t k = 0; k < ops; ++k) {
        size_t i = rnd() % live;
        release(p[i]);
        p[i] = alloc((double)(rnd() & 1023));
    }
    double sum = 0;
    for (Node* x : p) sum += x->v;
    for (Node* x : p) release(x);
    return sum;
}

void bench_pool(size_t live, size_t ops, int reps) {
    std::printf("\n### Allocation churn: %zu live objects, %zu replace operations (32-byte nodes)\n\n", live, ops);
    std::printf("| allocator | time (ms) | Mops/s | speedup |\n|---|---|---|---|\n");
    auto run_new = [&] {
        return churn(live, ops, [](double v) { return new Node{{0, 0, 0}, v}; }, [](Node* p) { delete p; });
    };
    auto run_pool = [&] {
        ObjectPool<Node> pool;
        return churn(live, ops, [&](double v) { return pool.create(Node{{0, 0, 0}, v}); }, [&](Node* p) { pool.destroy(p); });
    };
    if (run_new() != run_pool()) {
        std::fprintf(stderr, "MISMATCH in churn checksum\n");
        std::exit(1);
    }
    double tn = median_seconds([&] { double r = run_new(); keep(r); }, reps);
    double tp = median_seconds([&] { double r = run_pool(); keep(r); }, reps);
    std::printf("| new / delete | %.2f | %.1f | 1.00x |\n", tn * 1e3, (live + 2.0 * ops) / tn / 1e6);
    std::printf("| ObjectPool | %.2f | %.1f | %.2fx |\n", tp * 1e3, (live + 2.0 * ops) / tp / 1e6, tn / tp);
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    size_t threads = 0, n = 0, iters = 1;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--quick") quick = true;
        else if (a == "--threads") threads = std::stoul(next());
        else if (a == "--variant") only = next();
        else if (a == "--n") n = std::stoul(next());
        else if (a == "--iters") iters = std::stoul(next());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    if (!only.empty()) {  // single-variant mode for profilers
        if (n == 0) n = 1 << 20;
        std::vector<RecordAoS> aos;
        SoA soa;
        make_dataset(n, 12345, aos, soa);
        ThreadPool pool(threads);
        for (auto& v : make_variants(aos, soa, pool)) {
            if (v.name != only) continue;
            Stat s;
            for (size_t k = 0; k < iters; ++k) { s = v.run(); keep(s); }
            std::printf("%s: n=%zu iters=%zu count=%llu sum=%.4f\n", only.c_str(), n, iters, (unsigned long long)s.count, s.sum);
            return 0;
        }
        std::fprintf(stderr, "unknown variant %s\n", only.c_str());
        return 2;
    }

    std::printf("## pe_bench\n\nCPU: %s, hardware threads: %u, AVX2: %s\n", cpu_model().c_str(),
                std::thread::hardware_concurrency(), has_avx2() ? "yes" : "no");
    int reps = quick ? 3 : 7;
    bench_pipeline(1 << 16, reps, threads);                 // fits in L2
    bench_pipeline(quick ? 1 << 20 : 1 << 22, reps, threads);  // streams from DRAM
    bench_transpose(quick ? 1024 : 4096, reps, threads);
    bench_pool(100000, quick ? 500000 : 2000000, reps);
    return 0;
}
