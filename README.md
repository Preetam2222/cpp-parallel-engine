# cpp-parallel-engine

A small C++17 data-processing engine used to measure, one at a time, what each classic CPU
optimisation is worth: data layout (AoS vs SoA), loop unrolling, SIMD, cache blocking, memory
pooling and multithreading. Every optimised path is checked against a simple reference, tested
with GoogleTest (also under ASan/UBSan/TSan), and benchmarked with medians of repeated runs.

```
include/pe/    records.hpp      AoS / SoA layouts, 64-byte aligned buffer
               kernels.hpp      scoring pipeline: 7 implementations of the same computation
               thread_pool.hpp  fixed pool, chunked parallel_for, exception-safe
               pool.hpp         free-list object pool (contiguous blocks)
               transpose.hpp    naive vs cache-blocked vs threaded transpose
src/           kernel / transpose / dataset implementations
bench/         pe_bench: verifies each variant against the baseline, then times it
tests/         GoogleTest: 22 tests
scripts/       profile.sh: gprof + Valgrind cachegrind
```

## Build, test, benchmark

```bash
cmake -S . -B build && cmake --build build -j
./build/pe_tests                      # 22 tests
./build/pe_bench                      # all tables below (add --quick for a fast run)
./scripts/profile.sh                  # gprof + cachegrind (Linux)

# sanitizers
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DPE_SANITIZE=address,undefined
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DPE_SANITIZE=thread
```

Needs CMake 3.16+, a C++17 compiler, and GoogleTest (`apt install libgtest-dev` or
`brew install googletest`). The AVX2 path is selected at run time on x86-64; other CPUs (e.g. Apple
Silicon) use the portable `omp simd` path. The x86 numbers below were not measured on ARM.

## The workload

For each record compute `s = sqrt(x² + y² + z²) · w`; keep it if `s > 0.25`; return the sum and
count of kept scores. AoS records are 32 bytes (x, y, z, w plus id/flags/meta that the scan does
not need); SoA stores x, y, z, w as four separate arrays.

| variant | what changes |
|---|---|
| `aos_scalar` | naive loop, auto-vectorisation disabled (models naive code) |
| `aos_default` | same loop, compiler at `-O3` unrestricted |
| `soa_scalar` | layout change only |
| `soa_unrolled` | + 4× unroll with independent accumulators |
| `soa_autovec` | + branch-free `omp simd` reduction (portable vector path) |
| `soa_simd` | hand-written AVX2 intrinsics, runtime dispatch |
| `soa_simd_parallel` | AVX2 + thread pool, deterministic combine |

All variants round identically per record (`-ffp-contract=off`), so `count` matches exactly and
`sum` differs only by float summation order (partial sums are flushed to double every 512 records).

## Results

Measured on a 2-core cloud VM (Intel Xeon @ 2.1 GHz, AVX2 and AVX-512 available, GCC 13 `-O3`).
Median of 7 repetitions. Shared-VM timing is noisy: expect run-to-run variation of roughly ±15%.

**Scoring pipeline, 4,194,304 records (SoA 67 MB, AoS 134 MB; streams from memory)**

| variant | time (ms) | Mrecords/s | vs aos_scalar |
|---|---|---|---|
| aos_scalar | 32.84 | 128 | 1.00x |
| aos_default | 33.48 | 125 | 0.98x |
| soa_scalar | 23.96 | 175 | 1.37x |
| soa_unrolled | 17.78 | 236 | 1.85x |
| soa_autovec | 6.44 | 652 | 5.10x |
| soa_simd | 2.40 | 1752 | 13.71x |
| soa_simd_parallel (2 threads) | 1.56 | 2691 | 21.07x |

**Same pipeline, 65,536 records (SoA 1 MB; fits in cache)**

| variant | time (ms) | vs aos_scalar |
|---|---|---|
| aos_scalar | 0.336 | 1.00x |
| soa_simd | 0.018 | 19.02x |
| soa_simd_parallel (2 threads) | 0.033 | 10.29x |

At 65K records the threaded version is **slower** than single-threaded SIMD: each call is ~18 µs, which
is the same order as waking the pool. Threading only pays once there is enough work per call.

**Transpose, 4096×4096 floats**

| variant | time (ms) | speedup |
|---|---|---|
| naive | 235.6 | 1.00x |
| blocked, tile 16 / 32 / 64 | 115.6 / 70.6 / 78.6 | 2.0x / 3.3x / 3.0x |
| blocked tile 32 + 2 threads | 39.4 | 5.99x |

**Allocation churn** (100k live 32-byte objects, 2M replace operations)

| allocator | time (ms) | speedup |
|---|---|---|
| new / delete | 93.3 | 1.00x |
| ObjectPool | 21.3 | 4.38x |

### How to read these numbers

- The headline SIMD speedups are measured against a deliberately naive scalar baseline. Part of the
  gain is that the SIMD kernel is branch-free, while the scalar loop has a data-dependent `if` that
  mispredicts. A hand-tuned scalar baseline would narrow the gap.
- `soa_scalar` alone is only 1.4x at 4M records: the layout change matters mostly because it enables
  SIMD, and halves the bytes read (see cachegrind below).
- `aos_default` equals `aos_scalar`: GCC 13 does not vectorise the AoS loop on its own.
- Speedups depend on this CPU and compiler. Treat them as one data point, not a general constant.

### Profiling

`scripts/profile.sh` output on this machine:

- **gprof**: 93% of `aos_scalar` run time is in the kernel itself (6.75 ms/call at 1M records) versus
  0.75 ms/call for the AVX2 kernel, so the hot path is the scoring loop, not allocation or setup.
- **cachegrind** (200,000 records, 3 passes): L1 data **read** misses are 313,325 for AoS versus
  163,325 for SoA, about 1.9x. That matches the layout: AoS reads 32 bytes per record, SoA reads 16.
  (Write misses are identical because they come from building the dataset.)

## Tests

22 GoogleTest cases, run with `./build/pe_tests`:

- **kernels**: every variant vs an independent reference on 16 sizes (0, 1, 7, 8, 9, 511, 512, 513,
  100003, ...), threshold extremes, unaligned slices, 1 to 8 threads, bit-identical results across
  repeated parallel runs, matching AoS/SoA data.
- **thread pool**: every index visited exactly once across thread counts, sizes and grains; empty
  range; exception propagation and reuse; 2000 back-to-back calls; repeated construct/destroy.
- **object pool**: distinct addresses, destructors run, LIFO reuse, contiguity, 64-byte alignment,
  constructor exceptions return the slot, 20,000-step random churn.
- **transpose**: ragged shapes (1×1, 1×7, 33×65, 129×31, ...) and tile sizes vs the naive version,
  double-transpose identity.

The suite also passes under AddressSanitizer + UBSan and under ThreadSanitizer.

A test caught a real bug during development: a single-threaded pool called the body once over the
whole range instead of in `grain`-sized chunks, breaking the documented contract.

## Limitations

- Benchmarks were run on a 2-core VM; scaling beyond 2 threads is untested.
- x86-64 AVX2 and the portable `omp simd` path were measured; the NEON/ARM path compiles but was not
  benchmarked.
- `ObjectPool` is single-threaded by design (one pool per thread).
