#!/usr/bin/env bash
# Profile the scoring variants with gprof (where does time go) and Valgrind cachegrind
# (how many cache misses does each memory layout cause). Linux, needs: cmake, gprof, valgrind.
#   scripts/profile.sh            default sizes
#   N=500000 scripts/profile.sh   different record count
set -euo pipefail
cd "$(dirname "$0")/.."
N="${N:-1000000}"

echo "== gprof: flat profile, ${N} records x 40 iterations =="
cmake -S . -B build-prof -DPE_PROFILE=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-prof -j --target pe_bench >/dev/null
for v in aos_scalar soa_simd; do
  echo "-- $v"
  ( cd build-prof && rm -f gmon.out && ./pe_bench --variant "$v" --n "$N" --iters 40 >/dev/null \
    && gprof ./pe_bench gmon.out -b -p 2>/dev/null | sed -n '1,6p' )
done

echo
echo "== cachegrind: simulated cache behaviour (smaller input; ~50x slower than native) =="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j --target pe_bench >/dev/null
for v in aos_scalar soa_scalar soa_simd; do
  echo "-- $v"
  valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=/dev/null \
    ./build/pe_bench --variant "$v" --n 200000 --iters 3 2>&1 \
    | grep -E "D   refs|D1  misses|LLd misses|D1  miss rate|LLd miss rate" | sed 's/^==[0-9]*== //'
done
