#include "pe/records.hpp"

namespace pe {
namespace {
inline uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline float unit(uint64_t& s) { return (float)(splitmix64(s) >> 40) * (1.0f / 16777216.0f); }  // [0,1)
}  // namespace

void make_dataset(size_t n, uint64_t seed, std::vector<RecordAoS>& aos, SoA& soa) {
    aos.assign(n, RecordAoS{});
    soa.n = n;
    soa.x.allocate(n);
    soa.y.allocate(n);
    soa.z.allocate(n);
    soa.w.allocate(n);
    uint64_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        float x = unit(s) * 2.0f - 1.0f, y = unit(s) * 2.0f - 1.0f, z = unit(s) * 2.0f - 1.0f, w = unit(s);
        aos[i] = RecordAoS{x, y, z, w, (uint32_t)i, 0u, {0.0f, 0.0f}};
        soa.x[i] = x;
        soa.y[i] = y;
        soa.z[i] = z;
        soa.w[i] = w;
    }
}

}  // namespace pe
