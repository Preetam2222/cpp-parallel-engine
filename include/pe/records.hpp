// Data layouts: array-of-structs (AoS) vs struct-of-arrays (SoA), plus an aligned buffer.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

namespace pe {

// 32-byte record. Only x,y,z,w are used by the scoring pipeline, so a scan over AoS data
// moves twice as many bytes as it needs to (id/flags/meta ride along in every cache line).
struct RecordAoS {
    float x, y, z, w;
    uint32_t id, flags;
    float meta[2];
};
static_assert(sizeof(RecordAoS) == 32, "RecordAoS must stay 32 bytes");

// Move-only, 64-byte aligned (cache line / AVX-512 friendly) array of trivially-copyable T.
template <class T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t n) { allocate(n); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }
    ~AlignedBuffer() { release(); }

    void allocate(size_t n) {
        release();
        if (n == 0) return;
        size_t bytes = (n * sizeof(T) + 63) / 64 * 64;  // aligned_alloc needs a multiple of alignment
        p_ = static_cast<T*>(std::aligned_alloc(64, bytes));
        if (!p_) throw std::bad_alloc();
        n_ = n;
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return p_[i]; }
    const T& operator[](size_t i) const { return p_[i]; }

private:
    void release() { std::free(p_); p_ = nullptr; n_ = 0; }
    T* p_ = nullptr;
    size_t n_ = 0;
};

// Non-owning view of SoA data.
struct SoAView {
    const float *x, *y, *z, *w;
    size_t n;
    SoAView slice(size_t begin, size_t end) const {
        return {x + begin, y + begin, z + begin, w + begin, end - begin};
    }
};

struct SoA {
    AlignedBuffer<float> x, y, z, w;
    size_t n = 0;
    SoAView view() const { return {x.data(), y.data(), z.data(), w.data(), n}; }
};

// Deterministic synthetic dataset: x,y,z ~ U[-1,1), w ~ U[0,1). Same values in both layouts.
void make_dataset(size_t n, uint64_t seed, std::vector<RecordAoS>& aos, SoA& soa);

}  // namespace pe
