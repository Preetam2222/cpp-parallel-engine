// Fixed-size object pool (free-list allocator). Objects are carved out of large contiguous
// blocks, so allocation is a pointer pop, free is a pointer push, and objects created back to
// back are adjacent in memory. Not thread-safe: use one pool per thread.
#pragma once
#include <cstddef>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace pe {

template <class T, size_t BlockObjects = 4096>
class ObjectPool {
    static_assert(BlockObjects > 0, "BlockObjects must be positive");
    union Slot {
        Slot* next;
        alignas(T) unsigned char storage[sizeof(T)];
    };

public:
    ObjectPool() = default;
    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;
    // Does not run destructors of objects still alive; destroy() them first.
    ~ObjectPool() = default;

    // Raw storage for one T (uninitialised).
    void* allocate() {
        if (!free_) grow();
        Slot* s = free_;
        free_ = s->next;
        ++live_;
        return s->storage;
    }

    void deallocate(void* p) noexcept {
        Slot* s = reinterpret_cast<Slot*>(p);
        s->next = free_;
        free_ = s;
        --live_;
    }

    template <class... Args>
    T* create(Args&&... args) {
        void* p = allocate();
        try {
            return new (p) T(std::forward<Args>(args)...);
        } catch (...) {
            deallocate(p);
            throw;
        }
    }

    void destroy(T* p) noexcept {
        p->~T();
        deallocate(p);
    }

    size_t live() const { return live_; }
    size_t capacity() const { return blocks_.size() * BlockObjects; }

private:
    void grow() {
        blocks_.emplace_back(new Slot[BlockObjects]);
        Slot* b = blocks_.back().get();
        for (size_t i = BlockObjects; i-- > 0;) {  // reverse so allocation walks addresses upward
            b[i].next = free_;
            free_ = &b[i];
        }
    }

    std::vector<std::unique_ptr<Slot[]>> blocks_;
    Slot* free_ = nullptr;
    size_t live_ = 0;
};

}  // namespace pe
