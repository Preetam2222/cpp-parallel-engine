// Fixed-size thread pool with a chunked parallel_for. The calling thread participates, so
// ThreadPool(1) runs everything inline and ThreadPool(k) uses k threads in total.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace pe {

class ThreadPool {
public:
    using Body = std::function<void(size_t begin, size_t end)>;

    // threads == 0 -> std::thread::hardware_concurrency() (at least 1).
    explicit ThreadPool(size_t threads = 0) {
        if (threads == 0) threads = std::max<size_t>(1, std::thread::hardware_concurrency());
        for (size_t i = 0; i + 1 < threads; ++i) workers_.emplace_back([this] { worker_loop(); });
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    size_t size() const { return workers_.size() + 1; }

    // Calls body(begin, end) on disjoint [begin, end) ranges of at most `grain` indices that
    // together cover [0, n) exactly once. Blocks until done. If any body throws, remaining
    // chunks are skipped and the first exception is rethrown here. One call at a time.
    void parallel_for(size_t n, size_t grain, const Body& body) {
        if (n == 0) return;
        if (grain == 0) grain = 1;
        const size_t chunks = (n + grain - 1) / grain;
        if (workers_.empty()) {  // single-threaded pool: same chunking contract, run inline
            for (size_t b = 0; b < n; b += grain) body(b, std::min(b + grain, n));
            return;
        }
        if (chunks == 1) {
            body(0, n);
            return;
        }
        std::lock_guard<std::mutex> call_lock(call_m_);
        Job job;
        job.body = &body;
        job.n = n;
        job.grain = grain;
        job.chunks = chunks;
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &job;
            ++generation_;
            active_ = workers_.size();
        }
        cv_.notify_all();
        run_chunks(job);  // caller helps
        {
            std::unique_lock<std::mutex> lk(m_);
            done_cv_.wait(lk, [&] { return active_ == 0; });  // every worker has checked in
            job_ = nullptr;
        }
        if (job.error) std::rethrow_exception(job.error);
    }

private:
    struct Job {
        const Body* body = nullptr;
        size_t n = 0, grain = 1, chunks = 0;
        std::atomic<size_t> next{0};
        std::mutex err_m;
        std::exception_ptr error;
    };

    static void run_chunks(Job& j) {
        for (;;) {
            size_t c = j.next.fetch_add(1);
            if (c >= j.chunks) break;
            size_t b = c * j.grain, e = std::min(b + j.grain, j.n);
            try {
                (*j.body)(b, e);
            } catch (...) {
                std::lock_guard<std::mutex> lk(j.err_m);
                if (!j.error) j.error = std::current_exception();
                j.next.store(j.chunks);  // cancel the rest
            }
        }
    }

    void worker_loop() {
        size_t seen = 0;
        for (;;) {
            Job* j;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                j = job_;
            }
            run_chunks(*j);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--active_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_, call_m_;
    std::condition_variable cv_, done_cv_;
    Job* job_ = nullptr;
    size_t generation_ = 0, active_ = 0;
    bool stop_ = false;
};

}  // namespace pe
