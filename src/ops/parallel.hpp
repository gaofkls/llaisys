#pragma once

// A tiny portable parallel_for. The project should not require OpenMP (it is not
// available on every toolchain, macOS / clang being the common case), so the
// kernels split a range across std::thread workers instead.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace llaisys::ops {

inline unsigned int worker_count() {
    static const unsigned int count = [] {
        unsigned int hw = std::thread::hardware_concurrency();
        return hw == 0 ? 1u : hw;
    }();
    return count;
}

// Calls body(begin, end) over chunks covering [0, n). Falls back to a serial call
// when the range is too small to be worth the thread overhead. Exceptions must
// not escape body: worker threads would call std::terminate.
inline void parallel_for(size_t n, size_t min_chunk,
                         const std::function<void(size_t, size_t)> &body) {
    if (n == 0) {
        return;
    }

    const unsigned int hw = worker_count();
    if (hw <= 1 || n <= min_chunk) {
        body(0, n);
        return;
    }

    size_t workers = std::min<size_t>(hw, (n + min_chunk - 1) / min_chunk);
    if (workers <= 1) {
        body(0, n);
        return;
    }

    const size_t chunk = (n + workers - 1) / workers;
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);

    for (size_t t = 1; t < workers; ++t) {
        const size_t begin = std::min(n, t * chunk);
        const size_t end = std::min(n, begin + chunk);
        if (begin >= end) {
            break;
        }
        pool.emplace_back(body, begin, end);
    }

    body(0, std::min(n, chunk));

    for (auto &thread : pool) {
        thread.join();
    }
}

} // namespace llaisys::ops
