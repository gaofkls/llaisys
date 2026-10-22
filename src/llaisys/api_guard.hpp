#pragma once

// Every function exported from the shared library is called through a foreign
// (C) function boundary -- from Python via ctypes, for instance. C++ exceptions
// must never unwind across that boundary: there is no handler on the other side,
// and the runtime reacts by calling std::terminate. These helpers turn an
// exception into a printed message plus a well-defined failure value.

#include <exception>
#include <iostream>
#include <type_traits>
#include <utility>

namespace llaisys::api {

inline void report(const char *where, const char *what) {
    std::cerr << "[llaisys] " << where << ": " << what << std::endl;
}

template <typename Fail, typename Fn>
auto run(const char *where, Fail fail_value, Fn &&fn) noexcept
    -> std::invoke_result_t<Fn> {
    try {
        return std::forward<Fn>(fn)();
    } catch (const std::exception &e) {
        report(where, e.what());
    } catch (...) {
        report(where, "unknown exception");
    }
    return fail_value;
}

template <typename Fn>
void run_void(const char *where, Fn &&fn) noexcept {
    try {
        std::forward<Fn>(fn)();
    } catch (const std::exception &e) {
        report(where, e.what());
    } catch (...) {
        report(where, "unknown exception");
    }
}

} // namespace llaisys::api
