#pragma once

#include "../utils.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cstddef>

namespace llaisys::ops {

// All math is done in float32; half precision types only affect storage.
template <typename T>
inline float to_float(const T &value) {
    return utils::cast<float>(value);
}

template <typename T>
inline T from_float(float value) {
    return utils::cast<T>(value);
}

inline bool is_supported_float(llaisysDataType_t dtype) {
    return dtype == LLAISYS_DTYPE_F32 || dtype == LLAISYS_DTYPE_F16
        || dtype == LLAISYS_DTYPE_BF16;
}

} // namespace llaisys::ops
