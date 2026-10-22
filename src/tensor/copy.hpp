#pragma once

#include "tensor.hpp"

namespace llaisys {

// Copy `src` into `dst`, honouring both tensors' strides.
// dst and src must have the same shape and dtype; devices may differ.
void copy_strided(const tensor_t &dst, const tensor_t &src);

// Copy a single contiguous run of bytes between two devices.
void copy_run(std::byte *dst, const std::byte *src, size_t bytes,
              llaisysDeviceType_t dst_device, llaisysDeviceType_t src_device,
              int device_id);

} // namespace llaisys
