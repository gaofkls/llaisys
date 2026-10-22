#include "copy.hpp"

#include "../utils/check.hpp"

#include <cstring>
#include <vector>

namespace llaisys {

void copy_run(std::byte *dst, const std::byte *src, size_t bytes,
              llaisysDeviceType_t dst_device, llaisysDeviceType_t src_device,
              int device_id) {
    if (bytes == 0) {
        return;
    }
    if (dst_device == LLAISYS_DEVICE_CPU && src_device == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst, src, bytes);
        return;
    }

    llaisysMemcpyKind_t kind = LLAISYS_MEMCPY_D2D;
    if (src_device == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
    } else if (dst_device == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
    }

    // The active runtime decides which API performs the copy. Host-to-device and
    // device-to-device go through the destination device, device-to-host through
    // the source device.
    llaisysDeviceType_t active_device =
        (dst_device == LLAISYS_DEVICE_CPU) ? src_device : dst_device;
    core::context().setDevice(active_device, device_id);
    core::context().runtime().api()->memcpy_sync(dst, src, bytes, kind);
}

namespace {

void copy_dim(std::byte *dst, const std::byte *src,
              const std::vector<size_t> &shape,
              const std::vector<ptrdiff_t> &dst_strides,
              const std::vector<ptrdiff_t> &src_strides,
              size_t elem_size, size_t dim,
              llaisysDeviceType_t dst_device, llaisysDeviceType_t src_device, int device_id) {
    const size_t extent = shape[dim];

    if (dim + 1 == shape.size()) {
        if (dst_strides[dim] == 1 && src_strides[dim] == 1) {
            copy_run(dst, src, extent * elem_size, dst_device, src_device, device_id);
        } else {
            for (size_t i = 0; i < extent; ++i) {
                copy_run(dst + i * dst_strides[dim] * elem_size,
                         src + i * src_strides[dim] * elem_size, elem_size,
                         dst_device, src_device, device_id);
            }
        }
        return;
    }

    for (size_t i = 0; i < extent; ++i) {
        copy_dim(dst + i * dst_strides[dim] * elem_size,
                 src + i * src_strides[dim] * elem_size, shape, dst_strides,
                 src_strides, elem_size, dim + 1, dst_device, src_device, device_id);
    }
}

} // namespace

void copy_strided(const tensor_t &dst, const tensor_t &src) {
    CHECK_ARGUMENT(dst != nullptr && src != nullptr, "copy_strided: null tensor");
    CHECK_ARGUMENT(dst->dtype() == src->dtype(), "copy_strided: dtype mismatch");
    CHECK_SAME_SHAPE(dst->shape(), src->shape());

    const size_t elem_size = dst->elementSize();
    const size_t numel = dst->numel();
    if (numel == 0) {
        return;
    }

    if (dst->isContiguous() && src->isContiguous()) {
        copy_run(dst->data(), src->data(), numel * elem_size, dst->deviceType(),
                 src->deviceType(), dst->deviceId());
        return;
    }

    copy_dim(dst->data(), src->data(), dst->shape(), dst->strides(), src->strides(),
             elem_size, 0, dst->deviceType(), src->deviceType(), dst->deviceId());
}

} // namespace llaisys
