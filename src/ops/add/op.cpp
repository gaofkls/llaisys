#include "op.hpp"

#include "../op_common.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

namespace {

template <typename T>
void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    parallel_for(numel, 1 << 15, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            c[i] = from_float<T>(to_float(a[i]) + to_float(b[i]));
        }
    });
}

} // namespace

void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(),
           "add: all tensors must be contiguous.");
    CHECK_ARGUMENT(is_supported_float(c->dtype()), "add: only f32/f16/bf16 are supported.");

    core::context().setDevice(c->deviceType(), c->deviceId());

    const size_t numel = c->numel();
    switch (c->dtype()) {
    case LLAISYS_DTYPE_F32:
        return add_kernel(reinterpret_cast<float *>(c->data()),
                          reinterpret_cast<const float *>(a->data()),
                          reinterpret_cast<const float *>(b->data()), numel);
    case LLAISYS_DTYPE_F16:
        return add_kernel(reinterpret_cast<fp16_t *>(c->data()),
                          reinterpret_cast<const fp16_t *>(a->data()),
                          reinterpret_cast<const fp16_t *>(b->data()), numel);
    case LLAISYS_DTYPE_BF16:
        return add_kernel(reinterpret_cast<bf16_t *>(c->data()),
                          reinterpret_cast<const bf16_t *>(a->data()),
                          reinterpret_cast<const bf16_t *>(b->data()), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(c->dtype());
    }
}

} // namespace llaisys::ops
