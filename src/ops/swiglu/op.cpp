#include "op.hpp"

#include "../op_common.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {

namespace {

// out = up * (gate * sigmoid(gate))
template <typename T>
void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    parallel_for(numel, 1 << 15, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const float g = to_float(gate[i]);
            const float silu = g / (1.0f + std::exp(-g));
            out[i] = from_float<T>(to_float(up[i]) * silu);
        }
    });
}

} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_ARGUMENT(out->ndim() == 2, "swiglu: output must be 2-D.");
    CHECK_ARGUMENT(gate->ndim() == 2, "swiglu: gate must be 2-D.");
    CHECK_ARGUMENT(up->ndim() == 2, "swiglu: up must be 2-D.");
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous.");
    CHECK_ARGUMENT(is_supported_float(gate->dtype()),
                   "swiglu: only f32/f16/bf16 are supported.");

    core::context().setDevice(out->deviceType(), out->deviceId());

    const size_t numel = out->numel();
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_kernel(reinterpret_cast<float *>(out->data()),
                             reinterpret_cast<const float *>(gate->data()),
                             reinterpret_cast<const float *>(up->data()), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_kernel(reinterpret_cast<fp16_t *>(out->data()),
                             reinterpret_cast<const fp16_t *>(gate->data()),
                             reinterpret_cast<const fp16_t *>(up->data()), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_kernel(reinterpret_cast<bf16_t *>(out->data()),
                             reinterpret_cast<const bf16_t *>(gate->data()),
                             reinterpret_cast<const bf16_t *>(up->data()), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops
