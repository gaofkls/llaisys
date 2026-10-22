#include "op.hpp"

#include "../op_common.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

namespace {

template <typename T>
void argmax_kernel(const T *vals, size_t n, T *max_val, int64_t *max_idx) {
    size_t best = 0;
    float best_val = to_float(vals[0]);
    for (size_t i = 1; i < n; ++i) {
        const float value = to_float(vals[i]);
        if (value > best_val) {
            best_val = value;
            best = i;
        }
    }
    max_val[0] = from_float<T>(best_val);
    max_idx[0] = static_cast<int64_t>(best);
}

} // namespace

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be int64.");
    CHECK_ARGUMENT(vals->ndim() == 1, "argmax: vals must be 1D.");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->numel() == 1,
                   "argmax: max_idx must have shape (1,).");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->numel() == 1,
                   "argmax: max_val must have shape (1,).");
    ASSERT(vals->isContiguous() && max_val->isContiguous() && max_idx->isContiguous(),
           "argmax: all tensors must be contiguous.");
    CHECK_ARGUMENT(is_supported_float(vals->dtype()),
                   "argmax: only f32/f16/bf16 are supported.");

    core::context().setDevice(vals->deviceType(), vals->deviceId());

    const size_t n = vals->numel();
    CHECK_ARGUMENT(n > 0, "argmax: vals must not be empty.");

    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_kernel(reinterpret_cast<const float *>(vals->data()), n,
                             reinterpret_cast<float *>(max_val->data()),
                             reinterpret_cast<int64_t *>(max_idx->data()));
    case LLAISYS_DTYPE_F16:
        return argmax_kernel(reinterpret_cast<const fp16_t *>(vals->data()), n,
                             reinterpret_cast<fp16_t *>(max_val->data()),
                             reinterpret_cast<int64_t *>(max_idx->data()));
    case LLAISYS_DTYPE_BF16:
        return argmax_kernel(reinterpret_cast<const bf16_t *>(vals->data()), n,
                             reinterpret_cast<bf16_t *>(max_val->data()),
                             reinterpret_cast<int64_t *>(max_idx->data()));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}

} // namespace llaisys::ops
