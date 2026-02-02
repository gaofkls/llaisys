#include "op.hpp"

#include "../../utils.hpp"

namespace llaisys::ops {
template <typename T>
void argmax_(const T *vals, size_t n, T *max_val, int64_t *max_idx) {
    if (n == 0) {
        return;
    }
    T best_val = vals[0];
    size_t best_i = 0;

    for (size_t i = 1; i < n; ++i) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float v = llaisys::utils::cast<float>(vals[i]);
            float b = llaisys::utils::cast<float>(best_val);
            if (v > b) {
                best_val = vals[i];
                best_i = i;
            }
        } else {
            if (vals[i] > best_val) {
                best_val = vals[i];
                best_i = i;
            }
        }
    }

    max_val[0] = best_val;
    max_idx[0] = static_cast<int64_t>(best_i);
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be int64");
    CHECK_ARGUMENT(vals->ndim() == 1, "argmax: vals must be 1D");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->numel() == 1, "argmax: max_idx must be shape (1,)");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->numel() == 1, "argmax: max_val must be shape (1,)");
    ASSERT(vals->isContiguous() && max_val->isContiguous() && max_idx->isContiguous(), "argmax: all tensors must be contiguous.");

    size_t n = vals->numel();

    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<const float *>(vals->data()), n,
                       reinterpret_cast<float *>(max_val->data()),
                       reinterpret_cast<int64_t *>(max_idx->data()));
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<const llaisys::fp16_t *>(vals->data()), n,
                       reinterpret_cast<llaisys::fp16_t *>(max_val->data()),
                       reinterpret_cast<int64_t *>(max_idx->data()));
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<const llaisys::bf16_t *>(vals->data()), n,
                       reinterpret_cast<llaisys::bf16_t *>(max_val->data()),
                       reinterpret_cast<int64_t *>(max_idx->data()));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} 