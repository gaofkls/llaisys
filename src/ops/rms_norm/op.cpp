#include "op.hpp"

#include "../op_common.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::ops {

namespace {

template <typename T>
void rms_norm_kernel(T *out, const T *in, const T *weight, size_t rows, size_t dim,
                     float eps) {
    // The scale vector is read for every row, so convert it once up front.
    std::vector<float> wf(dim);
    for (size_t j = 0; j < dim; ++j) {
        wf[j] = to_float(weight[j]);
    }

    parallel_for(rows, 1, [&](size_t begin, size_t end) {
        for (size_t r = begin; r < end; ++r) {
            const T *row = in + r * dim;
            float sum_sq = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                const float v = to_float(row[j]);
                sum_sq += v * v;
            }
            const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
            T *out_row = out + r * dim;
            for (size_t j = 0; j < dim; ++j) {
                out_row[j] = from_float<T>(wf[j] * (to_float(row[j]) * scale));
            }
        }
    });
}

} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_ARGUMENT(in->ndim() == 2, "rms_norm: input must be 2-D.");
    CHECK_ARGUMENT(weight->ndim() == 1, "rms_norm: weight must be 1-D.");
    CHECK_ARGUMENT(out->ndim() == 2, "rms_norm: output must be 2-D.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous.");
    CHECK_ARGUMENT(is_supported_float(in->dtype()),
                   "rms_norm: only f32/f16/bf16 are supported.");

    const size_t rows = in->shape()[0];
    const size_t dim = in->shape()[1];
    CHECK_ARGUMENT(weight->shape()[0] == dim, "rms_norm: weight size must match the row size.");
    CHECK_ARGUMENT(out->shape()[0] == rows && out->shape()[1] == dim,
                   "rms_norm: output shape must match the input.");

    core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_kernel(reinterpret_cast<float *>(out->data()),
                               reinterpret_cast<const float *>(in->data()),
                               reinterpret_cast<const float *>(weight->data()), rows, dim,
                               eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_kernel(reinterpret_cast<fp16_t *>(out->data()),
                               reinterpret_cast<const fp16_t *>(in->data()),
                               reinterpret_cast<const fp16_t *>(weight->data()), rows, dim,
                               eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_kernel(reinterpret_cast<bf16_t *>(out->data()),
                               reinterpret_cast<const bf16_t *>(in->data()),
                               reinterpret_cast<const bf16_t *>(weight->data()), rows, dim,
                               eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}

} // namespace llaisys::ops
