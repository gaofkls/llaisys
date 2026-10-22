#include "op.hpp"

#include "../op_common.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::ops {

namespace {

// Rotate every head of every token. The rotation pairs the first half of the head
// with the second half, which is what Qwen2 / the reference implementation do:
//   a' = a * cos(phi) - b * sin(phi)
//   b' = b * cos(phi) + a * sin(phi)
//   phi(j, p) = p / theta^(2j/d)
template <typename T>
void rope_kernel(T *out, const T *in, const int64_t *pos_ids, size_t seq_len,
                 size_t nhead, size_t d, float theta) {
    const size_t half = d / 2;

    // Frequency table is shared, angles are per position.
    std::vector<double> inv_freq(half);
    for (size_t j = 0; j < half; ++j) {
        inv_freq[j] = std::pow(static_cast<double>(theta),
                               -2.0 * static_cast<double>(j) / static_cast<double>(d));
    }

    parallel_for(seq_len, 1, [&](size_t begin, size_t end) {
        std::vector<double> cos_tab(half);
        std::vector<double> sin_tab(half);

        for (size_t i = begin; i < end; ++i) {
            const double p = static_cast<double>(pos_ids[i]);
            for (size_t j = 0; j < half; ++j) {
                const double angle = p * inv_freq[j];
                cos_tab[j] = std::cos(angle);
                sin_tab[j] = std::sin(angle);
            }

            const T *in_row = in + i * nhead * d;
            T *out_row = out + i * nhead * d;

            for (size_t h = 0; h < nhead; ++h) {
                const T *x = in_row + h * d;
                T *y = out_row + h * d;
                for (size_t j = 0; j < half; ++j) {
                    const double a = static_cast<double>(to_float(x[j]));
                    const double b = static_cast<double>(to_float(x[j + half]));
                    y[j] = from_float<T>(static_cast<float>(a * cos_tab[j] - b * sin_tab[j]));
                    y[j + half] =
                        from_float<T>(static_cast<float>(b * cos_tab[j] + a * sin_tab[j]));
                }
            }
        }
    });
}

} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_ARGUMENT(in->ndim() == 3, "rope: input must have shape [seq, nhead, d].");
    CHECK_ARGUMENT(out->ndim() == 3, "rope: output must have shape [seq, nhead, d].");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0],
                   "rope: pos_ids must have shape [seq].");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous.");
    CHECK_ARGUMENT(is_supported_float(in->dtype()), "rope: only f32/f16/bf16 are supported.");

    const size_t seq_len = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t d = in->shape()[2];
    CHECK_ARGUMENT(d % 2 == 0, "rope: head dimension must be even.");

    core::context().setDevice(in->deviceType(), in->deviceId());

    const auto *pos = reinterpret_cast<const int64_t *>(pos_ids->data());

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_kernel(reinterpret_cast<float *>(out->data()),
                           reinterpret_cast<const float *>(in->data()), pos, seq_len,
                           nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_kernel(reinterpret_cast<fp16_t *>(out->data()),
                           reinterpret_cast<const fp16_t *>(in->data()), pos, seq_len,
                           nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_kernel(reinterpret_cast<bf16_t *>(out->data()),
                           reinterpret_cast<const bf16_t *>(in->data()), pos, seq_len,
                           nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}

} // namespace llaisys::ops
