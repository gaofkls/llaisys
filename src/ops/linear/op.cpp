#include "op.hpp"

#include "../op_common.hpp"
#include "../../utils.hpp"

#include <algorithm>

namespace llaisys::ops {

namespace {

// Tile sizes for the prefill GEMM. They live at namespace scope on purpose: MSVC
// rejects a function-local constexpr used as an array bound inside a lambda
// ("expression did not evaluate to a constant", C2131), because `[&]` makes the
// capture non-constant from its point of view.
constexpr size_t kTileRows = 4;
constexpr size_t kTileCols = 4;

// out[0, N] = in[0, K] @ weight[N, K]^T + bias[N]
//
// Decoding is a matrix-vector product, and it is memory bound: the weight matrix
// is streamed once. Walking the output features in order (instead of tiling the
// columns) keeps the weight access strictly sequential, so the hardware prefetcher
// does its job. Four accumulators break the FMA dependency chain.
template <typename T>
void gemv_kernel(T *out, const T *in, const T *weight, const T *bias, size_t K, size_t N) {
    parallel_for(N, 16, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const T *row = weight + i * K;
            float a0 = 0.0f;
            float a1 = 0.0f;
            float a2 = 0.0f;
            float a3 = 0.0f;
            size_t j = 0;
            for (; j + 4 <= K; j += 4) {
                a0 += to_float(in[j]) * to_float(row[j]);
                a1 += to_float(in[j + 1]) * to_float(row[j + 1]);
                a2 += to_float(in[j + 2]) * to_float(row[j + 2]);
                a3 += to_float(in[j + 3]) * to_float(row[j + 3]);
            }
            float sum = (a0 + a1) + (a2 + a3);
            for (; j < K; ++j) {
                sum += to_float(in[j]) * to_float(row[j]);
            }
            if (bias != nullptr) {
                sum += to_float(bias[i]);
            }
            out[i] = from_float<T>(sum);
        }
    });
}

// out[M, N] = in[M, K] @ weight[N, K]^T + bias[N]
//
// weight is stored row-major as [out_features, in_features], i.e. the transpose of
// the mathematical operand, so every output element is a dot product over K.
//
// The kernel walks 4x4 tiles of the output and keeps a float32 accumulator per
// element, which keeps the weight tile in registers across the K loop. Half
// precision inputs are converted on the fly -- never into a full sized temporary
// buffer, which is what made the previous version slower than a naive loop.
template <typename T>
void linear_kernel(T *out, const T *in, const T *weight, const T *bias, size_t M,
                   size_t K, size_t N) {
    const size_t mtiles = (M + kTileRows - 1) / kTileRows;
    const size_t ntiles = (N + kTileCols - 1) / kTileCols;

    auto compute_tile = [&](size_t m0, size_t n0) {
        const size_t rows = std::min(kTileRows, M - m0);
        const size_t cols = std::min(kTileCols, N - n0);
        float acc[kTileRows][kTileCols] = {};

        const T *x = in + m0 * K;
        for (size_t k = 0; k < K; ++k) {
            float xv[kTileRows];
            for (size_t r = 0; r < rows; ++r) {
                xv[r] = to_float(x[r * K + k]);
            }
            const T *w = weight + n0 * K + k;
            for (size_t c = 0; c < cols; ++c) {
                const float wv = to_float(w[c * K]);
                for (size_t r = 0; r < rows; ++r) {
                    acc[r][c] += xv[r] * wv;
                }
            }
        }

        for (size_t r = 0; r < rows; ++r) {
            T *orow = out + (m0 + r) * N + n0;
            for (size_t c = 0; c < cols; ++c) {
                float value = acc[r][c];
                if (bias != nullptr) {
                    value += to_float(bias[n0 + c]);
                }
                orow[c] = from_float<T>(value);
            }
        }
    };

    if (M <= kTileRows) {
        // Token-by-token decoding: a single row of activations, so parallelise
        // across output features instead (otherwise there is nothing to split).
        parallel_for(ntiles, 8, [&](size_t begin, size_t end) {
            for (size_t n0 = begin; n0 < end; ++n0) {
                compute_tile(0, n0 * kTileCols);
            }
        });
    } else {
        // Prefill: split the output rows; each worker streams the whole weight
        // matrix once, which keeps the activation tile hot in cache.
        parallel_for(mtiles, 1, [&](size_t begin, size_t end) {
            for (size_t m0 = begin; m0 < end; ++m0) {
                for (size_t n0 = 0; n0 < ntiles; ++n0) {
                    compute_tile(m0 * kTileRows, n0 * kTileCols);
                }
            }
        });
    }
}

template <typename T>
void linear_dispatch(T *out, const T *in, const T *weight, const T *bias, size_t M,
                     size_t K, size_t N) {
    if (M == 1) {
        gemv_kernel(out, in, weight, bias, K, N);
    } else {
        linear_kernel(out, in, weight, bias, M, K, N);
    }
}

} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_ARGUMENT(in->ndim() == 2, "linear: input must be 2-D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "linear: weight must be 2-D.");
    CHECK_ARGUMENT(out->ndim() == 2, "linear: output must be 2-D.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "linear: all tensors must be contiguous.");
    CHECK_ARGUMENT(is_supported_float(in->dtype()),
                   "linear: only f32/f16/bf16 are supported.");

    const size_t M = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t N = weight->shape()[0];

    CHECK_ARGUMENT(weight->shape()[1] == K,
                   "linear: weight in_features does not match the input.");
    CHECK_ARGUMENT(out->shape()[0] == M && out->shape()[1] == N,
                   "linear: output shape must be [M, out_features].");

    const void *bias_data = nullptr;
    if (bias != nullptr) {
        CHECK_SAME_DTYPE(bias->dtype(), in->dtype());
        CHECK_SAME_DEVICE(bias, in);
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == N,
                       "linear: bias must have shape [out_features].");
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous.");
        bias_data = bias->data();
    }

    core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_dispatch(reinterpret_cast<float *>(out->data()),
                             reinterpret_cast<const float *>(in->data()),
                             reinterpret_cast<const float *>(weight->data()),
                             reinterpret_cast<const float *>(bias_data), M, K, N);
    case LLAISYS_DTYPE_F16:
        return linear_dispatch(reinterpret_cast<fp16_t *>(out->data()),
                             reinterpret_cast<const fp16_t *>(in->data()),
                             reinterpret_cast<const fp16_t *>(weight->data()),
                             reinterpret_cast<const fp16_t *>(bias_data), M, K, N);
    case LLAISYS_DTYPE_BF16:
        return linear_dispatch(reinterpret_cast<bf16_t *>(out->data()),
                             reinterpret_cast<const bf16_t *>(in->data()),
                             reinterpret_cast<const bf16_t *>(weight->data()),
                             reinterpret_cast<const bf16_t *>(bias_data), M, K, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}

} // namespace llaisys::ops
