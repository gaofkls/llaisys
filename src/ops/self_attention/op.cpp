#include "op.hpp"

#include "../op_common.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace llaisys::ops {

namespace {

// attn_val[L, nh, dv] = causal_softmax(q[L, nh, dh] @ k[S, nkvh, dh]^T * scale) @ v[S, nkvh, dv]
//
// k / v may be longer than q (KV cache): the causal mask is aligned to the right,
// so query position l attends to key positions [0, l + (S - L)].
// GQA / MQA is handled by mapping nh query heads onto nkvh key/value heads.
template <typename T>
void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v, size_t qlen,
                           size_t kvlen, size_t nh, size_t nkvh, size_t dh, size_t dv,
                           float scale) {
    const size_t group = (nh >= nkvh) ? (nh / nkvh) : 1;
    const size_t offset = kvlen - qlen;

    parallel_for(nh * qlen, 1, [&](size_t begin, size_t end) {
        std::vector<float> score(kvlen);
        std::vector<float> out_buf(dv);

        for (size_t idx = begin; idx < end; ++idx) {
            const size_t h = idx / qlen;
            const size_t l = idx % qlen;
            const size_t kvh = std::min(h / group, nkvh - 1);

            const T *qv = q + (l * nh + h) * dh;
            const size_t limit = l + offset;

            float max_score = -INFINITY;
            for (size_t t = 0; t <= limit; ++t) {
                const T *kv = k + (t * nkvh + kvh) * dh;
                float acc = 0.0f;
                for (size_t d = 0; d < dh; ++d) {
                    acc += to_float(qv[d]) * to_float(kv[d]);
                }
                acc *= scale;
                score[t] = acc;
                max_score = std::max(max_score, acc);
            }

            float sum = 0.0f;
            for (size_t t = 0; t <= limit; ++t) {
                score[t] = std::exp(score[t] - max_score);
                sum += score[t];
            }
            const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

            std::fill(out_buf.begin(), out_buf.end(), 0.0f);
            for (size_t t = 0; t <= limit; ++t) {
                const float w = score[t] * inv_sum;
                if (w == 0.0f) {
                    continue;
                }
                const T *vv = v + (t * nkvh + kvh) * dv;
                for (size_t d = 0; d < dv; ++d) {
                    out_buf[d] += w * to_float(vv[d]);
                }
            }

            T *ov = attn_val + (l * nh + h) * dv;
            for (size_t d = 0; d < dv; ++d) {
                ov[d] = from_float<T>(out_buf[d]);
            }
        }
    });
}

} // namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_ARGUMENT(q->ndim() == 3, "self_attention: q must have shape [qlen, nh, dh].");
    CHECK_ARGUMENT(k->ndim() == 3, "self_attention: k must have shape [kvlen, nkvh, dh].");
    CHECK_ARGUMENT(v->ndim() == 3, "self_attention: v must have shape [kvlen, nkvh, dv].");
    CHECK_ARGUMENT(attn_val->ndim() == 3,
                   "self_attention: output must have shape [qlen, nh, dv].");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous()
               && v->isContiguous(),
           "self_attention: all tensors must be contiguous.");
    CHECK_ARGUMENT(is_supported_float(q->dtype()),
                   "self_attention: only f32/f16/bf16 are supported.");

    const size_t qlen = q->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t dh = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh = k->shape()[1];
    const size_t dv = v->shape()[2];

    CHECK_ARGUMENT(k->shape()[2] == dh, "self_attention: q/k head dim mismatch.");
    CHECK_ARGUMENT(v->shape()[0] == kvlen && v->shape()[1] == nkvh,
                   "self_attention: k/v length or head mismatch.");
    CHECK_ARGUMENT(kvlen >= qlen, "self_attention: kv length must be >= q length.");
    CHECK_ARGUMENT(nh % nkvh == 0, "self_attention: nh must be a multiple of nkvh.");
    CHECK_ARGUMENT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh
                       && attn_val->shape()[2] == dv,
                   "self_attention: output shape mismatch.");

    core::context().setDevice(q->deviceType(), q->deviceId());

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_kernel(
            reinterpret_cast<float *>(attn_val->data()), reinterpret_cast<const float *>(q->data()),
            reinterpret_cast<const float *>(k->data()), reinterpret_cast<const float *>(v->data()),
            qlen, kvlen, nh, nkvh, dh, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_kernel(
            reinterpret_cast<fp16_t *>(attn_val->data()), reinterpret_cast<const fp16_t *>(q->data()),
            reinterpret_cast<const fp16_t *>(k->data()), reinterpret_cast<const fp16_t *>(v->data()),
            qlen, kvlen, nh, nkvh, dh, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_kernel(
            reinterpret_cast<bf16_t *>(attn_val->data()), reinterpret_cast<const bf16_t *>(q->data()),
            reinterpret_cast<const bf16_t *>(k->data()), reinterpret_cast<const bf16_t *>(v->data()),
            qlen, kvlen, nh, nkvh, dh, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}

} // namespace llaisys::ops
