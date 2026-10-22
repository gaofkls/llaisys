#include "llaisys/ops.h"

#include "llaisys_tensor.hpp"

#include "api_guard.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/sample/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

namespace {
// Tensor handles crossing the C boundary are wrapped in LlaisysTensor; a null
// handle means "this optional argument was not supplied" (only linear's bias).
inline llaisys::tensor_t unwrap(llaisysTensor_t tensor) {
    return tensor ? tensor->tensor : nullptr;
}
} // namespace

__C {
    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        llaisys::api::run_void("llaisysAdd", [&]() {
            llaisys::ops::add(unwrap(c), unwrap(a), unwrap(b));
        });
    }
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        llaisys::api::run_void("llaisysArgmax", [&]() {
            llaisys::ops::argmax(unwrap(max_idx), unwrap(max_val), unwrap(vals));
        });
    }
    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        llaisys::api::run_void("llaisysEmbedding", [&]() {
            llaisys::ops::embedding(unwrap(out), unwrap(index), unwrap(weight));
        });
    }
    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        llaisys::api::run_void("llaisysLinear", [&]() {
            llaisys::ops::linear(unwrap(out), unwrap(in), unwrap(weight), unwrap(bias));
        });
    }
    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        llaisys::api::run_void("llaisysRearrange", [&]() {
            llaisys::ops::rearrange(unwrap(out), unwrap(in));
        });
    }
    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        llaisys::api::run_void("llaisysRmsNorm", [&]() {
            llaisys::ops::rms_norm(unwrap(out), unwrap(in), unwrap(weight), eps);
        });
    }
    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        llaisys::api::run_void("llaisysROPE", [&]() {
            llaisys::ops::rope(unwrap(out), unwrap(in), unwrap(pos_ids), theta);
        });
    }
    void llaisysSample(llaisysTensor_t token_id, llaisysTensor_t vals, float temperature,
                       size_t top_k, float top_p, uint64_t *rng_state) {
        llaisys::api::run_void("llaisysSample", [&]() {
            llaisys::ops::sample(unwrap(token_id), unwrap(vals), temperature, top_k, top_p,
                                 rng_state);
        });
    }
    void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
        llaisys::api::run_void("llaisysSelfAttention", [&]() {
            llaisys::ops::self_attention(unwrap(attn_val), unwrap(q), unwrap(k), unwrap(v), scale);
        });
    }
    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        llaisys::api::run_void("llaisysSwiGLU", [&]() {
            llaisys::ops::swiglu(unwrap(out), unwrap(gate), unwrap(up));
        });
    }
}
