#include "op.hpp"

#include "../op_common.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops {

namespace {

// splitmix64. A single 64-bit state word is small enough to be owned by the
// caller and carried across the C boundary; std::mt19937_64 would need a 2.5 kB
// state block and its output is not identical across standard libraries.
inline uint64_t next_bits(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Uniform in [0, 1) with 24 bits of resolution: enough for any realistic
// vocabulary, and exactly representable as a float.
inline float next_float(uint64_t *state) {
    return static_cast<float>(next_bits(state) >> 40) * (1.0f / 16777216.0f);
}

uint64_t *fallback_state() {
    // Only used when the caller passes no state pointer. Seeded once per thread
    // from the clock, so separate runs still differ.
    thread_local uint64_t state = 0;
    if (state == 0) {
        state = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())
                | 1ull;
    }
    return &state;
}

// Score lives next to its index so sorting never chases a second array - that
// indirection is what made a full-vocabulary sort twice as slow as it needed to
// be.
struct Candidate {
    float score;
    int32_t index;
};

// Descending score, ties broken by the lower index so the result is stable.
inline bool preferred(const Candidate &a, const Candidate &b) {
    if (a.score != b.score) {
        return a.score > b.score;
    }
    return a.index < b.index;
}

// A decode step reuses these across tokens; nothing is allocated after warmup.
thread_local std::vector<Candidate> candidates;
thread_local std::vector<float> weights;

} // namespace

void sample(tensor_t token_id, tensor_t vals, float temperature, size_t top_k, float top_p,
            uint64_t *rng_state) {
    CHECK_SAME_DEVICE(token_id, vals);
    CHECK_ARGUMENT(token_id->dtype() == LLAISYS_DTYPE_I64,
                   "sample: token_id must be int64.");
    CHECK_ARGUMENT(token_id->ndim() == 1 && token_id->numel() == 1,
                   "sample: token_id must have shape (1,).");
    CHECK_ARGUMENT(vals->ndim() == 1, "sample: vals must be 1D logits.");
    CHECK_ARGUMENT(is_supported_float(vals->dtype()),
                   "sample: only f32/f16/bf16 logits are supported.");
    ASSERT(vals->isContiguous() && token_id->isContiguous(),
           "sample: all tensors must be contiguous.");

    core::context().setDevice(vals->deviceType(), vals->deviceId());

    const size_t n = vals->numel();
    CHECK_ARGUMENT(n > 0, "sample: vals must not be empty.");

    int64_t *out = reinterpret_cast<int64_t *>(token_id->data());

    candidates.resize(n);
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const auto *src = reinterpret_cast<const float *>(vals->data());
        for (size_t i = 0; i < n; ++i) {
            candidates[i] = {src[i], static_cast<int32_t>(i)};
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const auto *src = reinterpret_cast<const fp16_t *>(vals->data());
        for (size_t i = 0; i < n; ++i) {
            candidates[i] = {to_float(src[i]), static_cast<int32_t>(i)};
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const auto *src = reinterpret_cast<const bf16_t *>(vals->data());
        for (size_t i = 0; i < n; ++i) {
            candidates[i] = {to_float(src[i]), static_cast<int32_t>(i)};
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }

    // Greedy whenever the distribution degenerates to a single candidate.
    // temperature <= 0 is treated as greedy instead of dividing by zero, which
    // is what inference stacks do in practice.
    const bool nucleus = top_p > 0.0f && top_p < 1.0f;
    const bool greedy = n == 1 || top_k == 1 || !(temperature > 0.0f)
                        || !std::isfinite(temperature);

    if (greedy) {
        size_t best = 0;
        for (size_t i = 1; i < n; ++i) {
            if (preferred(candidates[i], candidates[best])) {
                best = i;
            }
        }
        *out = static_cast<int64_t>(candidates[best].index);
        return;
    }

    // 1. Shift by the max for numerical stability, then apply temperature.
    float max_val = candidates[0].score;
    for (size_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, candidates[i].score);
    }
    const float inv_temperature = 1.0f / temperature;
    for (size_t i = 0; i < n; ++i) {
        candidates[i].score = (candidates[i].score - max_val) * inv_temperature;
    }

    // 2. Candidate set: the top-k logits, or the whole vocabulary.
    size_t keep = n;
    if (top_k > 1 && top_k < n) {
        keep = top_k;
    }
    if (keep < n) {
        // O(n) partial selection; only the first `keep` entries end up correct,
        // which is all we need unless nucleus truncation forces an ordering.
        std::nth_element(candidates.begin(), candidates.begin() + static_cast<ptrdiff_t>(keep),
                         candidates.end(), preferred);
    }
    candidates.resize(keep);
    if (nucleus) {
        // Sorting `keep` entries is cheap when top_k bounded the set. Without a
        // top_k this sorts the whole vocabulary; pass one to avoid that.
        std::sort(candidates.begin(), candidates.end(), preferred);
    }

    // 3. Softmax over the candidate set.
    weights.resize(keep);
    float total = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        const float w = std::exp(candidates[i].score);
        weights[i] = w;
        total += w;
    }
    if (!(total > 0.0f) || !std::isfinite(total)) {
        // Every candidate underflowed; return the best rather than dividing by zero.
        *out = static_cast<int64_t>(candidates[0].index);
        return;
    }

    // 4. Nucleus truncation: keep the shortest prefix whose mass reaches top_p.
    size_t cut = keep;
    if (nucleus) {
        float cumulative = 0.0f;
        cut = keep;
        for (size_t i = 0; i < keep; ++i) {
            cumulative += weights[i] / total;
            if (cumulative >= top_p) {
                cut = i + 1;
                break;
            }
        }
        if (cut == 0) {
            cut = 1;
        }
        total = 0.0f;
        for (size_t i = 0; i < cut; ++i) {
            total += weights[i];
        }
        if (!(total > 0.0f)) {
            *out = static_cast<int64_t>(candidates[0].index);
            return;
        }
    }

    // 5. Inverse transform sampling over the surviving mass.
    uint64_t *state = rng_state != nullptr ? rng_state : fallback_state();
    if (*state == 0) {
        *state = 0x853C49E6748FEA9Bull; // never let the generator stay at zero
    }
    const float target = next_float(state) * total;

    size_t chosen = cut - 1;
    float accumulated = 0.0f;
    for (size_t i = 0; i < cut; ++i) {
        accumulated += weights[i];
        if (target < accumulated) {
            chosen = i;
            break;
        }
    }

    *out = static_cast<int64_t>(candidates[chosen].index);
}

} // namespace llaisys::ops
