#pragma once

#include "../../tensor/tensor.hpp"

#include <cstdint>

namespace llaisys::ops {

// Draws one token from `vals` (1-D logits) and writes it to `token_id` (int64,
// shape (1,)).
//
//   temperature <= 0  -> greedy (argmax), deterministic
//   top_k == 1        -> greedy
//   top_k == 0        -> no top-k truncation
//   0 < top_p < 1     -> nucleus truncation after top-k
//
// `rng_state` is a single 64-bit word of generator state owned by the caller;
// the kernel advances it in place, so consecutive calls produce different draws
// and replaying the same seed reproduces the same sequence.
void sample(tensor_t token_id, tensor_t vals, float temperature, size_t top_k, float top_p,
            uint64_t *rng_state);

} // namespace llaisys::ops
