#include "llaisys/models/qwen2.h"

#include "api_guard.hpp"
#include "llaisys_tensor.hpp"

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
#include "../tensor/copy.hpp"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using llaisys::tensor_t;

enum class WeightKind {
    Unknown,
    InEmbed,
    OutEmbed,
    OutNorm,
    AttnNorm,
    QWeight,
    QBias,
    KWeight,
    KBias,
    VWeight,
    VBias,
    OWeight,
    MlpNorm,
    GateWeight,
    UpWeight,
    DownWeight,
};

struct WeightRef {
    WeightKind kind = WeightKind::Unknown;
    size_t layer = 0;
};

tensor_t unwrap(llaisysTensor_t handle) {
    return handle ? handle->tensor : nullptr;
}

tensor_t make(const std::vector<size_t> &shape, llaisysDataType_t dtype,
              llaisysDeviceType_t device, int device_id) {
    return llaisys::Tensor::create(shape, dtype, device, device_id);
}

// Reads one int64 back to the host, wherever the tensor lives.
int64_t read_i64(const tensor_t &tensor) {
    int64_t value = -1;
    if (tensor->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(&value, tensor->data(), sizeof(int64_t));
    } else {
        llaisys::copy_run(reinterpret_cast<std::byte *>(&value), tensor->data(),
                          sizeof(int64_t), LLAISYS_DEVICE_CPU, tensor->deviceType(),
                          tensor->deviceId());
    }
    return value;
}

WeightRef parse_weight_name(const std::string &name) {
    WeightRef ref;
    if (name == "model.embed_tokens.weight" || name == "tok_embeddings.weight") {
        ref.kind = WeightKind::InEmbed;
        return ref;
    }
    if (name == "lm_head.weight" || name == "output.weight") {
        ref.kind = WeightKind::OutEmbed;
        return ref;
    }
    if (name == "model.norm.weight" || name == "norm.weight") {
        ref.kind = WeightKind::OutNorm;
        return ref;
    }

    const std::string prefix = "model.layers.";
    if (name.rfind(prefix, 0) != 0) {
        return ref;
    }

    const size_t layer_begin = prefix.size();
    const size_t dot = name.find('.', layer_begin);
    if (dot == std::string::npos) {
        return ref;
    }
    try {
        ref.layer = static_cast<size_t>(std::stoul(name.substr(layer_begin, dot - layer_begin)));
    } catch (...) {
        return ref;
    }

    const std::string field = name.substr(dot + 1);
    if (field == "input_layernorm.weight") {
        ref.kind = WeightKind::AttnNorm;
    } else if (field == "post_attention_layernorm.weight") {
        ref.kind = WeightKind::MlpNorm;
    } else if (field == "self_attn.q_proj.weight") {
        ref.kind = WeightKind::QWeight;
    } else if (field == "self_attn.q_proj.bias") {
        ref.kind = WeightKind::QBias;
    } else if (field == "self_attn.k_proj.weight") {
        ref.kind = WeightKind::KWeight;
    } else if (field == "self_attn.k_proj.bias") {
        ref.kind = WeightKind::KBias;
    } else if (field == "self_attn.v_proj.weight") {
        ref.kind = WeightKind::VWeight;
    } else if (field == "self_attn.v_proj.bias") {
        ref.kind = WeightKind::VBias;
    } else if (field == "self_attn.o_proj.weight") {
        ref.kind = WeightKind::OWeight;
    } else if (field == "mlp.gate_proj.weight") {
        ref.kind = WeightKind::GateWeight;
    } else if (field == "mlp.up_proj.weight") {
        ref.kind = WeightKind::UpWeight;
    } else if (field == "mlp.down_proj.weight") {
        ref.kind = WeightKind::DownWeight;
    }
    return ref;
}

float read_as_float(llaisysDataType_t dtype, const std::byte *base, size_t index) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return reinterpret_cast<const float *>(base)[index];
    case LLAISYS_DTYPE_F16:
        return llaisys::utils::cast<float>(
            reinterpret_cast<const llaisys::fp16_t *>(base)[index]);
    case LLAISYS_DTYPE_BF16:
        return llaisys::utils::cast<float>(
            reinterpret_cast<const llaisys::bf16_t *>(base)[index]);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void write_from_float(llaisysDataType_t dtype, std::byte *base, size_t index, float value) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        reinterpret_cast<float *>(base)[index] = value;
        return;
    case LLAISYS_DTYPE_F16:
        reinterpret_cast<llaisys::fp16_t *>(base)[index] =
            llaisys::utils::cast<llaisys::fp16_t>(value);
        return;
    case LLAISYS_DTYPE_BF16:
        reinterpret_cast<llaisys::bf16_t *>(base)[index] =
            llaisys::utils::cast<llaisys::bf16_t>(value);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

// The concrete model, private to the C++ side. Every intermediate tensor is a
// shared_ptr<Tensor>, so scratch buffers are released automatically even when an
// op throws halfway through a layer.
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};
    llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    struct LayerCache {
        llaisysTensor_t k = nullptr; // keys *after* RoPE, shape [maxseq, nkvh, dh]
        llaisysTensor_t v = nullptr; // values, shape [maxseq, nkvh, dh]
    };
    std::vector<LayerCache> kv_cache;

    size_t cached_tokens = 0;
    size_t kv_capacity = 0;

    static constexpr size_t kInitialCacheLen = 256;

    LlaisysQwen2Model(const LlaisysQwen2Meta &m, llaisysDeviceType_t device, int *device_ids,
                      int ndevice)
        : meta(m), device_type(device), device_id(0) {
        if (device_ids != nullptr && ndevice > 0) {
            device_id = device_ids[0];
        }

        CHECK_ARGUMENT(meta.nlayer > 0 && meta.hs > 0 && meta.nh > 0 && meta.nkvh > 0
                           && meta.dh > 0 && meta.di > 0 && meta.maxseq > 0 && meta.voc > 0,
                       "qwen2: invalid model meta.");
        CHECK_ARGUMENT(meta.nh * meta.dh == meta.hs,
                       "qwen2: nh * dh must equal hidden_size.");
        CHECK_ARGUMENT(meta.nh % meta.nkvh == 0,
                       "qwen2: num_attention_heads must be a multiple of num_kv_heads.");

        weights.attn_norm_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_q_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_q_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_k_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_w = new llaisysTensor_t[meta.nlayer]();
        weights.attn_v_b = new llaisysTensor_t[meta.nlayer]();
        weights.attn_o_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_up_w = new llaisysTensor_t[meta.nlayer]();
        weights.mlp_down_w = new llaisysTensor_t[meta.nlayer]();

        // Reserve a small cache and grow it on demand. Allocating
        // max_position_embeddings up front would cost several GB for a 1.5B model
        // (131072 positions * 2 kv heads * 128 dims * 2 bytes * 2 tensors * 28 layers).
        kv_cache.resize(meta.nlayer);
        kv_capacity = std::min<size_t>(meta.maxseq, kInitialCacheLen);
        const std::vector<size_t> cache_shape = {kv_capacity, meta.nkvh, meta.dh};
        for (size_t i = 0; i < meta.nlayer; ++i) {
            kv_cache[i].k = wrap(make(cache_shape, meta.dtype, device_type, device_id));
            kv_cache[i].v = wrap(make(cache_shape, meta.dtype, device_type, device_id));
        }
    }

    ~LlaisysQwen2Model() {
        if (weights.attn_norm_w != nullptr) {
            for (size_t i = 0; i < meta.nlayer; ++i) {
                destroy(weights.attn_norm_w[i]);
                destroy(weights.attn_q_w[i]);
                destroy(weights.attn_q_b[i]);
                destroy(weights.attn_k_w[i]);
                destroy(weights.attn_k_b[i]);
                destroy(weights.attn_v_w[i]);
                destroy(weights.attn_v_b[i]);
                destroy(weights.attn_o_w[i]);
                destroy(weights.mlp_norm_w[i]);
                destroy(weights.mlp_gate_w[i]);
                destroy(weights.mlp_up_w[i]);
                destroy(weights.mlp_down_w[i]);
            }
        }
        destroy(weights.in_embed);
        destroy(weights.out_embed);
        destroy(weights.out_norm_w);

        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;
        delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;
        delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;

        for (auto &cache : kv_cache) {
            destroy(cache.k);
            destroy(cache.v);
        }
    }

    LlaisysQwen2Model(const LlaisysQwen2Model &) = delete;
    LlaisysQwen2Model &operator=(const LlaisysQwen2Model &) = delete;

    static llaisysTensor_t wrap(const tensor_t &tensor) { return new LlaisysTensor{tensor}; }

    static void destroy(llaisysTensor_t &handle) {
        delete handle;
        handle = nullptr;
    }

    // ---------------------------------------------------------------- weights --

    llaisysTensor_t *slot_for(WeightKind kind, size_t layer) {
        switch (kind) {
        case WeightKind::InEmbed:
            return &weights.in_embed;
        case WeightKind::OutEmbed:
            return &weights.out_embed;
        case WeightKind::OutNorm:
            return &weights.out_norm_w;
        case WeightKind::AttnNorm:
            return &weights.attn_norm_w[layer];
        case WeightKind::QWeight:
            return &weights.attn_q_w[layer];
        case WeightKind::QBias:
            return &weights.attn_q_b[layer];
        case WeightKind::KWeight:
            return &weights.attn_k_w[layer];
        case WeightKind::KBias:
            return &weights.attn_k_b[layer];
        case WeightKind::VWeight:
            return &weights.attn_v_w[layer];
        case WeightKind::VBias:
            return &weights.attn_v_b[layer];
        case WeightKind::OWeight:
            return &weights.attn_o_w[layer];
        case WeightKind::MlpNorm:
            return &weights.mlp_norm_w[layer];
        case WeightKind::GateWeight:
            return &weights.mlp_gate_w[layer];
        case WeightKind::UpWeight:
            return &weights.mlp_up_w[layer];
        case WeightKind::DownWeight:
            return &weights.mlp_down_w[layer];
        default:
            return nullptr;
        }
    }

    std::vector<size_t> expected_shape(WeightKind kind) const {
        const size_t q_dim = meta.nh * meta.dh;
        const size_t kv_dim = meta.nkvh * meta.dh;
        switch (kind) {
        case WeightKind::InEmbed:
        case WeightKind::OutEmbed:
            return {meta.voc, meta.hs};
        case WeightKind::OutNorm:
        case WeightKind::AttnNorm:
        case WeightKind::MlpNorm:
            return {meta.hs};
        case WeightKind::QWeight:
            return {q_dim, meta.hs};
        case WeightKind::QBias:
            return {q_dim};
        case WeightKind::KWeight:
        case WeightKind::VWeight:
            return {kv_dim, meta.hs};
        case WeightKind::KBias:
        case WeightKind::VBias:
            return {kv_dim};
        case WeightKind::OWeight:
            return {meta.hs, q_dim};
        case WeightKind::GateWeight:
        case WeightKind::UpWeight:
            return {meta.di, meta.hs};
        case WeightKind::DownWeight:
            return {meta.hs, meta.di};
        default:
            return {};
        }
    }

    bool load_weight(const std::string &name, const void *data, const std::vector<size_t> &shape,
                     llaisysDataType_t src_dtype) {
        const WeightRef ref = parse_weight_name(name);
        if (ref.kind == WeightKind::Unknown) {
            std::cerr << "[qwen2] unknown weight name: " << name << std::endl;
            return false;
        }
        if (ref.kind != WeightKind::InEmbed && ref.kind != WeightKind::OutEmbed
            && ref.kind != WeightKind::OutNorm && ref.layer >= meta.nlayer) {
            std::cerr << "[qwen2] layer index out of range for " << name << std::endl;
            return false;
        }

        llaisysTensor_t *slot = slot_for(ref.kind, ref.layer);
        if (slot == nullptr) {
            return false;
        }

        const std::vector<size_t> expected = expected_shape(ref.kind);
        if (shape != expected) {
            std::cerr << "[qwen2] shape mismatch for " << name << ": got [";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cerr << shape[i] << (i + 1 < shape.size() ? ", " : "");
            }
            std::cerr << "], expected [";
            for (size_t i = 0; i < expected.size(); ++i) {
                std::cerr << expected[i] << (i + 1 < expected.size() ? ", " : "");
            }
            std::cerr << "]" << std::endl;
            return false;
        }

        size_t numel = 1;
        for (size_t dim : shape) {
            numel *= dim;
        }

        tensor_t tensor = make(shape, meta.dtype, device_type, device_id);

        if (src_dtype == meta.dtype) {
            // Fast path: identical layout, one bulk copy (H2D on GPU).
            llaisys::copy_run(tensor->data(), static_cast<const std::byte *>(data),
                              numel * tensor->elementSize(), tensor->deviceType(),
                              LLAISYS_DEVICE_CPU, device_id);
        } else {
            // Slow path, load time only: convert element by element on the host.
            std::vector<std::byte> converted(numel * tensor->elementSize());
            const auto *src = static_cast<const std::byte *>(data);
            for (size_t i = 0; i < numel; ++i) {
                write_from_float(meta.dtype, converted.data(), i,
                                 read_as_float(src_dtype, src, i));
            }
            llaisys::copy_run(tensor->data(), converted.data(), converted.size(),
                              tensor->deviceType(), LLAISYS_DEVICE_CPU, device_id);
        }

        destroy(*slot);
        *slot = wrap(tensor);
        return true;
    }

    // --------------------------------------------------------------- forward --

    void check_layer_weights(size_t layer) const {
        const bool ok = weights.attn_norm_w[layer] && weights.attn_q_w[layer]
            && weights.attn_k_w[layer] && weights.attn_v_w[layer] && weights.attn_o_w[layer]
            && weights.mlp_norm_w[layer] && weights.mlp_gate_w[layer]
            && weights.mlp_up_w[layer] && weights.mlp_down_w[layer];
        CHECK_ARGUMENT(ok, "qwen2: some layer weights were never loaded.");
    }

    // Grows the KV cache by doubling until it can hold `needed` positions.
    void ensure_capacity(size_t needed) {
        if (needed <= kv_capacity) {
            return;
        }

        size_t new_capacity = (kv_capacity == 0) ? kInitialCacheLen : kv_capacity;
        while (new_capacity < needed) {
            new_capacity *= 2;
        }
        new_capacity = std::min(new_capacity, meta.maxseq);
        CHECK_ARGUMENT(needed <= new_capacity,
                       "qwen2: sequence is longer than max_position_embeddings.");

        const size_t keep = std::min(cached_tokens, new_capacity);
        for (auto &cache : kv_cache) {
            tensor_t old_k = unwrap(cache.k);
            tensor_t old_v = unwrap(cache.v);

            tensor_t new_k =
                make({new_capacity, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
            tensor_t new_v =
                make({new_capacity, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);

            if (keep > 0) {
                llaisys::ops::rearrange(new_k->slice(0, 0, keep), old_k->slice(0, 0, keep));
                llaisys::ops::rearrange(new_v->slice(0, 0, keep), old_v->slice(0, 0, keep));
            }

            destroy(cache.k);
            destroy(cache.v);
            cache.k = wrap(new_k);
            cache.v = wrap(new_v);
        }

        kv_capacity = new_capacity;
    }

    // Copies `count` freshly computed keys/values into the cache at [start, start+count).
    void cache_store(llaisysTensor_t cache, size_t start, size_t count,
                     const tensor_t &block) {
        tensor_t dst = unwrap(cache)->slice(0, start, start + count);
        llaisys::ops::rearrange(dst, block);
    }

    // Runs the forward pass and returns the next token. `temperature <= 0` or
    // `top_k == 1` selects greedy decoding; otherwise the logits go through the
    // sample operator, which advances `rng_state`.
    int64_t infer(const int64_t *token_ids, size_t ntoken, float temperature, size_t top_k,
                  float top_p, uint64_t *rng_state) {
        CHECK_ARGUMENT(token_ids != nullptr && ntoken > 0, "qwen2: empty input.");

        // Continue from the cached prefix. If the caller re-sent tokens we have
        // already processed, rewind one token so this call still produces a
        // prediction (recomputing a token is idempotent: RoPE uses the position).
        size_t start = cached_tokens;
        if (start >= ntoken) {
            start = ntoken - 1;
        }
        const size_t seq = ntoken - start;
        ensure_capacity(start + seq);

        const size_t hs = meta.hs;
        const size_t dh = meta.dh;
        const size_t nh = meta.nh;
        const size_t nkvh = meta.nkvh;
        const size_t kv_dim = nkvh * dh;

        CHECK_ARGUMENT(weights.in_embed && weights.out_embed && weights.out_norm_w,
                       "qwen2: model weights were never loaded.");

        // 1. Token ids -> embedding.
        tensor_t index = make({seq}, LLAISYS_DTYPE_I64, device_type, device_id);
        index->load(token_ids + start);

        tensor_t hidden = make({seq, hs}, meta.dtype, device_type, device_id);
        llaisys::ops::embedding(hidden, index, unwrap(weights.in_embed));

        // 2. Positions of the new tokens.
        std::vector<int64_t> pos_host(seq);
        for (size_t i = 0; i < seq; ++i) {
            pos_host[i] = static_cast<int64_t>(start + i);
        }
        tensor_t pos_ids = make({seq}, LLAISYS_DTYPE_I64, device_type, device_id);
        pos_ids->load(pos_host.data());

        const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            check_layer_weights(layer);

            // --- attention block ---
            tensor_t normed = make({seq, hs}, meta.dtype, device_type, device_id);
            llaisys::ops::rms_norm(normed, hidden, unwrap(weights.attn_norm_w[layer]),
                                   meta.epsilon);

            tensor_t q = make({seq, hs}, meta.dtype, device_type, device_id);
            llaisys::ops::linear(q, normed, unwrap(weights.attn_q_w[layer]),
                                 unwrap(weights.attn_q_b[layer]));

            tensor_t k = make({seq, kv_dim}, meta.dtype, device_type, device_id);
            llaisys::ops::linear(k, normed, unwrap(weights.attn_k_w[layer]),
                                 unwrap(weights.attn_k_b[layer]));

            tensor_t v = make({seq, kv_dim}, meta.dtype, device_type, device_id);
            llaisys::ops::linear(v, normed, unwrap(weights.attn_v_w[layer]),
                                 unwrap(weights.attn_v_b[layer]));

            // [seq, heads * head_dim] -> [seq, heads, head_dim]
            tensor_t q3 = q->view({seq, nh, dh});
            tensor_t k3 = k->view({seq, nkvh, dh});
            tensor_t v3 = v->view({seq, nkvh, dh});

            tensor_t q_rot = make({seq, nh, dh}, meta.dtype, device_type, device_id);
            llaisys::ops::rope(q_rot, q3, pos_ids, meta.theta);

            tensor_t k_rot = make({seq, nkvh, dh}, meta.dtype, device_type, device_id);
            llaisys::ops::rope(k_rot, k3, pos_ids, meta.theta);

            // Cache stores post-RoPE keys: RoPE only ever depends on the absolute
            // position, so there is no reason to redo it on every step.
            const size_t total = start + seq;
            cache_store(kv_cache[layer].k, start, seq, k_rot);
            cache_store(kv_cache[layer].v, start, seq, v3);

            tensor_t k_all = unwrap(kv_cache[layer].k)->slice(0, 0, total);
            tensor_t v_all = unwrap(kv_cache[layer].v)->slice(0, 0, total);

            tensor_t attn = make({seq, nh, dh}, meta.dtype, device_type, device_id);
            llaisys::ops::self_attention(attn, q_rot, k_all, v_all, scale);

            tensor_t attn2d = attn->view({seq, hs});
            tensor_t projected = make({seq, hs}, meta.dtype, device_type, device_id);
            llaisys::ops::linear(projected, attn2d, unwrap(weights.attn_o_w[layer]), nullptr);

            tensor_t hidden_attn = make({seq, hs}, meta.dtype, device_type, device_id);
            llaisys::ops::add(hidden_attn, hidden, projected);
            hidden = hidden_attn;

            // --- mlp block ---
            tensor_t normed2 = make({seq, hs}, meta.dtype, device_type, device_id);
            llaisys::ops::rms_norm(normed2, hidden, unwrap(weights.mlp_norm_w[layer]),
                                   meta.epsilon);

            tensor_t gate = make({seq, meta.di}, meta.dtype, device_type, device_id);
            llaisys::ops::linear(gate, normed2, unwrap(weights.mlp_gate_w[layer]), nullptr);

            tensor_t up = make({seq, meta.di}, meta.dtype, device_type, device_id);
            llaisys::ops::linear(up, normed2, unwrap(weights.mlp_up_w[layer]), nullptr);

            tensor_t act = make({seq, meta.di}, meta.dtype, device_type, device_id);
            llaisys::ops::swiglu(act, gate, up);

            tensor_t down = make({seq, hs}, meta.dtype, device_type, device_id);
            llaisys::ops::linear(down, act, unwrap(weights.mlp_down_w[layer]), nullptr);

            tensor_t hidden_mlp = make({seq, hs}, meta.dtype, device_type, device_id);
            llaisys::ops::add(hidden_mlp, hidden, down);
            hidden = hidden_mlp;
        }

        // 3. Final norm, then only the last row needs to reach the vocabulary.
        tensor_t final_norm = make({seq, hs}, meta.dtype, device_type, device_id);
        llaisys::ops::rms_norm(final_norm, hidden, unwrap(weights.out_norm_w), meta.epsilon);

        tensor_t last = final_norm->slice(0, seq - 1, seq);

        tensor_t logits = make({1, meta.voc}, meta.dtype, device_type, device_id);
        llaisys::ops::linear(logits, last, unwrap(weights.out_embed), nullptr);

        tensor_t logits1d = logits->view({meta.voc});
        tensor_t token = make({1}, LLAISYS_DTYPE_I64, device_type, device_id);
        if (top_k == 1 || !(temperature > 0.0f)) {
            tensor_t max_val = make({1}, meta.dtype, device_type, device_id);
            llaisys::ops::argmax(token, max_val, logits1d);
        } else {
            llaisys::ops::sample(token, logits1d, temperature, top_k, top_p, rng_state);
        }

        cached_tokens = start + seq;
        return read_i64(token);
    }

    void reset() {
        cached_tokens = 0;
    }
};

// ============================== C API ========================================

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                           llaisysDeviceType_t device,
                                                           int *device_ids, int ndevice) {
    return llaisys::api::run("llaisysQwen2ModelCreate",
                             static_cast<LlaisysQwen2Model *>(nullptr), [&]() {
                                 CHECK_ARGUMENT(meta != nullptr, "qwen2: meta is null.");
                                 return new LlaisysQwen2Model(*meta, device, device_ids, ndevice);
                             });
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    llaisys::api::run_void("llaisysQwen2ModelDestroy", [&]() { delete model; });
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    return llaisys::api::run("llaisysQwen2ModelWeights",
                             static_cast<LlaisysQwen2Weights *>(nullptr),
                             [&]() { return model ? &model->weights : nullptr; });
}

__export bool llaisysQwen2ModelLoadWeight(struct LlaisysQwen2Model *model,
                                          const char *weight_name, const void *data,
                                          const size_t *shape, size_t ndim, int dtype) {
    return llaisys::api::run("llaisysQwen2ModelLoadWeight", false, [&]() {
        CHECK_ARGUMENT(model != nullptr, "qwen2: model is null.");
        CHECK_ARGUMENT(weight_name != nullptr && data != nullptr, "qwen2: null weight.");
        CHECK_ARGUMENT(shape != nullptr || ndim == 0, "qwen2: null shape.");

        const std::vector<size_t> shape_vec(shape, shape + ndim);
        return model->load_weight(weight_name, data, shape_vec,
                                  static_cast<llaisysDataType_t>(dtype));
    });
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids,
                                        size_t ntoken) {
    return llaisys::api::run("llaisysQwen2ModelInfer", static_cast<int64_t>(-1), [&]() {
        CHECK_ARGUMENT(model != nullptr, "qwen2: model is null.");
        return model->infer(token_ids, ntoken, 0.0f, 1, 1.0f, nullptr);
    });
}

__export int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model *model,
                                              int64_t *token_ids, size_t ntoken,
                                              float temperature, size_t top_k, float top_p,
                                              uint64_t *rng_state) {
    return llaisys::api::run("llaisysQwen2ModelInferSample", static_cast<int64_t>(-1), [&]() {
        CHECK_ARGUMENT(model != nullptr, "qwen2: model is null.");
        CHECK_ARGUMENT(rng_state != nullptr, "qwen2: sampling needs a generator state.");
        return model->infer(token_ids, ntoken, temperature, top_k, top_p, rng_state);
    });
}

__export void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
    llaisys::api::run_void("llaisysQwen2ModelReset", [&]() {
        if (model != nullptr) {
            model->reset();
        }
    });
}
