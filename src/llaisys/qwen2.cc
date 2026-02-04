#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include <vector>
#include <memory>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

// 实际的模型实现结构体
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // KV-Cache 存储
    struct LayerCache {
        llaisysTensor_t k_cache;    // 存储原始K（未应用RoPE）
        llaisysTensor_t v_cache;    // 存储原始V
        size_t current_pos;         // 当前已缓存的位置
        
        LayerCache() : k_cache(nullptr), v_cache(nullptr), current_pos(0) {}
        
        // 移动构造函数
        LayerCache(LayerCache&& other) noexcept 
            : k_cache(other.k_cache), v_cache(other.v_cache), current_pos(other.current_pos) {
            other.k_cache = nullptr;
            other.v_cache = nullptr;
            other.current_pos = 0;
        }
        
        // 移动赋值运算符
        LayerCache& operator=(LayerCache&& other) noexcept {
            if (this != &other) {
                // 释放当前资源
                if (k_cache) tensorDestroy(k_cache);
                if (v_cache) tensorDestroy(v_cache);
                
                // 转移资源
                k_cache = other.k_cache;
                v_cache = other.v_cache;
                current_pos = other.current_pos;
                
                // 置空原对象
                other.k_cache = nullptr;
                other.v_cache = nullptr;
                other.current_pos = 0;
            }
            return *this;
        }
        
        ~LayerCache() {
            if (k_cache) tensorDestroy(k_cache);
            if (v_cache) tensorDestroy(v_cache);
        }
        
        // 明确删除拷贝构造函数和拷贝赋值运算符
        LayerCache(const LayerCache&) = delete;
        LayerCache& operator=(const LayerCache&) = delete;
    };
    
    std::vector<LayerCache> kv_cache;
    
    // 位置ID张量（预计算）
    llaisysTensor_t pos_ids;
    
    // 当前序列长度
    size_t current_seq_len;
    
    // 设备信息
    llaisysDeviceType_t device_type;
    int device_id;
    
    LlaisysQwen2Model(const LlaisysQwen2Meta& m, llaisysDeviceType_t dev_type, int* dev_ids, int ndevice)
        : meta(m), current_seq_len(0), device_type(dev_type), device_id(0), pos_ids(nullptr) {
        
        if (dev_ids && ndevice > 0) {
            device_id = dev_ids[0];
        }
        
        // 初始化权重结构
        memset(&weights, 0, sizeof(LlaisysQwen2Weights));
        
        // 为指针数组分配内存
        weights.attn_norm_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_q_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_q_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_k_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_k_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_v_w = new llaisysTensor_t[meta.nlayer];
        weights.attn_v_b = new llaisysTensor_t[meta.nlayer];
        weights.attn_o_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_norm_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_gate_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_up_w = new llaisysTensor_t[meta.nlayer];
        weights.mlp_down_w = new llaisysTensor_t[meta.nlayer];
        
        // 初始化指针为nullptr
        for (size_t i = 0; i < meta.nlayer; i++) {
            weights.attn_norm_w[i] = nullptr;
            weights.attn_q_w[i] = nullptr;
            weights.attn_q_b[i] = nullptr;
            weights.attn_k_w[i] = nullptr;
            weights.attn_k_b[i] = nullptr;
            weights.attn_v_w[i] = nullptr;
            weights.attn_v_b[i] = nullptr;
            weights.attn_o_w[i] = nullptr;
            weights.mlp_norm_w[i] = nullptr;
            weights.mlp_gate_w[i] = nullptr;
            weights.mlp_up_w[i] = nullptr;
            weights.mlp_down_w[i] = nullptr;
        }
        
        // 初始化位置ID张量
        size_t maxseq_shape[] = {meta.maxseq};
        pos_ids = tensorCreate(maxseq_shape, 1, LLAISYS_DTYPE_I64, device_type, device_id);
        
        if (pos_ids) {
            int64_t* pos_data = static_cast<int64_t*>(tensorGetData(pos_ids));
            for (size_t i = 0; i < meta.maxseq; i++) {
                pos_data[i] = i;
            }
        }
        
        // 初始化KV-Cache - 使用 reserve 和 emplace_back 避免拷贝
        kv_cache.reserve(meta.nlayer);
        for (size_t i = 0; i < meta.nlayer; i++) {
            kv_cache.emplace_back();  // 使用就地构造
        }
        
        // 然后为每个LayerCache创建张量
        size_t cache_shape[3] = {meta.maxseq, meta.nkvh, meta.dh};
        for (size_t i = 0; i < meta.nlayer; i++) {
            kv_cache[i].k_cache = tensorCreate(cache_shape, 3, meta.dtype, device_type, device_id);
            kv_cache[i].v_cache = tensorCreate(cache_shape, 3, meta.dtype, device_type, device_id);
        }
    }
    
    // 移动构造函数
    LlaisysQwen2Model(LlaisysQwen2Model&& other) noexcept 
        : meta(other.meta),
          weights(other.weights),
          kv_cache(std::move(other.kv_cache)),
          pos_ids(other.pos_ids),
          current_seq_len(other.current_seq_len),
          device_type(other.device_type),
          device_id(other.device_id) {
        
        // 转移所有权后置空原对象
        other.pos_ids = nullptr;
        memset(&other.weights, 0, sizeof(LlaisysQwen2Weights));
        other.kv_cache.clear();
    }
    
    // 删除拷贝构造函数和拷贝赋值运算符
    LlaisysQwen2Model(const LlaisysQwen2Model&) = delete;
    LlaisysQwen2Model& operator=(const LlaisysQwen2Model&) = delete;
    
    ~LlaisysQwen2Model() {
        // 释放权重指针数组
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
        
        // 释放位置ID张量
        if (pos_ids) tensorDestroy(pos_ids);
        
        // kv_cache会在vector析构时自动清理
    }
    
    // 更新KV-Cache
    void update_kv_cache(size_t layer_idx, llaisysTensor_t k_raw, llaisysTensor_t v_raw, size_t pos) {
        if (layer_idx >= kv_cache.size()) return;
        
        LayerCache& cache = kv_cache[layer_idx];
        if (!cache.k_cache || !cache.v_cache || !k_raw || !v_raw) return;
        
        // 获取数据指针
        void* k_cache_data = tensorGetData(cache.k_cache);
        void* v_cache_data = tensorGetData(cache.v_cache);
        void* k_raw_data = tensorGetData(k_raw);
        void* v_raw_data = tensorGetData(v_raw);
        
        if (!k_cache_data || !v_cache_data || !k_raw_data || !v_raw_data) return;
        
        // 计算要复制的数据量
        size_t kv_size = meta.nkvh * meta.dh;
        size_t elem_size = 0;
        
        switch (meta.dtype) {
            case LLAISYS_DTYPE_F32: elem_size = 4; break;
            case LLAISYS_DTYPE_F16: elem_size = 2; break;
            case LLAISYS_DTYPE_BF16: elem_size = 2; break;
            default: elem_size = 4; break;
        }
        
        // 计算目标位置
        size_t k_offset = pos * kv_size * elem_size;
        size_t v_offset = pos * kv_size * elem_size;
        
        // 复制数据
        memcpy(static_cast<char*>(k_cache_data) + k_offset, k_raw_data, kv_size * elem_size);
        memcpy(static_cast<char*>(v_cache_data) + v_offset, v_raw_data, kv_size * elem_size);
        
        // 更新当前位置
        cache.current_pos = pos + 1;
    }
    
    // 从缓存获取K,V并应用RoPE
    void get_kv_from_cache(size_t layer_idx, llaisysTensor_t& k_out, llaisysTensor_t& v_out, 
                          size_t start_pos, size_t seq_len) {
        if (layer_idx >= kv_cache.size()) {
            k_out = v_out = nullptr;
            return;
        }
        
        LayerCache& cache = kv_cache[layer_idx];
        if (!cache.k_cache || !cache.v_cache || cache.current_pos == 0) {
            k_out = v_out = nullptr;
            return;
        }
        
        // 实际可用的缓存长度
        size_t cache_len = std::min(cache.current_pos, start_pos + seq_len);
        size_t actual_len = cache_len - start_pos;
        
        if (actual_len == 0) {
            k_out = v_out = nullptr;
            return;
        }
        
        // 1. 从缓存切片获取原始K/V
        size_t kv_shape[3] = {actual_len, meta.nkvh, meta.dh};
        
        k_out = tensorSlice(cache.k_cache, 0, start_pos, cache_len);
        v_out = tensorSlice(cache.v_cache, 0, start_pos, cache_len);
        
        // reshape为3D
        k_out = tensorView(k_out, kv_shape, 3);
        v_out = tensorView(v_out, kv_shape, 3);
        
        // 2. 为RoPE创建位置ID
        size_t pos_shape[] = {actual_len};
        llaisysTensor_t pos_ids_slice = tensorCreate(pos_shape, 1, LLAISYS_DTYPE_I64, device_type, device_id);
        
        if (pos_ids_slice) {
            int64_t* pos_data = static_cast<int64_t*>(tensorGetData(pos_ids_slice));
            for (size_t i = 0; i < actual_len; i++) {
                pos_data[i] = start_pos + i;
            }
            
            // 3. 应用RoPE到K
            llaisysTensor_t k_with_rope = tensorCreate(kv_shape, 3, meta.dtype, device_type, device_id);
            llaisysROPE(k_with_rope, k_out, pos_ids_slice, meta.theta);
            
            // 替换为带RoPE的K
            tensorDestroy(k_out);
            k_out = k_with_rope;
            
            tensorDestroy(pos_ids_slice);
        }
        // 如果创建pos_ids失败，k_out保持原始K（未应用RoPE）
    }
};

// ==================== 完整注意力机制实现 ====================

// RMSNorm辅助函数
static void apply_rms_norm(llaisysTensor_t out, llaisysTensor_t in, 
                          llaisysTensor_t weight, float eps) {
    llaisysRmsNorm(out, in, weight, eps);
}

// 完整注意力层实现
static void apply_attention_layer(LlaisysQwen2Model* model, 
                                 llaisysTensor_t& hidden_states,
                                 size_t layer_idx,
                                 size_t seq_pos,
                                 size_t seq_len) {
    
    auto& cache = model->kv_cache[layer_idx];
    
    // 1. RMSNorm
    size_t hidden_shape[2];
    tensorGetShape(hidden_states, hidden_shape);
    
    llaisysTensor_t normed = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                         model->device_type, model->device_id);
    apply_rms_norm(normed, hidden_states, model->weights.attn_norm_w[layer_idx], 
                  model->meta.epsilon);
    
    // 2. 计算Q, K, V投影
    llaisysTensor_t q_proj = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                         model->device_type, model->device_id);
    llaisysTensor_t k_proj = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                         model->device_type, model->device_id);
    llaisysTensor_t v_proj = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                         model->device_type, model->device_id);
    
    // 线性投影
    llaisysLinear(q_proj, normed, model->weights.attn_q_w[layer_idx], 
                 model->weights.attn_q_b[layer_idx]);
    llaisysLinear(k_proj, normed, model->weights.attn_k_w[layer_idx],
                 model->weights.attn_k_b[layer_idx]);
    llaisysLinear(v_proj, normed, model->weights.attn_v_w[layer_idx],
                 model->weights.attn_v_b[layer_idx]);
    
    // 3. 重塑为多头格式
    // Q: [seq_len, num_heads, head_dim]
    // K, V: [seq_len, num_kv_heads, head_dim]
    size_t q_3d_shape[3] = {seq_len, model->meta.nh, model->meta.dh};
    size_t kv_3d_shape[3] = {seq_len, model->meta.nkvh, model->meta.dh};
    
    llaisysTensor_t q_3d = tensorView(q_proj, q_3d_shape, 3);
    llaisysTensor_t k_raw_3d = tensorView(k_proj, kv_3d_shape, 3);
    llaisysTensor_t v_raw_3d = tensorView(v_proj, kv_3d_shape, 3);
    
    // 4. 应用RoPE到Q（对于新token）
    size_t pos_shape[] = {seq_len};
    llaisysTensor_t pos_ids = tensorCreate(pos_shape, 1, LLAISYS_DTYPE_I64,
                                          model->device_type, model->device_id);
    
    int64_t* pos_data = static_cast<int64_t*>(tensorGetData(pos_ids));
    for (size_t i = 0; i < seq_len; i++) {
        pos_data[i] = seq_pos + i;
    }
    
    llaisysTensor_t q_rope = tensorCreate(q_3d_shape, 3, model->meta.dtype,
                                         model->device_type, model->device_id);
    llaisysROPE(q_rope, q_3d, pos_ids, model->meta.theta);
    
    // 5. 更新KV-Cache（存储原始K，未应用RoPE）
    if (seq_pos >= cache.current_pos) {
        // 对于多token，只更新最后一个
        if (seq_len == 1) {
            model->update_kv_cache(layer_idx, k_raw_3d, v_raw_3d, seq_pos);
        } else {
            // 取最后一个token
            llaisysTensor_t k_last = tensorSlice(k_raw_3d, 0, seq_len - 1, seq_len);
            llaisysTensor_t v_last = tensorSlice(v_raw_3d, 0, seq_len - 1, seq_len);
            
            size_t slice_shape[3] = {1, model->meta.nkvh, model->meta.dh};
            k_last = tensorView(k_last, slice_shape, 3);
            v_last = tensorView(v_last, slice_shape, 3);
            
            model->update_kv_cache(layer_idx, k_last, v_last, seq_pos + seq_len - 1);
            
            tensorDestroy(k_last);
            tensorDestroy(v_last);
        }
    }
    
    // 6. 准备用于注意力计算的K和V
    llaisysTensor_t k_for_attn = nullptr;
    llaisysTensor_t v_for_attn = nullptr;
    
    // 总序列长度 = 缓存长度 + 新token长度
    size_t total_seq_len = cache.current_pos + seq_len;
    
    if (cache.current_pos > 0) {
        // 有缓存：从缓存获取所有K/V（包括应用RoPE）
        model->get_kv_from_cache(layer_idx, k_for_attn, v_for_attn, 0, cache.current_pos);
        
        // 需要将新token的K/V与缓存的K/V拼接
        if (k_for_attn && v_for_attn && seq_len > 0) {
            // 应用RoPE到新token的K
            llaisysTensor_t k_new_rope = tensorCreate(kv_3d_shape, 3, model->meta.dtype,
                                                     model->device_type, model->device_id);
            llaisysROPE(k_new_rope, k_raw_3d, pos_ids, model->meta.theta);
            
            // TODO: 这里需要张量拼接操作
            // 简化：如果只有一个新token，且缓存是完整的，直接使用缓存的
            // 对于多个新token，需要更复杂的逻辑
            
            tensorDestroy(k_new_rope);
        }
    } else {
        // 无缓存：使用新token的K/V，并应用RoPE到K
        k_for_attn = tensorCreate(kv_3d_shape, 3, model->meta.dtype,
                                 model->device_type, model->device_id);
        llaisysROPE(k_for_attn, k_raw_3d, pos_ids, model->meta.theta);
        v_for_attn = v_raw_3d;  // V直接使用
    }
    
    // 7. 计算注意力
    float scale = 1.0f / sqrtf(static_cast<float>(model->meta.dh));
    
    // 注意力输出形状: [seq_len, num_heads, head_dim]
    size_t attn_3d_shape[3] = {seq_len, model->meta.nh, model->meta.dh};
    llaisysTensor_t attn_output_3d = tensorCreate(attn_3d_shape, 3, model->meta.dtype,
                                                 model->device_type, model->device_id);
    
    // 调用self-attention
    // 注意：如果使用缓存，K/V的序列长度会大于Q的序列长度
    // self-attention内部会处理因果掩码
    llaisysSelfAttention(attn_output_3d, q_rope, k_for_attn, v_for_attn, scale);
    
    // 8. 重塑回2D: [seq_len, hidden_size]
    size_t attn_out_2d_shape[2] = {seq_len, model->meta.hs};
    llaisysTensor_t attn_output = tensorView(attn_output_3d, attn_out_2d_shape, 2);
    
    // 9. 输出投影
    llaisysTensor_t proj_output = tensorCreate(attn_out_2d_shape, 2, model->meta.dtype,
                                              model->device_type, model->device_id);
    llaisysLinear(proj_output, attn_output, model->weights.attn_o_w[layer_idx], nullptr);
    
    // 10. 残差连接
    llaisysTensor_t new_hidden = tensorCreate(attn_out_2d_shape, 2, model->meta.dtype,
                                             model->device_type, model->device_id);
    llaisysAdd(new_hidden, hidden_states, proj_output);
    
    // 11. 清理
    tensorDestroy(hidden_states);
    hidden_states = new_hidden;
    
    // 释放中间张量
    tensorDestroy(normed);
    tensorDestroy(q_proj);
    tensorDestroy(k_proj);
    tensorDestroy(v_proj);
    tensorDestroy(q_3d);
    tensorDestroy(k_raw_3d);
    tensorDestroy(v_raw_3d);
    tensorDestroy(pos_ids);
    tensorDestroy(q_rope);
    if (k_for_attn && k_for_attn != k_raw_3d) tensorDestroy(k_for_attn);
    if (v_for_attn && v_for_attn != v_raw_3d) tensorDestroy(v_for_attn);
    tensorDestroy(attn_output_3d);
    tensorDestroy(attn_output);
    tensorDestroy(proj_output);
}

// MLP层实现
static void apply_mlp_layer(LlaisysQwen2Model* model,
                           llaisysTensor_t& hidden_states,
                           size_t layer_idx,
                           size_t seq_len) {
    
    // 1. RMSNorm
    size_t hidden_shape[2];
    tensorGetShape(hidden_states, hidden_shape);
    
    llaisysTensor_t normed = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                         model->device_type, model->device_id);
    apply_rms_norm(normed, hidden_states, model->weights.mlp_norm_w[layer_idx],
                  model->meta.epsilon);
    
    // 2. Gate和Up投影
    llaisysTensor_t gate_proj = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                            model->device_type, model->device_id);
    llaisysTensor_t up_proj = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                          model->device_type, model->device_id);
    
    llaisysLinear(gate_proj, normed, model->weights.mlp_gate_w[layer_idx], nullptr);
    llaisysLinear(up_proj, normed, model->weights.mlp_up_w[layer_idx], nullptr);
    
    // 3. SwiGLU激活
    llaisysTensor_t activated = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                            model->device_type, model->device_id);
    llaisysSwiGLU(activated, gate_proj, up_proj);
    
    // 4. Down投影
    llaisysTensor_t down_proj = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                            model->device_type, model->device_id);
    llaisysLinear(down_proj, activated, model->weights.mlp_down_w[layer_idx], nullptr);
    
    // 5. 残差连接
    llaisysTensor_t new_hidden = tensorCreate(hidden_shape, 2, model->meta.dtype,
                                             model->device_type, model->device_id);
    llaisysAdd(new_hidden, hidden_states, down_proj);
    
    // 6. 清理
    tensorDestroy(hidden_states);
    hidden_states = new_hidden;
    
    tensorDestroy(normed);
    tensorDestroy(gate_proj);
    tensorDestroy(up_proj);
    tensorDestroy(activated);
    tensorDestroy(down_proj);
}

// ==================== C API 实现 ====================

// 创建模型实例
__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta, 
    llaisysDeviceType_t device, 
    int *device_ids, 
    int ndevice) {
    
    if (!meta) {
        std::cerr << "Error: meta is null in llaisysQwen2ModelCreate" << std::endl;
        return nullptr;
    }
    
    try {
        return new LlaisysQwen2Model(*meta, device, device_ids, ndevice);
    } catch (const std::exception& e) {
        std::cerr << "Error creating Qwen2 model: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Unknown error creating Qwen2 model" << std::endl;
        return nullptr;
    }
}

// 销毁模型
__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    if (model) {
        delete model;
    }
}

// 获取权重结构
__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
    if (!model) {
        std::cerr << "Error: model is null in llaisysQwen2ModelWeights" << std::endl;
        return nullptr;
    }
    return &(model->weights);
}

// 推理函数 - 完整实现
__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    if (!model || !token_ids || ntoken == 0) {
        std::cerr << "Error: invalid input to llaisysQwen2ModelInfer" << std::endl;
        return -1;
    }
    
    // 确定要处理的新token数量
    size_t start_pos = model->current_seq_len;
    size_t new_tokens = ntoken;
    
    if (start_pos > 0 && start_pos < ntoken) {
        // 部分token已经处理过
        new_tokens = ntoken - start_pos;
        token_ids = &token_ids[start_pos];
    } else if (start_pos >= ntoken) {
        // 所有token都已经处理过
        std::cout << "[Infer] All tokens already processed" << std::endl;
        return 0; // 简化处理
    }
    
    int64_t next_token = 0;
    
    // 处理新的token（一次处理一个，支持KV-Cache）
    for (size_t pos_offset = 0; pos_offset < new_tokens; pos_offset++) {
        size_t absolute_pos = start_pos + pos_offset;
        int64_t token_id = token_ids[pos_offset];
        
        // 1. 嵌入查找
        size_t index_shape[] = {1};
        llaisysTensor_t index_tensor = tensorCreate(index_shape, 1, LLAISYS_DTYPE_I64,
                                                   model->device_type, model->device_id);
        
        if (!index_tensor) {
            std::cerr << "Error: Failed to create index tensor" << std::endl;
            return -1;
        }
        
        int64_t* index_data = static_cast<int64_t*>(tensorGetData(index_tensor));
        index_data[0] = token_id;
        
        size_t embed_shape[] = {1, model->meta.hs};
        llaisysTensor_t hidden = tensorCreate(embed_shape, 2, model->meta.dtype,
                                             model->device_type, model->device_id);
        
        if (!hidden) {
            std::cerr << "Error: Failed to create hidden tensor" << std::endl;
            tensorDestroy(index_tensor);
            return -1;
        }
        
        if (!model->weights.in_embed) {
            std::cerr << "Error: in_embed weight not loaded" << std::endl;
            tensorDestroy(index_tensor);
            tensorDestroy(hidden);
            return -1;
        }
        
        llaisysEmbedding(hidden, index_tensor, model->weights.in_embed);
        tensorDestroy(index_tensor);
        
        // 2. 通过所有Transformer层
        for (size_t layer = 0; layer < model->meta.nlayer; layer++) {
            // 检查权重是否加载
            if (!model->weights.attn_norm_w[layer] || !model->weights.attn_q_w[layer] ||
                !model->weights.attn_k_w[layer] || !model->weights.attn_v_w[layer] ||
                !model->weights.attn_o_w[layer] || !model->weights.mlp_norm_w[layer] ||
                !model->weights.mlp_gate_w[layer] || !model->weights.mlp_up_w[layer] ||
                !model->weights.mlp_down_w[layer]) {
                std::cerr << "Error: weights not loaded for layer " << layer << std::endl;
                tensorDestroy(hidden);
                return -1;
            }
            
            // 完整注意力层
            apply_attention_layer(model, hidden, layer, absolute_pos, 1);
            
            // MLP层
            apply_mlp_layer(model, hidden, layer, 1);
        }
        
        // 3. 最终RMSNorm
        size_t final_shape[] = {1, model->meta.hs};
        llaisysTensor_t final_norm = tensorCreate(final_shape, 2, model->meta.dtype,
                                                 model->device_type, model->device_id);
        
        if (!final_norm) {
            std::cerr << "Error: Failed to create final_norm tensor" << std::endl;
            tensorDestroy(hidden);
            return -1;
        }
        
        if (!model->weights.out_norm_w) {
            std::cerr << "Error: out_norm_w not loaded" << std::endl;
            tensorDestroy(hidden);
            tensorDestroy(final_norm);
            return -1;
        }
        
        apply_rms_norm(final_norm, hidden, model->weights.out_norm_w, model->meta.epsilon);
        tensorDestroy(hidden);
        
        // 4. 输出投影（LM Head）
        size_t logits_shape[] = {1, model->meta.voc};
        llaisysTensor_t logits = tensorCreate(logits_shape, 2, model->meta.dtype,
                                             model->device_type, model->device_id);
        
        if (!logits) {
            std::cerr << "Error: Failed to create logits tensor" << std::endl;
            tensorDestroy(final_norm);
            return -1;
        }
        
        if (!model->weights.out_embed) {
            std::cerr << "Error: out_embed not loaded" << std::endl;
            tensorDestroy(final_norm);
            tensorDestroy(logits);
            return -1;
        }
        
        llaisysLinear(logits, final_norm, model->weights.out_embed, nullptr);
        tensorDestroy(final_norm);
        
        // 5. 使用argmax采样
        size_t max_shape[] = {1};
        llaisysTensor_t max_idx = tensorCreate(max_shape, 1, LLAISYS_DTYPE_I64,
                                              model->device_type, model->device_id);
        llaisysTensor_t max_val = tensorCreate(max_shape, 1, model->meta.dtype,
                                              model->device_type, model->device_id);
        
        if (!max_idx || !max_val) {
            std::cerr << "Error: Failed to create max_idx/max_val tensors" << std::endl;
            tensorDestroy(logits);
            if (max_idx) tensorDestroy(max_idx);
            if (max_val) tensorDestroy(max_val);
            return -1;
        }
        
        llaisysArgmax(max_idx, max_val, logits);
        
        // 获取下一个token
        int64_t* next_token_ptr = static_cast<int64_t*>(tensorGetData(max_idx));
        next_token = next_token_ptr[0];
        
        // 清理
        tensorDestroy(logits);
        tensorDestroy(max_idx);
        tensorDestroy(max_val);
        
        // 更新模型状态
        model->current_seq_len = absolute_pos + 1;
    }
    
    return next_token;
}

// 重置模型状态
__export void llaisysQwen2ModelReset(struct LlaisysQwen2Model * model) {
    if (!model) return;
    
    // 重置序列长度
    model->current_seq_len = 0;
    
    // 重置KV-Cache位置
    for (auto& cache : model->kv_cache) {
        cache.current_pos = 0;
    }
}