#include "op.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 获取维度信息
    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();
    const auto& v_shape = v->shape();
    
    int qlen = static_cast<int>(q_shape[0]);      // 查询序列长度
    int nh = static_cast<int>(q_shape[1]);        // 查询头数
    int hd = static_cast<int>(q_shape[2]);        // 查询维度
    int kvlen = static_cast<int>(k_shape[0]);     // KV序列长度
    int nkvh = static_cast<int>(k_shape[1]);      // KV头数
    int dv = static_cast<int>(v_shape[2]);        // 值维度
    
    // 检查数据类型并分别处理
    switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32: {
            float* q_data = reinterpret_cast<float*>(q->data());
            float* k_data = reinterpret_cast<float*>(k->data());
            float* v_data = reinterpret_cast<float*>(v->data());
            float* attn_val_data = reinterpret_cast<float*>(attn_val->data());
            
            // 计算步长（对应原始维度 (seq, head, dim)）
            int q_stride_seq = nh * hd;      // 查询序列步长
            int q_stride_head = hd;          // 查询头步长
            int k_stride_seq = nkvh * hd;    // 键序列步长
            int k_stride_head = hd;          // 键头步长
            int v_stride_seq = nkvh * dv;    // 值序列步长
            int v_stride_head = dv;          // 值头步长
            int attn_val_stride_seq = nh * dv;  // 输出序列步长
            int attn_val_stride_head = dv;      // 输出头步长
            
            // 处理每个查询头
            for (int h = 0; h < nh; h++) {   
                // 计算对应的KV头（如果是GQA/MQA）
                int kv_head = (nh != nkvh) ? h / (nh / nkvh) : h;
                
                // 处理每个查询位置
                for (int sq = 0; sq < qlen; sq++) {  
                    // 计算注意力分数
                    std::vector<float> attn_scores(kvlen);
                    
                    for (int tk = 0; tk < kvlen; tk++) {
                        float score = 0.0f;
                        
                        // 计算 Q[sq,h,:] 和 K[tk,kv_head,:] 的点积
                        for (int dim = 0; dim < hd; dim++) {
                            int q_idx = sq * q_stride_seq + h * q_stride_head + dim;
                            int k_idx = tk * k_stride_seq + kv_head * k_stride_head + dim;
                            
                            float q_val = q_data[q_idx];
                            float k_val = k_data[k_idx];
                            score += q_val * k_val;
                        }
                        
                        attn_scores[tk] = score * scale;
                    }
                    
                    // 应用因果掩码
                    // PyTorch: L=qlen, S=kvlen, diagonal=S-L=kvlen-qlen
                    // tril(diagonal) 保留满足 tk <= sq + (kvlen - qlen) 的元素
                    for (int tk = 0; tk < kvlen; tk++) {
                        if (tk > sq + kvlen - qlen) {
                            attn_scores[tk] = -std::numeric_limits<float>::infinity();
                        }
                    }
                    
                    // 应用softmax
                    float max_score = *std::max_element(attn_scores.begin(), attn_scores.end());
                    
                    float sum_exp = 0.0f;
                    for (int tk = 0; tk < kvlen; tk++) {
                        if (std::isfinite(attn_scores[tk])) {
                            attn_scores[tk] = std::exp(attn_scores[tk] - max_score);
                            sum_exp += attn_scores[tk];
                        } else {
                            attn_scores[tk] = 0.0f;  // -inf becomes 0 after exp
                        }
                    }
                    
                    // 归一化，防止除以0
                    if (sum_exp > 1e-8f) {
                        for (auto& val : attn_scores) {
                            val /= sum_exp;
                        }
                    } else {
                        // 如果所有值都被掩码，均匀分配权重
                        for (auto& val : attn_scores) {
                            val = 1.0f / kvlen;
                        }
                    }
                    
                    // 计算输出
                    for (int dim_v = 0; dim_v < dv; dim_v++) {
                        float result = 0.0f;
                        
                        for (int tk = 0; tk < kvlen; tk++) {
                            float weight = attn_scores[tk];
                            int v_idx = tk * v_stride_seq + kv_head * v_stride_head + dim_v;
                            float v_val = v_data[v_idx];
                            result += weight * v_val;
                        }
                        
                        // 存储到输出 [sq, h, dim_v]
                        int out_idx = sq * attn_val_stride_seq + h * attn_val_stride_head + dim_v;
                        attn_val_data[out_idx] = result;
                    }
                }
            }
            break;
        }
        case LLAISYS_DTYPE_F16: {
            // 对于f16类型，使用更高精度的中间计算
            llaisys::fp16_t* q_data = reinterpret_cast<llaisys::fp16_t*>(q->data());
            llaisys::fp16_t* k_data = reinterpret_cast<llaisys::fp16_t*>(k->data());
            llaisys::fp16_t* v_data = reinterpret_cast<llaisys::fp16_t*>(v->data());
            llaisys::fp16_t* attn_val_data = reinterpret_cast<llaisys::fp16_t*>(attn_val->data());
            
            // 计算步长（对应原始维度 (seq, head, dim)）
            int q_stride_seq = nh * hd;      // 查询序列步长
            int q_stride_head = hd;          // 查询头步长
            int k_stride_seq = nkvh * hd;    // 键序列步长
            int k_stride_head = hd;          // 键头步长
            int v_stride_seq = nkvh * dv;    // 值序列步长
            int v_stride_head = dv;          // 值头步长
            int attn_val_stride_seq = nh * dv;  // 输出序列步长
            int attn_val_stride_head = dv;      // 输出头步长
            
            // 处理每个查询头
            for (int h = 0; h < nh; h++) {   
                // 计算对应的KV头（如果是GQA/MQA）
                int kv_head = (nh != nkvh) ? h / (nh / nkvh) : h;
                
                // 处理每个查询位置
                for (int sq = 0; sq < qlen; sq++) {  
                    // 使用双精度进行中间计算以提高精度
                    std::vector<double> raw_scores(kvlen);
                    
                    for (int tk = 0; tk < kvlen; tk++) {
                        double score = 0.0;
                        
                        // 计算 Q[sq,h,:] 和 K[tk,kv_head,:] 的点积
                        for (int dim = 0; dim < hd; dim++) {
                            int q_idx = sq * q_stride_seq + h * q_stride_head + dim;
                            int k_idx = tk * k_stride_seq + kv_head * k_stride_head + dim;
                            
                            double q_val = static_cast<double>(llaisys::utils::cast<float>(q_data[q_idx]));
                            double k_val = static_cast<double>(llaisys::utils::cast<float>(k_data[k_idx]));
                            score += q_val * k_val;
                        }
                        
                        raw_scores[tk] = score * static_cast<double>(scale);
                    }
                    
                    // 应用因果掩码
                    for (int tk = 0; tk < kvlen; tk++) {
                        if (tk > sq + kvlen - qlen) {
                            raw_scores[tk] = -std::numeric_limits<double>::infinity();
                        }
                    }
                    
                    // 应用softmax - 使用双精度提高精度
                    double max_score = *std::max_element(raw_scores.begin(), raw_scores.end());
                    
                    std::vector<double> exp_scores(kvlen);
                    double sum_exp = 0.0;
                    for (int tk = 0; tk < kvlen; tk++) {
                        if (std::isfinite(raw_scores[tk])) {
                            exp_scores[tk] = std::exp(raw_scores[tk] - max_score);
                            sum_exp += exp_scores[tk];
                        } else {
                            exp_scores[tk] = 0.0;  // -inf becomes 0 after exp
                        }
                    }
                    
                    // 归一化，防止除以0
                    std::vector<float> attn_weights(kvlen);
                    if (sum_exp > 1e-8) {
                        for (int tk = 0; tk < kvlen; tk++) {
                            attn_weights[tk] = static_cast<float>(exp_scores[tk] / sum_exp);
                        }
                    } else {
                        // 如果所有值都被掩码，均匀分配权重
                        for (int tk = 0; tk < kvlen; tk++) {
                            attn_weights[tk] = 1.0f / kvlen;
                        }
                    }
                    
                    // 计算输出
                    for (int dim_v = 0; dim_v < dv; dim_v++) {
                        double result = 0.0;  // 使用双精度累积
                        
                        for (int tk = 0; tk < kvlen; tk++) {
                            double weight = static_cast<double>(attn_weights[tk]);
                            int v_idx = tk * v_stride_seq + kv_head * v_stride_head + dim_v;
                            double v_val = static_cast<double>(llaisys::utils::cast<float>(v_data[v_idx]));
                            result += weight * v_val;
                        }
                        
                        // 存储到输出 [sq, h, dim_v]
                        int out_idx = sq * attn_val_stride_seq + h * attn_val_stride_head + dim_v;
                        attn_val_data[out_idx] = llaisys::utils::cast<llaisys::fp16_t>(static_cast<float>(result));
                    }
                }
            }
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            // 对于bf16类型，使用更高精度的中间计算
            llaisys::bf16_t* q_data = reinterpret_cast<llaisys::bf16_t*>(q->data());
            llaisys::bf16_t* k_data = reinterpret_cast<llaisys::bf16_t*>(k->data());
            llaisys::bf16_t* v_data = reinterpret_cast<llaisys::bf16_t*>(v->data());
            llaisys::bf16_t* attn_val_data = reinterpret_cast<llaisys::bf16_t*>(attn_val->data());
            
            // 计算步长（对应原始维度 (seq, head, dim)）
            int q_stride_seq = nh * hd;      // 查询序列步长
            int q_stride_head = hd;          // 查询头步长
            int k_stride_seq = nkvh * hd;    // 键序列步长
            int k_stride_head = hd;          // 键头步长
            int v_stride_seq = nkvh * dv;    // 值序列步长
            int v_stride_head = dv;          // 值头步长
            int attn_val_stride_seq = nh * dv;  // 输出序列步长
            int attn_val_stride_head = dv;      // 输出头步长
            
            // 处理每个查询头
            for (int h = 0; h < nh; h++) {   
                // 计算对应的KV头（如果是GQA/MQA）
                int kv_head = (nh != nkvh) ? h / (nh / nkvh) : h;
                
                // 处理每个查询位置
                for (int sq = 0; sq < qlen; sq++) {  
                    // 使用双精度进行中间计算以提高精度
                    std::vector<double> raw_scores(kvlen);
                    
                    for (int tk = 0; tk < kvlen; tk++) {
                        double score = 0.0;
                        
                        // 计算 Q[sq,h,:] 和 K[tk,kv_head,:] 的点积
                        for (int dim = 0; dim < hd; dim++) {
                            int q_idx = sq * q_stride_seq + h * q_stride_head + dim;
                            int k_idx = tk * k_stride_seq + kv_head * k_stride_head + dim;
                            
                            double q_val = static_cast<double>(llaisys::utils::cast<float>(q_data[q_idx]));
                            double k_val = static_cast<double>(llaisys::utils::cast<float>(k_data[k_idx]));
                            score += q_val * k_val;
                        }
                        
                        raw_scores[tk] = score * static_cast<double>(scale);
                    }
                    
                    // 应用因果掩码
                    for (int tk = 0; tk < kvlen; tk++) {
                        if (tk > sq + kvlen - qlen) {
                            raw_scores[tk] = -std::numeric_limits<double>::infinity();
                        }
                    }
                    
                    // 应用softmax - 使用双精度提高精度
                    double max_score = *std::max_element(raw_scores.begin(), raw_scores.end());
                    
                    std::vector<double> exp_scores(kvlen);
                    double sum_exp = 0.0;
                    for (int tk = 0; tk < kvlen; tk++) {
                        if (std::isfinite(raw_scores[tk])) {
                            exp_scores[tk] = std::exp(raw_scores[tk] - max_score);
                            sum_exp += exp_scores[tk];
                        } else {
                            exp_scores[tk] = 0.0;  // -inf becomes 0 after exp
                        }
                    }
                    
                    // 归一化，防止除以0
                    std::vector<float> attn_weights(kvlen);
                    if (sum_exp > 1e-8) {
                        for (int tk = 0; tk < kvlen; tk++) {
                            attn_weights[tk] = static_cast<float>(exp_scores[tk] / sum_exp);
                        }
                    } else {
                        // 如果所有值都被掩码，均匀分配权重
                        for (int tk = 0; tk < kvlen; tk++) {
                            attn_weights[tk] = 1.0f / kvlen;
                        }
                    }
                    
                    // 计算输出
                    for (int dim_v = 0; dim_v < dv; dim_v++) {
                        double result = 0.0;  // 使用双精度累积
                        
                        for (int tk = 0; tk < kvlen; tk++) {
                            double weight = static_cast<double>(attn_weights[tk]);
                            int v_idx = tk * v_stride_seq + kv_head * v_stride_head + dim_v;
                            double v_val = static_cast<double>(llaisys::utils::cast<float>(v_data[v_idx]));
                            result += weight * v_val;
                        }
                        
                        // 存储到输出 [sq, h, dim_v]
                        int out_idx = sq * attn_val_stride_seq + h * attn_val_stride_head + dim_v;
                        attn_val_data[out_idx] = llaisys::utils::cast<llaisys::bf16_t>(static_cast<float>(result));
                    }
                }
            }
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}

} // namespace llaisys::ops