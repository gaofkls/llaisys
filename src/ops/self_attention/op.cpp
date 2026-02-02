#include "op.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 使用正确的张量API获取维度
    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();
    const auto& v_shape = v->shape();
    
    int seqlen = static_cast<int>(q_shape[0]);      // 查询序列长度
    int nhead = static_cast<int>(q_shape[1]);       // 查询头数
    int d = static_cast<int>(q_shape[2]);           // 查询维度
    int total_len = static_cast<int>(k_shape[0]);   // 总长度（包括缓存）
    int nkvhead = static_cast<int>(k_shape[1]);     // 键值头数
    int dv = static_cast<int>(v_shape[2]);          // 值维度
    
    // 获取数据指针 - 假设所有数据都被转换成float进行处理
    float* q_data = reinterpret_cast<float*>(q->data());
    float* k_data = reinterpret_cast<float*>(k->data());
    float* v_data = reinterpret_cast<float*>(v->data());
    float* attn_val_data = reinterpret_cast<float*>(attn_val->data());
    
    // 计算步长以便正确访问多维数组
    int q_stride_seq = nhead * d;      // 查询序列步长
    int q_stride_head = d;             // 查询头步长
    
    int k_stride_seq = nkvhead * d;    // 键序列步长
    int k_stride_head = d;             // 键头步长
    
    int v_stride_seq = nkvhead * dv;   // 值序列步长
    int v_stride_head = dv;            // 值头步长
    
    int attn_val_stride_seq = nhead * dv;  // 输出序列步长
    int attn_val_stride_head = dv;         // 输出头步长
    
    // 对于多查询注意力机制（当 nhead != nkvhead），需要处理头分组
    bool is_grouped_query = (nhead != nkvhead);
    
    for (int sq = 0; sq < seqlen; sq++) {  // 遍历查询序列
        for (int h = 0; h < nhead; h++) {  // 遍历查询头
            // 确定对应的KV头（用于GQA - 分组查询注意力）
            int kv_head = is_grouped_query ? h / (nhead / nkvhead) : h;
            
            // 临时数组存储这个查询位置和头的注意力分数
            std::vector<float> attn_scores(total_len);
            
            // 计算 Q[sq,h,:] * K^T[:,kv_head,:]
            for (int tk = 0; tk < total_len; tk++) {
                float score = 0.0f;
                
                // 计算 Q[sq,h,:] 和 K[tk,kv_head,:] 的点积
                for (int dim = 0; dim < d; dim++) {
                    // Q[sq, h, dim] 的线性索引
                    int q_idx = sq * q_stride_seq + h * q_stride_head + dim;
                    // K[tk, kv_head, dim] 的线性索引
                    int k_idx = tk * k_stride_seq + kv_head * k_stride_head + dim;
                    
                    float q_val = q_data[q_idx];
                    float k_val = k_data[k_idx];
                    score += q_val * k_val;
                }
                
                // 应用缩放因子
                attn_scores[tk] = score * scale;
            }
            
            // 应用因果掩码：屏蔽未来的位置
            // 假设键/值序列的后seqlen个位置是当前序列，前面的是缓存
            int current_seq_start = total_len - seqlen;  // 当前序列在KV中的起始位置
            
            for (int tk = 0; tk < total_len; tk++) {
                // 只对当前序列中的位置应用因果掩码
                if (tk >= current_seq_start) {
                    // tk在当前序列中的相对位置
                    int rel_kv_pos = tk - current_seq_start;
                    // sq是查询在当前序列中的位置
                    
                    // 如果kv位置在查询位置之后，掩码
                    if (rel_kv_pos > sq) {
                        attn_scores[tk] = -std::numeric_limits<float>::infinity();
                    }
                }
                // 对于缓存位置（tk < current_seq_start），允许访问（不掩码）
            }
            
            // 应用softmax获取注意力权重
            // 为了数值稳定性找到最大值
            float max_score = *std::max_element(attn_scores.begin(), attn_scores.end());
            
            // 计算指数和softmax归一化
            float sum_exp = 0.0f;
            for (int tk = 0; tk < total_len; tk++) {
                attn_scores[tk] = std::exp(attn_scores[tk] - max_score);
                sum_exp += attn_scores[tk];
            }
            
            // 防止除零
            if (sum_exp == 0.0f) {
                sum_exp = 1.0f;
            }
            
            // 归一化得到概率
            for (auto& val : attn_scores) {
                val /= sum_exp;
            }
            
            // 计算最终输出 Y = A * V
            for (int dim_v = 0; dim_v < dv; dim_v++) {
                float result = 0.0f;
                
                for (int tk = 0; tk < total_len; tk++) {
                    float weight = attn_scores[tk];
                    // V[tk, kv_head, dim_v] 的线性索引
                    int v_idx = tk * v_stride_seq + kv_head * v_stride_head + dim_v;
                    float v_val = v_data[v_idx];
                    result += weight * v_val;
                }
                
                // 存储到输出张量 [sq, h, dim_v]
                int out_idx = sq * attn_val_stride_seq + h * attn_val_stride_head + dim_v;
                attn_val_data[out_idx] = result;
            }
        }
    }
}

} // namespace llaisys::ops