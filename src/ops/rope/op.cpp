#include "op.hpp"
#include <cmath>
#include <vector>
#include "../../tensor/tensor.hpp"
namespace llaisys::ops {

// 前向声明模板函数
template<typename T>
void rope_impl(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, 
               size_t seqlen, size_t nhead, size_t d, size_t half_d);

// 特化版本处理 float
template<>
void rope_impl<float>(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, 
                      size_t seqlen, size_t nhead, size_t d, size_t half_d);

// 特化版本处理 fp16_t
template<>
void rope_impl<fp16_t>(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, 
                       size_t seqlen, size_t nhead, size_t d, size_t half_d);

// 特化版本处理 bf16_t  
template<>
void rope_impl<bf16_t>(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, 
                       size_t seqlen, size_t nhead, size_t d, size_t half_d);

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 检查输入输出形状是否匹配
    if (out->shape() != in->shape()) {
        throw std::runtime_error("Output and input shapes must match");
    }
    
    // 检查数据类型
    if (out->dtype() != in->dtype()) {
        throw std::runtime_error("Output and input data types must match");
    }
    
    // 检查pos_ids的数据类型
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("pos_ids must be int64");
    }
    
    // 获取张量信息
    const auto& out_shape = out->shape();
    const auto& in_shape = in->shape();
    const auto& pos_shape = pos_ids->shape();
    
    // 检查维度数
    if (in_shape.size() != 3) {
        throw std::runtime_error("Input tensor must have shape [seqlen, nhead, d]");
    }
    
    size_t seqlen = in_shape[0];
    size_t nhead = in_shape[1];
    size_t d = in_shape[2];
    size_t half_d = d / 2;
    
    // 检查维度是否有效
    if (d % 2 != 0) {
        throw std::runtime_error("Dimension d must be even");
    }
    
    // 检查pos_ids的形状
    if (pos_shape.size() != 1 || pos_shape[0] != seqlen) {
        throw std::runtime_error("pos_ids must have shape [seqlen]");
    }
    
    // 检查张量是否连续
    if (!out->isContiguous() || !in->isContiguous() || !pos_ids->isContiguous()) {
        throw std::runtime_error("All tensors must be contiguous");
    }
    
    // 根据数据类型进行处理
    switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            rope_impl<float>(out, in, pos_ids, theta, seqlen, nhead, d, half_d);
            break;
        case LLAISYS_DTYPE_F16:
            rope_impl<fp16_t>(out, in, pos_ids, theta, seqlen, nhead, d, half_d);
            break;
        case LLAISYS_DTYPE_BF16:
            rope_impl<bf16_t>(out, in, pos_ids, theta, seqlen, nhead, d, half_d);
            break;
        default:
            throw std::runtime_error("RoPE only supports float32, float16, and bfloat16");
    }
}

// float 版本
template<>
void rope_impl<float>(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, 
                      size_t seqlen, size_t nhead, size_t d, size_t half_d) {
    // 获取数据指针
    std::byte* out_data_byte = out->data();
    std::byte* in_data_byte = in->data();
    std::byte* pos_data_byte = pos_ids->data();
    
    float* out_data = reinterpret_cast<float*>(out_data_byte);
    float* in_data = reinterpret_cast<float*>(in_data_byte);
    int64_t* pos_data = reinterpret_cast<int64_t*>(pos_data_byte);
    
    // 预计算角度因子
    std::vector<double> freqs(half_d);
    double base = static_cast<double>(theta);
    double dim_double = static_cast<double>(d);
    
    for (size_t j = 0; j < half_d; ++j) {
        double exponent = -static_cast<double>(2 * j) / dim_double;
        freqs[j] = std::pow(base, exponent);
    }
    
    // 对序列中的每个token
    for (size_t i = 0; i < seqlen; ++i) {
        double pos = static_cast<double>(pos_data[i]);
        
        // 对每个注意力头
        for (size_t h = 0; h < nhead; ++h) {
            // 计算当前token和头的起始索引
            size_t base_idx = i * nhead * d + h * d;
            
            // 对d/2对(a_j, b_j)
            for (size_t j = 0; j < half_d; ++j) {
                // 计算索引
                size_t a_idx = base_idx + j;
                size_t b_idx = base_idx + j + half_d;
                
                // 获取输入值
                float a_j = in_data[a_idx];
                float b_j = in_data[b_idx];
                
                // 计算旋转角度
                double angle = pos * freqs[j];
                
                // 计算cos和sin
                double cos_angle = std::cos(angle);
                double sin_angle = std::sin(angle);
                
                // 应用旋转公式
                out_data[a_idx] = static_cast<float>(a_j * cos_angle - b_j * sin_angle);
                out_data[b_idx] = static_cast<float>(b_j * cos_angle + a_j * sin_angle);
            }
        }
    }
}

// fp16_t 版本
template<>
void rope_impl<fp16_t>(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, 
                       size_t seqlen, size_t nhead, size_t d, size_t half_d) {
    // 获取数据指针
    std::byte* out_data_byte = out->data();
    std::byte* in_data_byte = in->data();
    std::byte* pos_data_byte = pos_ids->data();
    
    fp16_t* out_data = reinterpret_cast<fp16_t*>(out_data_byte);
    fp16_t* in_data = reinterpret_cast<fp16_t*>(in_data_byte);
    int64_t* pos_data = reinterpret_cast<int64_t*>(pos_data_byte);
    
    // 预计算角度因子
    std::vector<double> freqs(half_d);
    double base = static_cast<double>(theta);
    double dim_double = static_cast<double>(d);
    
    for (size_t j = 0; j < half_d; ++j) {
        double exponent = -static_cast<double>(2 * j) / dim_double;
        freqs[j] = std::pow(base, exponent);
    }
    
    // 对序列中的每个token
    for (size_t i = 0; i < seqlen; ++i) {
        double pos = static_cast<double>(pos_data[i]);
        
        // 对每个注意力头
        for (size_t h = 0; h < nhead; ++h) {
            // 计算当前token和头的起始索引
            size_t base_idx = i * nhead * d + h * d;
            
            // 对d/2对(a_j, b_j)
            for (size_t j = 0; j < half_d; ++j) {
                // 计算索引
                size_t a_idx = base_idx + j;
                size_t b_idx = base_idx + j + half_d;
                
                // 获取输入值并转换为float
                float a_j = utils::_f16_to_f32(in_data[a_idx]);
                float b_j = utils::_f16_to_f32(in_data[b_idx]);
                
                // 计算旋转角度
                double angle = pos * freqs[j];
                
                // 计算cos和sin
                double cos_angle = std::cos(angle);
                double sin_angle = std::sin(angle);
                
                // 应用旋转公式并转换回fp16_t
                float a_prime = static_cast<float>(a_j * cos_angle - b_j * sin_angle);
                float b_prime = static_cast<float>(b_j * cos_angle + a_j * sin_angle);
                
                out_data[a_idx] = utils::_f32_to_f16(a_prime);
                out_data[b_idx] = utils::_f32_to_f16(b_prime);
            }
        }
    }
}

// bf16_t 版本
template<>
void rope_impl<bf16_t>(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, 
                       size_t seqlen, size_t nhead, size_t d, size_t half_d) {
    // 获取数据指针
    std::byte* out_data_byte = out->data();
    std::byte* in_data_byte = in->data();
    std::byte* pos_data_byte = pos_ids->data();
    
    bf16_t* out_data = reinterpret_cast<bf16_t*>(out_data_byte);
    bf16_t* in_data = reinterpret_cast<bf16_t*>(in_data_byte);
    int64_t* pos_data = reinterpret_cast<int64_t*>(pos_data_byte);
    
    // 预计算角度因子
    std::vector<double> freqs(half_d);
    double base = static_cast<double>(theta);
    double dim_double = static_cast<double>(d);
    
    for (size_t j = 0; j < half_d; ++j) {
        double exponent = -static_cast<double>(2 * j) / dim_double;
        freqs[j] = std::pow(base, exponent);
    }
    
    // 对序列中的每个token
    for (size_t i = 0; i < seqlen; ++i) {
        double pos = static_cast<double>(pos_data[i]);
        
        // 对每个注意力头
        for (size_t h = 0; h < nhead; ++h) {
            // 计算当前token和头的起始索引
            size_t base_idx = i * nhead * d + h * d;
            
            // 对d/2对(a_j, b_j)
            for (size_t j = 0; j < half_d; ++j) {
                // 计算索引
                size_t a_idx = base_idx + j;
                size_t b_idx = base_idx + j + half_d;
                
                // 获取输入值并转换为float
                float a_j = utils::_bf16_to_f32(in_data[a_idx]);
                float b_j = utils::_bf16_to_f32(in_data[b_idx]);
                
                // 计算旋转角度
                double angle = pos * freqs[j];
                
                // 计算cos和sin
                double cos_angle = std::cos(angle);
                double sin_angle = std::sin(angle);
                
                // 应用旋转公式并转换回bf16_t
                float a_prime = static_cast<float>(a_j * cos_angle - b_j * sin_angle);
                float b_prime = static_cast<float>(b_j * cos_angle + a_j * sin_angle);
                
                out_data[a_idx] = utils::_f32_to_bf16(a_prime);
                out_data[b_idx] = utils::_f32_to_bf16(b_prime);
            }
        }
    }
}

} // namespace llaisys::ops