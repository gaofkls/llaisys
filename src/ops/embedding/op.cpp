#include "op.hpp"
#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 检查维度
    if (index->ndim() != 1) {
        throw std::runtime_error("Index tensor must be 1-D");
    }
    if (weight->ndim() != 2) {
        throw std::runtime_error("Weight tensor must be 2-D");
    }
    if (out->ndim() != 2) {
        throw std::runtime_error("Output tensor must be 2-D");
    }
    
    // 检查维度匹配
    size_t batch_size = index->shape()[0];
    size_t embedding_dim = weight->shape()[1];
    
    if (out->shape()[0] != batch_size || out->shape()[1] != embedding_dim) {
        throw std::runtime_error("Output tensor dimensions do not match");
    }
    
    // 检查数据类型 - index 必须是 Int64
    if (index->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("Index tensor must be Int64 type");
    }
    
    // 检查连续性（为了性能）
    if (!out->isContiguous() || !weight->isContiguous()) {
        throw std::runtime_error("Output and weight tensors must be contiguous");
    }
    
    // 获取数据指针
    int64_t* index_data = reinterpret_cast<int64_t*>(index->data());
    std::byte* weight_data = weight->data();
    std::byte* out_data = out->data();
    
    // 获取元素大小（对于 float 类型通常是 4 字节）
    size_t elem_size = out->elementSize();
    
    // 遍历每个索引
    for (size_t i = 0; i < batch_size; ++i) {
        int64_t idx = index_data[i];
        
        // 检查索引范围
        if (idx < 0 || static_cast<size_t>(idx) >= weight->shape()[0]) {
            throw std::runtime_error("Index out of bounds");
        }
        
        // 计算源和目标的内存位置
        const std::byte* src_row = weight_data + idx * embedding_dim * elem_size;
        std::byte* dst_row = out_data + i * embedding_dim * elem_size;
        
        // 复制数据
        std::memcpy(dst_row, src_row, embedding_dim * elem_size);
    }
}
} // namespace llaisys::ops