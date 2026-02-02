#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <type_traits>

namespace llaisys::ops {

/**
 * @brief 模板函数，逐元素计算 SwiGLU 激活函数
 * 
 * SwiGLU 函数定义为: output = up * (gate * sigmoid(gate))
 * 其中 sigmoid(x) = 1 / (1 + exp(-x))
 * 
 * @tparam T 输入和输出张量的数据类型
 * @param[out] out 输出张量，包含 SwiGLU 运算的结果
 * @param[in] gate 输入张量，表示门控值
 * @param[in] up 输入张量，被门控调制的张量
 * @param[in] numel 张量中的元素数量
 */
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float g;  // 门控值转换为 float 进行计算
        float u;  // 上路值转换为 float 进行计算
        
        // 将半精度类型 (BF16/FP16) 转换为 float 进行计算
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            g = llaisys::utils::cast<float>(gate[i]);
            u = llaisys::utils::cast<float>(up[i]);
        } else {
            // 对于其他类型 (如 FP32)，直接转换为 float
            g = static_cast<float>(gate[i]);
            u = static_cast<float>(up[i]);
        }

        // 计算门控的 sigmoid: sigmoid(g) = 1 / (1 + e^(-g))
        float sig = 1.0f / (1.0f + std::exp(-g));
        
        // 计算 SwiGLU: up * (gate * sigmoid(gate))
        float val = u * (g * sig);

        // 如需要，将结果转回原始数据类型 (对于半精度)
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(val);
        } else {
            out[i] = static_cast<T>(val);
        }
    }
}

/**
 * @brief 执行 SwiGLU 运算的公共 API 函数
 * 
 * 此函数执行验证检查，并根据张量的数据类型调度到相应的模板特化版本
 * 
 * @param[out] out 输出张量 (将包含 SwiGLU 结果)
 * @param[in] gate 用于 SwiGLU 的门控组件的输入张量
 * @param[in] up 用于 SwiGLU 的上路组件的输入张量
 */
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 验证所有张量都在同一设备上
    CHECK_SAME_DEVICE(out, gate, up);
    
    // 验证所有张量都有相同的数据类型
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    
    // 验证所有张量都有相同的形状
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    // 确保所有张量在内存中都是连续的，以便高效处理
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: 所有张量都必须是连续的");

    // 获取张量中的总元素数
    size_t numel = out->numel();

    // 根据数据类型调度到适当的模板特化
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out->data()),
                       reinterpret_cast<const float *>(gate->data()),
                       reinterpret_cast<const float *>(up->data()),
                       numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                       reinterpret_cast<const llaisys::fp16_t *>(gate->data()),
                       reinterpret_cast<const llaisys::fp16_t *>(up->data()),
                       numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                       reinterpret_cast<const llaisys::bf16_t *>(gate->data()),
                       reinterpret_cast<const llaisys::bf16_t *>(up->data()),
                       numel);
    default:
        // 对不支持的数据类型抛出异常
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops