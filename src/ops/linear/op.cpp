#include "op.hpp"
#include <cstring>
#include <vector>
#include <cstdint>

namespace llaisys::ops {

// Forward declarations
template<typename T>
void linear_impl(T* out, const T* in, const T* weight, const T* bias,
                 size_t batch_size, size_t in_features, size_t out_features);

void linear_impl_f16(std::byte* out, const std::byte* in, const std::byte* weight, 
                     const std::byte* bias,
                     size_t batch_size, size_t in_features, size_t out_features);

void linear_impl_bf16(std::byte* out, const std::byte* in, const std::byte* weight, 
                      const std::byte* bias,
                      size_t batch_size, size_t in_features, size_t out_features);

// Proper F16 conversion helper functions
float f16_to_f32(uint16_t f16) {
    // Extract components from half-precision format (IEEE 754)
    uint32_t sign = ((f16 >> 15) & 0x1);
    uint32_t exp = ((f16 >> 10) & 0x1F);
    uint32_t mantissa = (f16 & 0x3FF);
    
    uint32_t result;
    
    if (exp == 0) {  // Zero or subnormal
        if (mantissa == 0) {
            result = sign << 31;  // +0 or -0
        } else {
            // Subnormal number - normalize it
            int s_exp = -14;
            while ((mantissa & 0x400) == 0) {  // While leading bit is not set
                mantissa <<= 1;
                s_exp--;
            }
            mantissa &= 0x3FF;  // Remove the implicit leading bit
            result = (sign << 31) | ((s_exp + 127) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {  // Infinity or NaN
        result = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {  // Normalized number
        result = (sign << 31) | ((exp - 15 + 127) << 23) | (mantissa << 13);
    }
    
    union {
        uint32_t u32;
        float f32;
    } u;
    u.u32 = result;
    return u.f32;
}

uint16_t f32_to_f16(float f32) {
    union {
        float f32;
        uint32_t u32;
    } u;
    u.f32 = f32;
    
    uint32_t bits = u.u32;
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;
    
    uint16_t result;
    
    if (exp == 0) {  // Zero or subnormal
        result = (uint16_t)(sign << 15);
    } else if (exp == 0xFF) {  // Infinity or NaN
        result = (uint16_t)((sign << 15) | (0x1F << 10) | (mantissa ? 0x200 : 0));  // If NaN, keep it as NaN
    } else {
        int new_exp = (int)exp - 127 + 15;  // Adjust bias from float32 to float16
        
        if (new_exp >= 31) {  // Overflow to infinity
            result = (uint16_t)((sign << 15) | (0x1F << 10));
        } else if (new_exp <= 0) {  // Underflow to subnormal or zero
            if (new_exp < -10) {  // Would definitely underflow
                result = (uint16_t)(sign << 15);
            } else {
                // Create subnormal number
                mantissa |= 0x800000;  // Add implicit leading bit
                mantissa >>= (-new_exp + 1);  // Shift right to create subnormal
                result = (uint16_t)((sign << 15) | (mantissa >> 13));
                
                // Round to nearest even
                if ((mantissa >> 12) & 1) {  // Check if we should round up
                    if ((mantissa & 0xFFF) > 0x800 || ((mantissa & 0xFFF) == 0x800 && (result & 1))) {
                        result++;
                    }
                }
            }
        } else {  // Normal number
            result = (uint16_t)((sign << 15) | (new_exp << 10) | (mantissa >> 13));
            
            // Round to nearest even
            if ((mantissa >> 12) & 1) {  // Check if we should round up
                if ((mantissa & 0xFFF) > 0x800 || ((mantissa & 0xFFF) == 0x800 && (result & 1))) {
                    if (result == (uint16_t)((sign << 15) | (0x1F << 10) - 1)) {  // Would overflow to infinity
                        result = (uint16_t)((sign << 15) | (0x1F << 10));  // Infinity
                    } else {
                        result++;
                    }
                }
            }
        }
    }
    
    return result;
}

// BF16 conversion helper functions
float bf16_to_f32(uint16_t bf16) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.u32 = static_cast<uint32_t>(bf16) << 16;  // BF16 occupies high 16 bits
    return u.f32;
}

uint16_t f32_to_bf16(float f32) {
    union {
        float f32;
        uint32_t u32;
    } u;
    u.f32 = f32;
    return static_cast<uint16_t>(u.u32 >> 16);  // Take high 16 bits for BF16
}

// Main function
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Check dimensions
    if (in->ndim() != 2) {
        throw std::runtime_error("Input tensor must be 2-D");
    }
    if (weight->ndim() != 2) {
        throw std::runtime_error("Weight tensor must be 2-D");
    }
    if (out->ndim() != 2) {
        throw std::runtime_error("Output tensor must be 2-D");
    }
    
    // Get dimensions
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];  // W shape is [out_features, in_features]
    
    // Check dimension matching
    if (weight->shape()[1] != in_features) {
        throw std::runtime_error("Weight feature dimension does not match input");
    }
    if (out->shape()[0] != batch_size || out->shape()[1] != out_features) {
        throw std::runtime_error("Output tensor dimensions do not match");
    }
    
    // Check data types (including BF16)
    if (in->dtype() != LLAISYS_DTYPE_F32 && in->dtype() != LLAISYS_DTYPE_F64 &&
        in->dtype() != LLAISYS_DTYPE_F16 && in->dtype() != LLAISYS_DTYPE_BF16) {
        throw std::runtime_error("Input tensor must be floating point type");
    }
    if (weight->dtype() != in->dtype()) {
        throw std::runtime_error("Weight tensor must have same dtype as input");
    }
    if (out->dtype() != in->dtype()) {
        throw std::runtime_error("Output tensor must have same dtype as input");
    }
    
    // Check continuity
    if (!out->isContiguous() || !in->isContiguous() || !weight->isContiguous()) {
        throw std::runtime_error("All tensors must be contiguous");
    }
    
    // Check bias (if provided)
    bool has_bias = (bias != nullptr);
    if (has_bias) {
        if (bias->ndim() != 1) {
            throw std::runtime_error("Bias tensor must be 1-D");
        }
        if (bias->shape()[0] != out_features) {
            throw std::runtime_error("Bias dimension does not match output features");
        }
        if (bias->dtype() != in->dtype()) {
            throw std::runtime_error("Bias tensor must have same dtype as input");
        }
        if (!bias->isContiguous()) {
            throw std::runtime_error("Bias tensor must be contiguous");
        }
    }
    
    // Get data pointers
    std::byte* in_data = in->data();
    std::byte* weight_data = weight->data();
    std::byte* out_data = out->data();
    std::byte* bias_data = has_bias ? bias->data() : nullptr;
    
    // Select implementation based on data type
    switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            linear_impl<float>(reinterpret_cast<float*>(out_data),
                               reinterpret_cast<const float*>(in_data),
                               reinterpret_cast<const float*>(weight_data),
                               has_bias ? reinterpret_cast<const float*>(bias_data) : nullptr,
                               batch_size, in_features, out_features);
            break;
        case LLAISYS_DTYPE_F64:
            linear_impl<double>(reinterpret_cast<double*>(out_data),
                                reinterpret_cast<const double*>(in_data),
                                reinterpret_cast<const double*>(weight_data),
                                has_bias ? reinterpret_cast<const double*>(bias_data) : nullptr,
                                batch_size, in_features, out_features);
            break;
        case LLAISYS_DTYPE_F16:
            // For half precision floats, special processing is needed
            linear_impl_f16(out_data, in_data, weight_data, bias_data,
                           batch_size, in_features, out_features);
            break;
        case LLAISYS_DTYPE_BF16:
            // For brain float 16, special processing is needed
            linear_impl_bf16(out_data, in_data, weight_data, bias_data,
                            batch_size, in_features, out_features);
            break;
        default:
            throw std::runtime_error("Unsupported data type for linear operation");
    }
}

// Template implementation, supporting float and double
template<typename T>
void linear_impl(T* out, const T* in, const T* weight, const T* bias,
                 size_t batch_size, size_t in_features, size_t out_features) {
    // Initialize output to zero
    std::memset(out, 0, batch_size * out_features * sizeof(T));
    
    // Perform matrix multiplication: out = in * weight^T
    // Note: weight shape is [out_features, in_features], need to transpose
    for (size_t b = 0; b < batch_size; ++b) {
        const T* in_row = in + b * in_features;  // in[b, :]
        
        for (size_t i = 0; i < out_features; ++i) {
            T sum = T(0);
            const T* weight_row = weight + i * in_features;  // weight[i, :]
            
            // Calculate dot product
            for (size_t j = 0; j < in_features; ++j) {
                sum += in_row[j] * weight_row[j];
            }
            
            out[b * out_features + i] = sum;
        }
    }
    
    // Add bias
    if (bias) {
        for (size_t b = 0; b < batch_size; ++b) {
            T* out_row = out + b * out_features;
            for (size_t i = 0; i < out_features; ++i) {
                out_row[i] += bias[i];
            }
        }
    }
}

// Half precision float implementation (improved version)
void linear_impl_f16(std::byte* out, const std::byte* in, const std::byte* weight, 
                     const std::byte* bias,
                     size_t batch_size, size_t in_features, size_t out_features) {
    // Since half precision float calculations are complex, convert to float for calculation
    
    // Temporarily allocate float buffers
    std::vector<float> in_float(batch_size * in_features);
    std::vector<float> weight_float(out_features * in_features);
    std::vector<float> bias_float;
    std::vector<float> out_float(batch_size * out_features);
    
    bool has_bias = (bias != nullptr);
    if (has_bias) {
        bias_float.resize(out_features);
    }
    
    // Convert F16 to F32
    const uint16_t* in_f16 = reinterpret_cast<const uint16_t*>(in);
    const uint16_t* weight_f16 = reinterpret_cast<const uint16_t*>(weight);
    const uint16_t* bias_f16 = has_bias ? reinterpret_cast<const uint16_t*>(bias) : nullptr;
    
    // Convert input
    for (size_t i = 0; i < batch_size * in_features; ++i) {
        in_float[i] = f16_to_f32(in_f16[i]);
    }
    
    // Convert weights
    for (size_t i = 0; i < out_features * in_features; ++i) {
        weight_float[i] = f16_to_f32(weight_f16[i]);
    }
    
    // Convert bias
    if (has_bias) {
        for (size_t i = 0; i < out_features; ++i) {
            bias_float[i] = f16_to_f32(bias_f16[i]);
        }
    }
    
    // Call float version implementation
    linear_impl<float>(out_float.data(),
                       in_float.data(),
                       weight_float.data(),
                       has_bias ? bias_float.data() : nullptr,
                       batch_size, in_features, out_features);
    
    // Convert F32 back to F16
    uint16_t* out_f16 = reinterpret_cast<uint16_t*>(out);
    for (size_t i = 0; i < batch_size * out_features; ++i) {
        out_f16[i] = f32_to_f16(out_float[i]);
    }
}

// Brain float 16 implementation
void linear_impl_bf16(std::byte* out, const std::byte* in, const std::byte* weight, 
                      const std::byte* bias,
                      size_t batch_size, size_t in_features, size_t out_features) {
    // Since brain float 16 calculations are complex, convert to float for calculation
    
    // Temporarily allocate float buffers
    std::vector<float> in_float(batch_size * in_features);
    std::vector<float> weight_float(out_features * in_features);
    std::vector<float> bias_float;
    std::vector<float> out_float(batch_size * out_features);
    
    bool has_bias = (bias != nullptr);
    if (has_bias) {
        bias_float.resize(out_features);
    }
    
    // Convert BF16 to F32
    const uint16_t* in_bf16 = reinterpret_cast<const uint16_t*>(in);
    const uint16_t* weight_bf16 = reinterpret_cast<const uint16_t*>(weight);
    const uint16_t* bias_bf16 = has_bias ? reinterpret_cast<const uint16_t*>(bias) : nullptr;
    
    // Convert input
    for (size_t i = 0; i < batch_size * in_features; ++i) {
        in_float[i] = bf16_to_f32(in_bf16[i]);
    }
    
    // Convert weights
    for (size_t i = 0; i < out_features * in_features; ++i) {
        weight_float[i] = bf16_to_f32(weight_bf16[i]);
    }
    
    // Convert bias
    if (has_bias) {
        for (size_t i = 0; i < out_features; ++i) {
            bias_float[i] = bf16_to_f32(bias_bf16[i]);
        }
    }
    
    // Call float version implementation
    linear_impl<float>(out_float.data(),
                       in_float.data(),
                       weight_float.data(),
                       has_bias ? bias_float.data() : nullptr,
                       batch_size, in_features, out_features);
    
    // Convert F32 back to BF16
    uint16_t* out_bf16 = reinterpret_cast<uint16_t*>(out);
    for (size_t i = 0; i < batch_size * out_features; ++i) {
        out_bf16[i] = f32_to_bf16(out_float[i]);
    }
}

} // namespace llaisys::ops