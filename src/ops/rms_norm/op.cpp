#include "op.hpp"
#include <cmath>
#include <cstring>
#include <vector>

namespace llaisys::ops {

// Helper functions for F16 conversion - make them static to avoid duplicate definition
static float f16_to_f32(uint16_t f16) {
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

static uint16_t f32_to_f16(float f32) {
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

// Helper functions for BF16 conversion - make them static to avoid duplicate definition
static float bf16_to_f32(uint16_t bf16) {
    union {
        uint32_t u32;
        float f32;
    } u;
    u.u32 = static_cast<uint32_t>(bf16) << 16;  // BF16 occupies high 16 bits
    return u.f32;
}

static uint16_t f32_to_bf16(float f32) {
    union {
        float f32;
        uint32_t u32;
    } u;
    u.f32 = f32;
    return static_cast<uint16_t>(u.u32 >> 16);  // Take high 16 bits for BF16
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // Check dimensions
    if (in->ndim() != 2) {
        throw std::runtime_error("Input tensor must be 2-D");
    }
    if (out->ndim() != 2) {
        throw std::runtime_error("Output tensor must be 2-D");
    }
    if (weight->ndim() != 1) {
        throw std::runtime_error("Weight tensor must be 1-D");
    }
    
    // Get dimensions
    size_t batch_size = in->shape()[0];
    size_t feature_dim = in->shape()[1];
    
    // Check shape compatibility
    if (out->shape()[0] != batch_size || out->shape()[1] != feature_dim) {
        throw std::runtime_error("Output tensor dimensions do not match input");
    }
    if (weight->shape()[0] != feature_dim) {
        throw std::runtime_error("Weight tensor dimension does not match input feature dimension");
    }
    
    // Check data types
    if (in->dtype() != out->dtype()) {
        throw std::runtime_error("Input and output tensors must have the same data type");
    }
    if (in->dtype() != weight->dtype()) {
        throw std::runtime_error("Input and weight tensors must have the same data type");
    }
    
    // Check if all tensors are contiguous
    if (!in->isContiguous() || !out->isContiguous() || !weight->isContiguous()) {
        throw std::runtime_error("All tensors must be contiguous");
    }
    
    // Support floating point types including F16 and BF16
    if (in->dtype() != LLAISYS_DTYPE_F32 && in->dtype() != LLAISYS_DTYPE_F64 && 
        in->dtype() != LLAISYS_DTYPE_F16 && in->dtype() != LLAISYS_DTYPE_BF16) {
        throw std::runtime_error("RMS norm only supports floating point types");
    }
    
    // Process based on data type
    if (in->dtype() == LLAISYS_DTYPE_F32) {
        float* in_data = reinterpret_cast<float*>(in->data());
        float* out_data = reinterpret_cast<float*>(out->data());
        float* weight_data = reinterpret_cast<float*>(weight->data());
        
        for (size_t b = 0; b < batch_size; ++b) {
            // Calculate RMS normalization for each row
            float sum_sq = 0.0f;
            const float* in_row = in_data + b * feature_dim;
            float* out_row = out_data + b * feature_dim;
            
            // Compute sum of squares for the current row
            for (size_t j = 0; j < feature_dim; ++j) {
                float val = in_row[j];
                sum_sq += val * val;
            }
            
            // Compute normalization factor
            float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(feature_dim) + eps);
            
            // Apply normalization and weight
            for (size_t j = 0; j < feature_dim; ++j) {
                out_row[j] = weight_data[j] * in_row[j] * inv_rms;
            }
        }
    } else if (in->dtype() == LLAISYS_DTYPE_F64) {
        double* in_data = reinterpret_cast<double*>(in->data());
        double* out_data = reinterpret_cast<double*>(out->data());
        double* weight_data = reinterpret_cast<double*>(weight->data());
        
        for (size_t b = 0; b < batch_size; ++b) {
            // Calculate RMS normalization for each row
            double sum_sq = 0.0;
            const double* in_row = in_data + b * feature_dim;
            double* out_row = out_data + b * feature_dim;
            
            // Compute sum of squares for the current row
            for (size_t j = 0; j < feature_dim; ++j) {
                double val = in_row[j];
                sum_sq += val * val;
            }
            
            // Compute normalization factor
            double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(feature_dim) + eps);
            
            // Apply normalization and weight
            for (size_t j = 0; j < feature_dim; ++j) {
                out_row[j] = weight_data[j] * in_row[j] * inv_rms;
            }
        }
    } else if (in->dtype() == LLAISYS_DTYPE_F16) {
        // For F16, convert to F32 for computation and back to F16 for output
        uint16_t* in_data = reinterpret_cast<uint16_t*>(in->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        uint16_t* weight_data = reinterpret_cast<uint16_t*>(weight->data());
        
        // Allocate temporary F32 arrays for computation
        std::vector<float> temp_in(feature_dim);
        std::vector<float> temp_weight(feature_dim);
        std::vector<float> temp_out(feature_dim);
        
        for (size_t b = 0; b < batch_size; ++b) {
            // Convert F16 input row to F32
            const uint16_t* in_row = in_data + b * feature_dim;
            for (size_t j = 0; j < feature_dim; ++j) {
                temp_in[j] = f16_to_f32(in_row[j]);
            }
            
            // Convert F16 weight to F32
            for (size_t j = 0; j < feature_dim; ++j) {
                temp_weight[j] = f16_to_f32(weight_data[j]);
            }
            
            // Calculate RMS normalization for the current row
            float sum_sq = 0.0f;
            for (size_t j = 0; j < feature_dim; ++j) {
                float val = temp_in[j];
                sum_sq += val * val;
            }
            
            // Compute normalization factor
            float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(feature_dim) + eps);
            
            // Apply normalization and weight
            uint16_t* out_row = out_data + b * feature_dim;
            for (size_t j = 0; j < feature_dim; ++j) {
                temp_out[j] = temp_weight[j] * temp_in[j] * inv_rms;
                out_row[j] = f32_to_f16(temp_out[j]);
            }
        }
    } else if (in->dtype() == LLAISYS_DTYPE_BF16) {
        // For BF16, convert to F32 for computation and back to BF16 for output
        uint16_t* in_data = reinterpret_cast<uint16_t*>(in->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        uint16_t* weight_data = reinterpret_cast<uint16_t*>(weight->data());
        
        // Allocate temporary F32 arrays for computation
        std::vector<float> temp_in(feature_dim);
        std::vector<float> temp_weight(feature_dim);
        std::vector<float> temp_out(feature_dim);
        
        for (size_t b = 0; b < batch_size; ++b) {
            // Convert BF16 input row to F32
            const uint16_t* in_row = in_data + b * feature_dim;
            for (size_t j = 0; j < feature_dim; ++j) {
                temp_in[j] = bf16_to_f32(in_row[j]);
            }
            
            // Convert BF16 weight to F32
            for (size_t j = 0; j < feature_dim; ++j) {
                temp_weight[j] = bf16_to_f32(weight_data[j]);
            }
            
            // Calculate RMS normalization for the current row
            float sum_sq = 0.0f;
            for (size_t j = 0; j < feature_dim; ++j) {
                float val = temp_in[j];
                sum_sq += val * val;
            }
            
            // Compute normalization factor
            float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(feature_dim) + eps);
            
            // Apply normalization and weight
            uint16_t* out_row = out_data + b * feature_dim;
            for (size_t j = 0; j < feature_dim; ++j) {
                temp_out[j] = temp_weight[j] * temp_in[j] * inv_rms;
                out_row[j] = f32_to_bf16(temp_out[j]);
            }
        }
    }
}

} // namespace llaisys::ops