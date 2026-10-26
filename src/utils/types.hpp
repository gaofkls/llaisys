#include "llaisys.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace llaisys {
struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t;

namespace utils {
inline size_t dsize(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return sizeof(char);
    case LLAISYS_DTYPE_BOOL:
        return sizeof(char);
    case LLAISYS_DTYPE_I8:
        return sizeof(int8_t);
    case LLAISYS_DTYPE_I16:
        return sizeof(int16_t);
    case LLAISYS_DTYPE_I32:
        return sizeof(int32_t);
    case LLAISYS_DTYPE_I64:
        return sizeof(int64_t);
    case LLAISYS_DTYPE_U8:
        return sizeof(uint8_t);
    case LLAISYS_DTYPE_U16:
        return sizeof(uint16_t);
    case LLAISYS_DTYPE_U32:
        return sizeof(uint32_t);
    case LLAISYS_DTYPE_U64:
        return sizeof(uint64_t);
    case LLAISYS_DTYPE_F8:
        return 1; // usually 8-bit float (custom)
    case LLAISYS_DTYPE_F16:
        return 2; // 16-bit float
    case LLAISYS_DTYPE_BF16:
        return 2; // bfloat16
    case LLAISYS_DTYPE_F32:
        return sizeof(float);
    case LLAISYS_DTYPE_F64:
        return sizeof(double);
    case LLAISYS_DTYPE_C16:
        return 2; // 2 bytes complex (not standard)
    case LLAISYS_DTYPE_C32:
        return 4; // 4 bytes complex
    case LLAISYS_DTYPE_C64:
        return 8; // 8 bytes complex
    case LLAISYS_DTYPE_C128:
        return 16; // 16 bytes complex
    case LLAISYS_DTYPE_INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

inline const char *dtype_to_str(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return "byte";
    case LLAISYS_DTYPE_BOOL:
        return "bool";
    case LLAISYS_DTYPE_I8:
        return "int8";
    case LLAISYS_DTYPE_I16:
        return "int16";
    case LLAISYS_DTYPE_I32:
        return "int32";
    case LLAISYS_DTYPE_I64:
        return "int64";
    case LLAISYS_DTYPE_U8:
        return "uint8";
    case LLAISYS_DTYPE_U16:
        return "uint16";
    case LLAISYS_DTYPE_U32:
        return "uint32";
    case LLAISYS_DTYPE_U64:
        return "uint64";
    case LLAISYS_DTYPE_F8:
        return "float8";
    case LLAISYS_DTYPE_F16:
        return "float16";
    case LLAISYS_DTYPE_BF16:
        return "bfloat16";
    case LLAISYS_DTYPE_F32:
        return "float32";
    case LLAISYS_DTYPE_F64:
        return "float64";
    case LLAISYS_DTYPE_C16:
        return "complex16";
    case LLAISYS_DTYPE_C32:
        return "complex32";
    case LLAISYS_DTYPE_C64:
        return "complex64";
    case LLAISYS_DTYPE_C128:
        return "complex128";
    case LLAISYS_DTYPE_INVALID:
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

// Half precision conversions live in the header on purpose: the ops kernels call
// them once per element, and only an inline (loop free) definition lets the
// compiler vectorise the surrounding loop.
inline float _f16_to_f32(fp16_t val) {
    const uint32_t h = val._v;
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1Fu;
    const uint32_t mantissa = h & 0x3FFu;

    // Normal numbers: rebias the exponent (127 - 15 == 112).
    const uint32_t normal = sign | ((exponent + 112u) << 23) | (mantissa << 13);

    // Subnormals: value == mantissa * 2^-24, no loop needed.
    const float sub = static_cast<float>(mantissa) * 5.9604644775390625e-8f;
    uint32_t sub_bits;
    std::memcpy(&sub_bits, &sub, sizeof(sub_bits));
    sub_bits |= sign;

    uint32_t bits;
    if (exponent == 0) {
        bits = sub_bits;
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = normal;
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline fp16_t _f32_to_f16(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(f32));

    const uint16_t sign = static_cast<uint16_t>((f32 >> 16) & 0x8000u);
    const uint32_t magnitude = f32 & 0x7FFFFFFFu;
    const int32_t exponent = static_cast<int32_t>(magnitude >> 23) - 127; // unbiased
    const uint32_t mantissa = magnitude & 0x7FFFFFu;

    if (exponent > 15) { // does not fit in fp16: overflow to infinity, keep NaN
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00u)};
        }
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00u)};
    }

    // Round to nearest, ties to even. Truncating instead (what this used to do)
    // biases every result towards zero by up to one ulp, i.e. by 2.4e-4 relative
    // for a value around one -- enough to fail a 1e-3 comparison against a
    // reference that rounds.
    if (exponent >= -14) { // normal fp16 result: 13 mantissa bits are dropped
        const uint32_t kept = mantissa >> 13;
        const uint32_t dropped = mantissa & 0x1FFFu;
        const uint32_t round_up =
            (dropped > 0x1000u) || (dropped == 0x1000u && (kept & 1u) != 0);
        // The carry out of the mantissa propagates into the exponent field, which
        // is exactly the IEEE result (and yields infinity past the largest value).
        const uint32_t bits = ((static_cast<uint32_t>(exponent + 15) << 10) | kept) + round_up;
        return fp16_t{static_cast<uint16_t>(sign | bits)};
    }

    if (exponent >= -25) { // subnormal fp16 result
        const uint32_t m = mantissa | 0x800000u; // implicit leading one
        const uint32_t shift = 13u + static_cast<uint32_t>(-14 - exponent);
        const uint32_t kept = m >> shift;
        const uint32_t dropped = m & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        const uint32_t round_up =
            (dropped > halfway) || (dropped == halfway && (kept & 1u) != 0);
        return fp16_t{static_cast<uint16_t>(sign | (kept + round_up))};
    }

    return fp16_t{static_cast<uint16_t>(sign)}; // underflows to signed zero
}

inline float _bf16_to_f32(bf16_t val) {
    const uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

inline bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFFu + ((bits32 >> 16) & 1u);
    const uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}

template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(val);
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_f16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, float>::value) {
        return _f16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_f16_to_f32(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(val);
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return _bf16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_bf16_to_f32(val));
    } else {
        return static_cast<TypeTo>(val);
    }
}

} // namespace utils
} // namespace llaisys
