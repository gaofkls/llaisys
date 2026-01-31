#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // 获取维度数
    size_t ndim = this->ndim();
    
    // 如果是0维张量（标量），则认为是连续的
    if (ndim == 0) {
        return true;
    }
    
    // 从最后一个维度开始向前检查步长
    ptrdiff_t expected_stride = 1;
    
    for (size_t i = ndim; i > 0; --i) {
        size_t dim_idx = i - 1;
        
        // 检查当前维度的步长是否符合预期
        if (this->strides()[dim_idx] != static_cast<ptrdiff_t>(expected_stride)) {
            return false;
        }
        // 更新期望的步长
        expected_stride *= this->shape()[dim_idx];
    }
    
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 检查order向量的长度是否与当前张量的维度数相等
    if (order.size() != this->ndim()) {
        throw std::runtime_error("Permutation order size must match tensor dimensions");
    }
    
    // 检查order向量是否包含有效的维度索引（无重复且范围正确）
    std::vector<bool> used(this->ndim(), false);
    for (size_t idx : order) {
        if (idx >= this->ndim()) {
            throw std::runtime_error("Invalid dimension index in permutation order");
        }
        if (used[idx]) {
            throw std::runtime_error("Duplicate dimension index in permutation order");
        }
        used[idx] = true;
    }
    
    // 构建新的形状和步长
    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());
    
    for (size_t i = 0; i < order.size(); ++i) {
        size_t orig_dim = order[i];
        new_shape[i] = this->shape()[orig_dim];
        new_strides[i] = this->strides()[orig_dim];
    }
    
    // 创建新的TensorMeta
    TensorMeta new_meta{this->dtype(), new_shape, new_strides};
    
    // 返回一个新的张量，具有重新排列的维度，但共享相同的存储
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 计算原张量的元素总数
    size_t original_numel = this->numel();
    
    // 计算新形状的元素总数
    size_t new_numel = 1;
    for (size_t dim : shape) {
        new_numel *= dim;
    }
    
    // 检查新形状的元素总数是否与原张量匹配
    if (original_numel != new_numel) {
        throw std::runtime_error("New shape is incompatible with original tensor size");
    }
    
    // 检查张量是否连续，只有连续张量才能安全地改变视图
    if (!this->isContiguous()) {
        throw std::runtime_error("Cannot view non-contiguous tensor, call contiguous() first");
    }
    
    // 计算新形状对应的步长（C连续顺序）
    size_t ndim = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim);
    
    size_t stride = 1;
    for (size_t i = 1; i <= ndim; i++) {
        new_strides[ndim - i] = stride;
        stride *= shape[ndim - i];
    }
    
    // 创建新的TensorMeta，保持相同的数据类型
    TensorMeta new_meta{this->dtype(), shape, new_strides};
    
    // 返回一个具有新形状和步长的新张量，共享相同的存储
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 验证维度参数
    if (dim >= this->ndim()) {
        throw std::runtime_error("Dimension out of range");
    }
    
    // 验证起始和结束索引
    if (start >= end || end > this->shape()[dim]) {
        throw std::runtime_error("Invalid start or end indices for slicing");
    }
    
    // 计算新张量的形状
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;
    
    // 计算新张量的偏移量（在存储中的字节偏移）
    size_t new_offset = this->_offset + start * this->strides()[dim] * this->elementSize();
    
    // 计算新张量的步长（保持原来的步长不变）
    std::vector<ptrdiff_t> new_strides = this->strides();
    
    // 创建新的TensorMeta
    TensorMeta new_meta{this->dtype(), new_shape, new_strides};
    
    // 返回一个新的张量，使用相同的存储但不同的偏移和形状
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 添加对存储的空指针检查
    if (!_storage) {
        throw std::runtime_error("Tensor storage is not initialized");
    }
    
    // 获取张量所需的总字节数
    size_t total_bytes = this->numel() * this->elementSize();
    
    // 设置与张量设备匹配的当前设备上下文
    core::context().setDevice(this->deviceType(), this->deviceId());
    
    // 根据张量的设备类型执行内存拷贝
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // 对于CPU张量，执行简单的memcpy操作
        std::memcpy(this->data(), src_, total_bytes);
    } else {
        // 对于GPU或其他设备张量，使用运行时API从主机到设备进行内存拷贝
        core::context().runtime().api()->memcpy_sync(
            this->data(),           // 目标：张量的数据指针
            src_,                   // 源：输入数据指针
            total_bytes,            // 要拷贝的字节数
            LLAISYS_MEMCPY_H2D      // 拷贝方向：主机到设备
        );
    }
}
tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
