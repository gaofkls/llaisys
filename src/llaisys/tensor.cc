#include "llaisys_tensor.hpp"

#include "api_guard.hpp"

#include <algorithm>
#include <vector>

__C {
    llaisysTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id) {
        return llaisys::api::run("tensorCreate", static_cast<llaisysTensor_t>(nullptr), [&]() {
            std::vector<size_t> shape_vec(shape, shape + ndim);
            return new LlaisysTensor{
                llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
        });
    }

    void tensorDestroy(
        llaisysTensor_t tensor) {
        llaisys::api::run_void("tensorDestroy", [&]() { delete tensor; });
    }

    void *tensorGetData(
        llaisysTensor_t tensor) {
        return llaisys::api::run("tensorGetData", static_cast<void *>(nullptr), [&]() {
            return static_cast<void *>(tensor->tensor->data());
        });
    }

    size_t tensorGetNdim(
        llaisysTensor_t tensor) {
        return llaisys::api::run("tensorGetNdim", size_t(0),
                                 [&]() { return tensor->tensor->ndim(); });
    }

    void tensorGetShape(
        llaisysTensor_t tensor,
        size_t * shape) {
        llaisys::api::run_void("tensorGetShape", [&]() {
            std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
        });
    }

    void tensorGetStrides(
        llaisysTensor_t tensor,
        ptrdiff_t * strides) {
        llaisys::api::run_void("tensorGetStrides", [&]() {
            std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(),
                      strides);
        });
    }

    llaisysDataType_t tensorGetDataType(
        llaisysTensor_t tensor) {
        return llaisys::api::run("tensorGetDataType", LLAISYS_DTYPE_INVALID,
                                 [&]() { return tensor->tensor->dtype(); });
    }

    llaisysDeviceType_t tensorGetDeviceType(
        llaisysTensor_t tensor) {
        return llaisys::api::run("tensorGetDeviceType", LLAISYS_DEVICE_CPU,
                                 [&]() { return tensor->tensor->deviceType(); });
    }

    int tensorGetDeviceId(
        llaisysTensor_t tensor) {
        return llaisys::api::run("tensorGetDeviceId", -1,
                                 [&]() { return tensor->tensor->deviceId(); });
    }

    void tensorDebug(
        llaisysTensor_t tensor) {
        llaisys::api::run_void("tensorDebug", [&]() { tensor->tensor->debug(); });
    }

    uint8_t tensorIsContiguous(
        llaisysTensor_t tensor) {
        return llaisys::api::run("tensorIsContiguous", uint8_t(0),
                                 [&]() { return uint8_t(tensor->tensor->isContiguous()); });
    }

    void tensorLoad(
        llaisysTensor_t tensor,
        const void *data) {
        llaisys::api::run_void("tensorLoad", [&]() { tensor->tensor->load(data); });
    }

    llaisysTensor_t tensorView(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim) {
        return llaisys::api::run("tensorView", static_cast<llaisysTensor_t>(nullptr), [&]() {
            std::vector<size_t> shape_vec(shape, shape + ndim);
            return new LlaisysTensor{tensor->tensor->view(shape_vec)};
        });
    }

    llaisysTensor_t tensorPermute(
        llaisysTensor_t tensor,
        size_t * order) {
        return llaisys::api::run("tensorPermute", static_cast<llaisysTensor_t>(nullptr), [&]() {
            std::vector<size_t> order_vec(order, order + tensor->tensor->ndim());
            return new LlaisysTensor{tensor->tensor->permute(order_vec)};
        });
    }

    llaisysTensor_t tensorSlice(
        llaisysTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end) {
        return llaisys::api::run("tensorSlice", static_cast<llaisysTensor_t>(nullptr), [&]() {
            return new LlaisysTensor{tensor->tensor->slice(dim, start, end)};
        });
    }
}
