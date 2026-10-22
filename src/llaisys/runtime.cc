#include "llaisys/runtime.h"
#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"

#include "api_guard.hpp"

// Llaisys API for setting context runtime.
__C void llaisysSetContextRuntime(llaisysDeviceType_t device_type, int device_id) {
    llaisys::api::run_void("llaisysSetContextRuntime", [&]() {
        llaisys::core::context().setDevice(device_type, device_id);
    });
}

// Llaisys API for getting the runtime APIs
__C const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
    return llaisys::api::run("llaisysGetRuntimeAPI",
                             static_cast<const LlaisysRuntimeAPI *>(nullptr), [&]() {
                                 return llaisys::device::getRuntimeAPI(device_type);
                             });
}
