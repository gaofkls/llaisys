#include "op.hpp"

#include "../op_common.hpp"
#include "../../tensor/copy.hpp"
#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be int64.");
    CHECK_ARGUMENT(index->ndim() == 1, "embedding: index must be 1-D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "embedding: weight must be 2-D.");
    CHECK_ARGUMENT(out->ndim() == 2, "embedding: output must be 2-D.");
    ASSERT(out->isContiguous() && weight->isContiguous() && index->isContiguous(),
           "embedding: all tensors must be contiguous.");

    const size_t ntoken = index->shape()[0];
    const size_t vocab = weight->shape()[0];
    const size_t dim = weight->shape()[1];

    CHECK_ARGUMENT(out->shape()[0] == ntoken && out->shape()[1] == dim,
                   "embedding: output shape must be [ntoken, dim].");

    core::context().setDevice(out->deviceType(), out->deviceId());

    const size_t row_bytes = dim * out->elementSize();
    auto *idx = reinterpret_cast<const int64_t *>(index->data());
    std::byte *dst = out->data();
    const std::byte *src = weight->data();

    for (size_t i = 0; i < ntoken; ++i) {
        const int64_t token = idx[i];
        CHECK_ARGUMENT(token >= 0 && static_cast<size_t>(token) < vocab,
                       "embedding: token id out of range.");
        copy_run(dst + i * row_bytes, src + static_cast<size_t>(token) * row_bytes,
                 row_bytes, out->deviceType(), weight->deviceType(), out->deviceId());
    }
}

} // namespace llaisys::ops
