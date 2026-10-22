#include "op.hpp"

#include "../../tensor/copy.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    // Same logical content, possibly different strides: this is exactly the
    // strided copy that Tensor::contiguous() relies on.
    copy_strided(out, in);
}
} // namespace llaisys::ops
