#!/usr/bin/env bash
# Fallback build for machines without xmake (the official flow is `xmake && xmake install`).
#
# Produces python/llaisys/libllaisys/libllaisys.{so,dylib} so `import llaisys`
# works straight from the source tree. CUDA sources are not compiled: the nvidia
# backend is still unimplemented, so ENABLE_NVIDIA_API stays off just like with
# xmake's default configuration.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

cxx="${CXX:-clang++}"
case "$(uname -s)" in
Darwin) libname="llaisys.dylib" ;;
*) libname="libllaisys.so" ;;
esac

sources=(
  src/utils/*.cpp
  src/device/*.cpp
  src/device/cpu/*.cpp
  src/core/*/*.cpp
  src/tensor/*.cpp
  src/ops/*/*.cpp
  src/ops/*/cpu/*.cpp
  src/llaisys/tensor.cc
  src/llaisys/ops.cc
  src/llaisys/runtime.cc
  src/llaisys/qwen2.cc
)

"$cxx" -std=c++17 -O2 -fPIC -shared \
  -Wall -Wextra -Werror -Wno-unknown-pragmas -Wno-unused-parameter \
  -Iinclude -Isrc \
  "${sources[@]}" \
  -o "python/llaisys/libllaisys/$libname"

echo "built python/llaisys/libllaisys/$libname"
