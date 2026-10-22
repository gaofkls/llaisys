#!/usr/bin/env bash
# Sets up everything test/test_infer.py needs: a Python venv with torch, the
# llaisys package installed from this tree, the shared library built, and the
# DeepSeek-R1-Distill-Qwen-1.5B weights on disk (about 3.1 GB).
#
#   ./scripts/setup_torch_env.sh                    # deps + weights
#   SKIP_DOWNLOAD=1 ./scripts/setup_torch_env.sh    # deps only
#   FORCE=1 ./scripts/setup_torch_env.sh            # redo pip install
#
# Environment overrides:
#   VENV          venv location      (default ~/.venvs/llaisys-torch)
#   PYTHON        interpreter used to create the venv
#   HF_ENDPOINT   model hub          (default https://hf-mirror.com, because
#                                     huggingface.co is unreachable here)
#   PIP_INDEX     pip index url      (default: whatever pip is configured with;
#                                     in mainland China try
#                                     https://pypi.tuna.tsinghua.edu.cn/simple)
#   MODEL_DIR     weight directory   (default ~/models/DeepSeek-R1-Distill-Qwen-1.5B)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

venv="${VENV:-$HOME/.venvs/llaisys-torch}"
python="${PYTHON:-$(command -v python3)}"
endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
model_dir="${MODEL_DIR:-$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B}"
repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "Python 环境：$venv"
if [[ ! -x "$venv/bin/python" ]]; then
  "$python" -m venv "$venv"
  echo "created venv with $("$venv/bin/python" -V)"
else
  echo "reusing $("$venv/bin/python" -V)"
fi
pip="$venv/bin/pip"
python="$venv/bin/python"

if [[ "${FORCE:-0}" == "1" ]] || ! "$python" -c "import torch, transformers" >/dev/null 2>&1; then
  step "安装 torch / transformers / accelerate"
  index_args=()
  [[ -n "${PIP_INDEX:-}" ]] && index_args=(--index-url "$PIP_INDEX")
  "$pip" install --quiet --upgrade pip "${index_args[@]}"
  "$pip" install --retries 5 --timeout 60 "${index_args[@]}" \
    torch transformers accelerate safetensors
else
  echo "torch and transformers already importable, skipping (FORCE=1 to redo)"
fi

step "以可编辑模式安装 llaisys（改了 python/ 下的代码不用重装）"
"$pip" install --quiet --editable "$root/python"

step "确认共享库存在"
lib="python/llaisys/libllaisys/llaisys.dylib"
[[ -f "$lib" ]] || lib="python/llaisys/libllaisys/libllaisys.so"
if [[ ! -f "$lib" ]]; then
  echo "not found, building with scripts/build_local.sh"
  ./scripts/build_local.sh
else
  echo "found $lib"
fi
"$python" -c "import llaisys; print('llaisys imports from', llaisys.__file__)"

if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  step "跳过权重下载（SKIP_DOWNLOAD=1）"
else
  step "下载权重到 $model_dir（约 3.1 GB）"
  mkdir -p "$model_dir"
  if [[ -f "$model_dir/model.safetensors" ]] && [[ "${FORCE:-0}" != "1" ]]; then
    echo "model.safetensors already present, skipping"
  elif [[ -x "$venv/bin/hf" ]]; then
    HF_ENDPOINT="$endpoint" "$venv/bin/hf" download "$repo_id" --local-dir "$model_dir"
  elif [[ -x "$venv/bin/huggingface-cli" ]]; then
    HF_ENDPOINT="$endpoint" "$venv/bin/huggingface-cli" download "$repo_id" --local-dir "$model_dir"
  else
    HF_ENDPOINT="$endpoint" "$python" - "$repo_id" "$model_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
  fi
fi

step "完成"
cat <<EOF
如果 hf download 因为网络或权限失败，可以用可续传的直连下载器代替：
  $python scripts/fetch_model.py "$model_dir"

自检（不依赖 torch，几秒钟）：
  $python test/selfcheck_numpy.py

和 PyTorch 逐 token 对齐（贪婪解码，官方验收方式）：
  $python test/test_infer.py --model "$model_dir" --device cpu --test --max_steps 32

采样生成（可复现）：
  $python test/test_infer.py --model "$model_dir" --device cpu \\
      --top_k 50 --top_p 0.8 --temperature 0.8 --seed 1234 --max_steps 64
EOF
