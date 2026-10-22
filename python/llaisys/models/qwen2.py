import ctypes
import json
import mmap
import random
import struct
from pathlib import Path
from typing import List, Optional, Sequence

from ..libllaisys import DeviceType
from ..libllaisys.qwen2 import SAFETENSORS_DTYPES, Qwen2Native


class Qwen2:
    """Qwen2 inference front-end: reads safetensors, runs the C++ backend."""

    def __init__(self, model_path, device=DeviceType.CPU, device_id: int = 0):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        config_file = path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {path}")
        with open(config_file, "r", encoding="utf-8") as handle:
            self.config = json.load(handle)

        self._native = Qwen2Native()
        if not self._native.create(self.config, device, device_id):
            raise RuntimeError("Failed to create the Qwen2 model in the C++ backend")

        # Seed used by the most recent generate(..., seed=None) call, so a
        # random run can be replayed afterwards.
        self.last_seed = None

        self._load_weights(path)

    # ---------------------------------------------------------------- weights --

    def _load_weights(self, path: Path):
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No *.safetensors files found in {path}")

        total = 0
        for file_path in files:
            total += self._load_safetensors(file_path)
        print(f"Loaded {total} tensors from {len(files)} file(s)")

    def _load_safetensors(self, file_path: Path) -> int:
        """Stream one safetensors file into the backend without materialising it.

        The file is memory-mapped copy-on-write so even a 3 GB checkpoint never
        lands in the Python heap; the backend copies straight out of the mapping.
        """
        with open(file_path, "rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
            data_base = 8 + header_size

            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
            loaded = 0
            try:
                for name, info in header.items():
                    if name == "__metadata__":
                        continue

                    dtype_str = info["dtype"]
                    if dtype_str not in SAFETENSORS_DTYPES:
                        raise ValueError(f"unsupported dtype {dtype_str} for {name}")

                    start, end = info["data_offsets"]
                    if end == start:
                        continue

                    shape = tuple(int(dim) for dim in info["shape"])
                    buffer = (ctypes.c_char * (end - start)).from_buffer(
                        mapping, data_base + start
                    )
                    ok = self._native.load_weight(
                        name,
                        ctypes.addressof(buffer),
                        shape,
                        SAFETENSORS_DTYPES[dtype_str],
                    )
                    del buffer  # release the mmap buffer before closing it

                    if not ok:
                        raise RuntimeError(f"Backend rejected weight '{name}' from {file_path.name}")
                    loaded += 1
            finally:
                try:
                    mapping.close()
                except BufferError:
                    pass

        return loaded

    # -------------------------------------------------------------- inference --

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: Optional[int] = None,
    ) -> List[int]:
        """Generate tokens with top-k / top-p / temperature sampling.

        Greedy decoding is selected by `top_k == 1` or `temperature <= 0`, which
        is also what the test harness uses when it compares against PyTorch.
        Sampling runs entirely in the C++ backend (`llaisysSample`); `seed`
        makes a run reproducible, and `self.last_seed` records what was used.
        """
        if max_new_tokens is None:
            max_new_tokens = 512
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")

        use_sampling = not (top_k == 1 or temperature <= 0.0)
        if use_sampling:
            if seed is None:
                seed = random.SystemRandom().getrandbits(63)
            # One 64-bit word of generator state, advanced in place by the
            # backend. Kept per generate() call so a run is self contained.
            rng_state = ctypes.c_uint64(seed)
        else:
            rng_state = None
        self.last_seed = seed

        self._native.reset()

        history = list(int(token) for token in inputs)
        if not history:
            raise ValueError("generate() needs at least one prompt token")

        generated: List[int] = []
        eos = self._native.meta.eos_token_id

        for _ in range(max_new_tokens):
            # The whole sequence is handed over each step: the backend keeps the
            # KV cache and only processes tokens it has not seen yet.
            if use_sampling:
                next_token = self._native.infer_next_token_sample(
                    history, temperature, top_k, top_p, rng_state
                )
            else:
                next_token = self._native.infer_next_token(history)
            if next_token == eos:
                break
            generated.append(next_token)
            history.append(next_token)

        # history == prompt + generated, all plain Python ints
        return history

    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        raise NotImplementedError(
            "This method needs a tokenizer; call generate() with token ids instead."
        )

    def reset(self):
        self._native.reset()

    def __del__(self):
        native = getattr(self, "_native", None)
        if native is not None:
            native.destroy()
