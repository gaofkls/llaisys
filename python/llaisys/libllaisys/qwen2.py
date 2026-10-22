import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int64,
    c_size_t,
    c_uint64,
    c_void_p,
)
from types import SimpleNamespace

from . import LIB_LLAISYS
from .llaisys_types import DataType, DeviceType

# <safetensors dtype string> -> llaisysDataType_t
SAFETENSORS_DTYPES = {
    "F32": int(DataType.F32),
    "F16": int(DataType.F16),
    "BF16": int(DataType.BF16),
    "F64": int(DataType.F64),
}

# <config.json torch_dtype> -> llaisysDataType_t
TORCH_DTYPES = {
    "float32": int(DataType.F32),
    "float16": int(DataType.F16),
    "bfloat16": int(DataType.BF16),
}


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


def _bind():
    LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        c_int,
        POINTER(c_int),
        c_int,
    ]
    LIB_LLAISYS.llaisysQwen2ModelCreate.restype = c_void_p

    LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

    LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
    LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64

    LIB_LLAISYS.llaisysQwen2ModelInferSample.argtypes = [
        c_void_p,
        POINTER(c_int64),
        c_size_t,
        c_float,
        c_size_t,
        c_float,
        POINTER(c_uint64),
    ]
    LIB_LLAISYS.llaisysQwen2ModelInferSample.restype = c_int64

    LIB_LLAISYS.llaisysQwen2ModelReset.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2ModelReset.restype = None

    LIB_LLAISYS.llaisysQwen2ModelLoadWeight.argtypes = [
        c_void_p,
        c_char_p,
        c_void_p,
        POINTER(c_size_t),
        c_size_t,
        c_int,
    ]
    LIB_LLAISYS.llaisysQwen2ModelLoadWeight.restype = c_bool


_bind()


def _first_int(value, default=0):
    """config.json sometimes stores eos_token_id as a list."""
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else default
    return int(value) if value is not None else default


class Qwen2Native:
    """Thin ctypes wrapper around the C++ Qwen2 model."""

    def __init__(self):
        self._handle = None
        self.meta = None

    # ------------------------------------------------------------------ setup --

    def create(self, config, device=DeviceType.CPU, device_id=0):
        nlayer = int(config.get("num_hidden_layers", 28))
        nh = int(config.get("num_attention_heads", 12))
        nkvh = int(config.get("num_key_value_heads", nh))
        hs = int(config.get("hidden_size", 1536))
        if hs % nh != 0:
            print(f"[llaisys] hidden_size {hs} is not divisible by heads {nh}")
            return False
        dh = hs // nh

        torch_dtype = str(config.get("torch_dtype", "float32"))
        if torch_dtype not in TORCH_DTYPES:
            print(f"[llaisys] unsupported torch_dtype '{torch_dtype}', using float32")
        dtype = TORCH_DTYPES.get(torch_dtype, int(DataType.F32))

        eos = _first_int(config.get("eos_token_id"), 0)

        meta = LlaisysQwen2Meta()
        meta.dtype = dtype
        meta.nlayer = nlayer
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = nkvh
        meta.dh = dh
        meta.di = int(config.get("intermediate_size", 4 * hs))
        meta.maxseq = int(config.get("max_position_embeddings", 32768))
        meta.voc = int(config.get("vocab_size", 151936))
        meta.epsilon = float(config.get("rms_norm_eps", 1e-6))
        meta.theta = float(config.get("rope_theta", 10000.0))
        meta.end_token = eos

        device_ids = (c_int * 1)(int(device_id))
        self._handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), int(device), device_ids, 1
        )
        if not self._handle:
            print("[llaisys] failed to create the Qwen2 model")
            return False

        self.meta = SimpleNamespace(
            dtype=dtype,
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=meta.di,
            maxseq=meta.maxseq,
            voc=meta.voc,
            epsilon=meta.epsilon,
            theta=meta.theta,
            eos_token_id=eos,
        )
        return True

    # ---------------------------------------------------------------- weights --

    def load_weight(self, name, address, shape, dtype):
        """`address` is the host address of the raw tensor bytes (no copy made)."""
        if not self._handle:
            raise RuntimeError("load_weight called before create()")
        shape_array = (c_size_t * len(shape))(*shape)
        return bool(
            LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                self._handle,
                name.encode("utf-8"),
                c_void_p(address),
                shape_array,
                len(shape),
                int(dtype),
            )
        )

    # --------------------------------------------------------------- inference --

    def infer_next_token(self, token_ids):
        """Pass the whole sequence seen so far; the model reuses its KV cache."""
        ids = list(token_ids)
        if not ids:
            raise ValueError("infer_next_token needs at least one token")
        array = (c_int64 * len(ids))(*ids)
        token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._handle, array, len(ids)))
        if token < 0:
            raise RuntimeError("llaisysQwen2ModelInfer failed, see the log above")
        return token

    def infer_next_token_sample(self, token_ids, temperature, top_k, top_p, rng_state):
        """Same as `infer_next_token`, but draws the token from the logits.

        `rng_state` is a ctypes.c_uint64; the backend advances it in place, so
        holding on to the same object is what makes a sequence reproducible.
        """
        ids = list(token_ids)
        if not ids:
            raise ValueError("infer_next_token_sample needs at least one token")
        array = (c_int64 * len(ids))(*ids)
        token = int(
            LIB_LLAISYS.llaisysQwen2ModelInferSample(
                self._handle,
                array,
                len(ids),
                float(temperature),
                int(top_k),
                float(top_p),
                ctypes.byref(rng_state),
            )
        )
        if token < 0:
            raise RuntimeError("llaisysQwen2ModelInferSample failed, see the log above")
        return token

    def reset(self):
        if self._handle:
            LIB_LLAISYS.llaisysQwen2ModelReset(self._handle)

    def destroy(self):
        if self._handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


__all__ = ["Qwen2Native", "LlaisysQwen2Meta", "SAFETENSORS_DTYPES", "TORCH_DTYPES"]
