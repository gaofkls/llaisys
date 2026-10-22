"""Torch-free sanity checks for the llaisys backend.

The upstream test suite compares against PyTorch. This script instead compares
against small numpy reference implementations, so the backend can be validated on
a machine without torch installed. It covers:

  * the tensor API (view / permute / slice / contiguous / rearrange)
  * every op in f32, plus f16 / bf16 round trips
  * a tiny randomly initialised Qwen2 whose logits are checked against a numpy
    forward pass
  * KV-cache equivalence: token-by-token decoding must match one-shot prefill

Usage:
    python test/selfcheck_numpy.py
"""

import ctypes
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import llaisys  # noqa: E402
from llaisys.libllaisys.qwen2 import Qwen2Native  # noqa: E402

F32 = llaisys.DataType.F32
F16 = llaisys.DataType.F16
BF16 = llaisys.DataType.BF16
I64 = llaisys.DataType.I64


def make_tensor(shape, dtype, data=None):
    tensor = llaisys.Tensor(tuple(shape), dtype, llaisys.DeviceType.CPU)
    if data is not None:
        array = np.ascontiguousarray(data)
        tensor.load(array.ctypes.data)
    return tensor


def as_numpy(tensor):
    """Read an f32 tensor back into numpy, honouring its strides."""
    shape = tensor.shape()
    strides = tensor.strides()
    if not shape:
        count = 1
    else:
        count = 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides))
    buffer = (ctypes.c_float * count)()
    ctypes.memmove(buffer, tensor.data_ptr(), count * 4)
    flat = np.frombuffer(buffer, dtype=np.float32)

    packed = tuple(np.cumprod([1] + list(shape[::-1]))[:-1][::-1]) if shape else ()
    if tuple(strides) == packed:
        return flat[: int(np.prod(shape) if shape else 1)].reshape(shape).copy()

    out = np.empty(shape, dtype=np.float32)
    for index in np.ndindex(*shape):
        out[index] = flat[sum(i * s for i, s in zip(index, strides))]
    return out


def to_bits(array, dtype):
    """Encode float32 data into the storage format of `dtype`."""
    if dtype == F32:
        return array.astype(np.float32)
    if dtype == F16:
        return array.astype(np.float16)
    if dtype == BF16:
        bits = array.astype(np.float32).view(np.uint32) >> 16
        return bits.astype(np.uint16)
    raise ValueError(dtype)


def from_bits(bits, dtype):
    """Decode a raw uint16 storage array into float32."""
    if dtype == F32:
        return np.asarray(bits, dtype=np.float32)
    if dtype == F16:
        return np.ascontiguousarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)
    if dtype == BF16:
        widened = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
        return widened.view(np.float32)
    raise ValueError(dtype)


def quantize(array, dtype):
    """Round float32 data to the precision of `dtype`."""
    if dtype == F32:
        return np.asarray(array, dtype=np.float32)
    if dtype == F16:
        return np.asarray(array, dtype=np.float16).astype(np.float32)
    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32) & 0xFFFF0000
    return bits.view(np.float32)


def check(name, actual, expected, atol, rtol=0.0):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    diff = np.abs(actual - expected) - (atol + rtol * np.abs(expected))
    if np.any(diff > 0):
        worst = int(np.argmax(diff))
        raise AssertionError(
            f"{name}: max abs error {np.max(np.abs(actual - expected)):.3e} "
            f"at flat index {worst} (got {actual.ravel()[worst]}, want {expected.ravel()[worst]})"
        )
    print(f"  ok  {name}")


# ----------------------------------------------------------------- tensor API --


def test_tensor():
    base = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    tensor = make_tensor((3, 4, 5), F32, base)

    check("tensor.load", as_numpy(tensor), base, 0.0)
    assert tensor.is_contiguous()

    view = tensor.view(6, 10)
    check("tensor.view", as_numpy(view), base.reshape(6, 10), 0.0)
    assert tuple(view.strides()) == (10, 1)

    permuted = tensor.permute(2, 0, 1)
    check("tensor.permute", as_numpy(permuted), base.transpose(2, 0, 1), 0.0)
    assert not permuted.is_contiguous()

    sliced = tensor.slice(2, 1, 4)
    check("tensor.slice", as_numpy(sliced), base[:, :, 1:4], 0.0)

    # A sliced dim-0 block of a packed tensor stays contiguous, and rearranage must
    # be able to pack an arbitrary layout (the permuted one) back to C order.
    packed = llaisys.Tensor(tuple(permuted.shape()), F32, llaisys.DeviceType.CPU)
    llaisys.Ops.rearrange(packed, permuted)
    check("tensor.rearrange", as_numpy(packed), np.ascontiguousarray(base.transpose(2, 0, 1)), 0.0)


# ----------------------------------------------------------------------- ops --


def test_ops():
    rng = np.random.default_rng(7)

    # add
    a = rng.standard_normal((64, 96)).astype(np.float32)
    b = rng.standard_normal((64, 96)).astype(np.float32)
    c = make_tensor((64, 96), F32)
    llaisys.Ops.add(c, make_tensor((64, 96), F32, a), make_tensor((64, 96), F32, b))
    check("add", as_numpy(c), a + b, 1e-6, 1e-6)

    # argmax
    vals = rng.standard_normal(1000).astype(np.float32)
    idx = make_tensor((1,), I64)
    val = make_tensor((1,), F32)
    llaisys.Ops.argmax(idx, val, make_tensor((1000,), F32, vals))
    check("argmax.index", as_numpy_i64(idx), [int(np.argmax(vals))], 0.0)

    # embedding
    table = rng.standard_normal((50, 32)).astype(np.float32)
    ids = np.array([3, 0, 49, 7], dtype=np.int64)
    out = make_tensor((4, 32), F32)
    llaisys.Ops.embedding(out, make_tensor((4,), I64, ids), make_tensor((50, 32), F32, table))
    check("embedding", as_numpy(out), table[ids], 0.0)

    # linear, with and without bias
    x = rng.standard_normal((7, 40)).astype(np.float32) * 0.1
    w = rng.standard_normal((24, 40)).astype(np.float32) * 0.1
    bias = rng.standard_normal(24).astype(np.float32)
    y = make_tensor((7, 24), F32)
    llaisys.Ops.linear(y, make_tensor((7, 40), F32, x), make_tensor((24, 40), F32, w),
                       make_tensor((24,), F32, bias))
    expected = x @ w.T + bias
    check("linear.bias", as_numpy(y), expected, 1e-5, 1e-5)

    y2 = make_tensor((7, 24), F32)
    llaisys.Ops.linear(y2, make_tensor((7, 40), F32, x), make_tensor((24, 40), F32, w), None)
    check("linear.nobias", as_numpy(y2), x @ w.T, 1e-5, 1e-5)

    # rms_norm
    x = rng.standard_normal((5, 40)).astype(np.float32)
    gamma = rng.standard_normal(40).astype(np.float32)
    out = make_tensor((5, 40), F32)
    eps = 1e-6
    llaisys.Ops.rms_norm(out, make_tensor((5, 40), F32, x), make_tensor((40,), F32, gamma), eps)
    expected = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * gamma
    check("rms_norm", as_numpy(out), expected, 1e-5, 1e-5)

    # rope
    seq, nhead, dh = 9, 3, 16
    x = rng.standard_normal((seq, nhead, dh)).astype(np.float32)
    pos = np.array([17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=np.int64)
    out = make_tensor((seq, nhead, dh), F32)
    theta = 10000.0
    llaisys.Ops.rope(out, make_tensor((seq, nhead, dh), F32, x), make_tensor((seq,), I64, pos), theta)
    check("rope", as_numpy(out), rope_ref(x, pos, theta), 1e-5, 1e-5)

    # self attention, kv longer than q (KV cache case) and GQA
    qlen, kvlen, nh, nkvh = 4, 11, 4, 2
    q = rng.standard_normal((qlen, nh, dh)).astype(np.float32)
    k = rng.standard_normal((kvlen, nkvh, dh)).astype(np.float32)
    v = rng.standard_normal((kvlen, nkvh, dh)).astype(np.float32)
    scale = 1.0 / np.sqrt(dh)
    out = make_tensor((qlen, nh, dh), F32)
    llaisys.Ops.self_attention(out, make_tensor((qlen, nh, dh), F32, q),
                              make_tensor((kvlen, nkvh, dh), F32, k),
                              make_tensor((kvlen, nkvh, dh), F32, v), scale)
    check("self_attention", as_numpy(out), attention_ref(q, k, v, scale), 1e-5, 1e-5)

    # swiglu
    gate = rng.standard_normal((32, 80)).astype(np.float32)
    up = rng.standard_normal((32, 80)).astype(np.float32)
    out = make_tensor((32, 80), F32)
    llaisys.Ops.swiglu(out, make_tensor((32, 80), F32, gate), make_tensor((32, 80), F32, up))
    check("swiglu", as_numpy(out), up * (gate / (1.0 + np.exp(-gate))), 1e-5, 1e-5)

    # half precision round trips: same reference, looser tolerance
    for dtype, atol in ((F16, 2e-3), (BF16, 2e-2)):
        x = rng.standard_normal((6, 40)).astype(np.float32) * 0.1
        w2 = rng.standard_normal((16, 40)).astype(np.float32) * 0.1
        y = make_tensor((6, 16), dtype)
        llaisys.Ops.linear(
            y,
            make_tensor((6, 40), dtype, to_bits(x, dtype)),
            make_tensor((16, 40), dtype, to_bits(w2, dtype)),
            None,
        )
        expected = quantize(x, dtype) @ quantize(w2, dtype).T
        check(f"linear.{dtype.name.lower()}", from_bits(as_numpy_raw(y, dtype), dtype), expected, atol)

        g = rng.standard_normal((8, 24)).astype(np.float32)
        u = rng.standard_normal((8, 24)).astype(np.float32)
        out = make_tensor((8, 24), dtype)
        llaisys.Ops.swiglu(out, make_tensor((8, 24), dtype, to_bits(g, dtype)),
                           make_tensor((8, 24), dtype, to_bits(u, dtype)))
        gq = quantize(g, dtype)
        uq = quantize(u, dtype)
        check(f"swiglu.{dtype.name.lower()}", from_bits(as_numpy_raw(out, dtype), dtype),
              uq * (gq / (1.0 + np.exp(-gq))), atol)


def as_numpy_i64(tensor):
    count = int(np.prod(tensor.shape()))
    buffer = (ctypes.c_int64 * count)()
    ctypes.memmove(buffer, tensor.data_ptr(), count * 8)
    return np.array(buffer, dtype=np.int64)


def as_numpy_raw(tensor, dtype):
    count = int(np.prod(tensor.shape()))
    if dtype == F32:
        buffer = (ctypes.c_float * count)()
        ctypes.memmove(buffer, tensor.data_ptr(), count * 4)
        return np.array(buffer, dtype=np.float32).reshape(tensor.shape())
    buffer = (ctypes.c_uint16 * count)()
    ctypes.memmove(buffer, tensor.data_ptr(), count * 2)
    return np.array(buffer, dtype=np.uint16).reshape(tensor.shape())


# ------------------------------------------------------------------ reference --


def rope_ref(x, pos, theta):
    seq, nhead, dh = x.shape
    half = dh // 2
    inv = theta ** (-2.0 * np.arange(half) / dh)
    angles = pos.astype(np.float32)[:, None] * inv[None, :]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def attention_ref(q, k, v, scale):
    qlen, nh, dh = q.shape
    kvlen, nkvh, _ = k.shape
    group = nh // nkvh
    kk = np.repeat(k, group, axis=1)
    vv = np.repeat(v, group, axis=1)
    scores = np.einsum("lhd,shd->hls", q, kk) * scale
    mask = np.tril(np.ones((qlen, kvlen), dtype=bool), k=kvlen - qlen)
    scores = np.where(mask[None, :, :], scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("hls,shd->lhd", probs, vv)


# ----------------------------------------------------------------------- model --


class TinyQwen2:
    """A hand-rolled Qwen2 forward pass used as the oracle for the C++ backend."""

    def __init__(self, config, seed=11):
        rng = np.random.default_rng(seed)
        self.cfg = config
        hs, nh, nkvh, dh = (config["hidden_size"], config["num_attention_heads"],
                            config["num_key_value_heads"], config["hidden_size"] // config["num_attention_heads"])
        di, voc, nlayer = config["intermediate_size"], config["vocab_size"], config["num_hidden_layers"]

        def rand(*shape):
            return (rng.standard_normal(shape) * 0.08).astype(np.float32)

        self.w = {
            "model.embed_tokens.weight": rand(voc, hs),
            "lm_head.weight": rand(voc, hs),
            "model.norm.weight": (1.0 + rng.standard_normal(hs) * 0.02).astype(np.float32),
        }
        for layer in range(nlayer):
            p = f"model.layers.{layer}."
            self.w[p + "input_layernorm.weight"] = (1.0 + rng.standard_normal(hs) * 0.02).astype(np.float32)
            self.w[p + "post_attention_layernorm.weight"] = (1.0 + rng.standard_normal(hs) * 0.02).astype(np.float32)
            self.w[p + "self_attn.q_proj.weight"] = rand(nh * dh, hs)
            self.w[p + "self_attn.q_proj.bias"] = rand(nh * dh)
            self.w[p + "self_attn.k_proj.weight"] = rand(nkvh * dh, hs)
            self.w[p + "self_attn.k_proj.bias"] = rand(nkvh * dh)
            self.w[p + "self_attn.v_proj.weight"] = rand(nkvh * dh, hs)
            self.w[p + "self_attn.v_proj.bias"] = rand(nkvh * dh)
            self.w[p + "self_attn.o_proj.weight"] = rand(hs, nh * dh)
            self.w[p + "mlp.gate_proj.weight"] = rand(di, hs)
            self.w[p + "mlp.up_proj.weight"] = rand(di, hs)
            self.w[p + "mlp.down_proj.weight"] = rand(hs, di)

    def logits(self, tokens):
        cfg = self.cfg
        eps = cfg["rms_norm_eps"]
        theta = cfg["rope_theta"]
        nh, nkvh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hs = cfg["hidden_size"]
        dh = hs // nh
        scale = 1.0 / np.sqrt(dh)

        def rms(x, w):
            return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

        pos = np.arange(len(tokens), dtype=np.int64)
        x = self.w["model.embed_tokens.weight"][tokens]

        for layer in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{layer}."
            h = rms(x, self.w[p + "input_layernorm.weight"])
            q = (h @ self.w[p + "self_attn.q_proj.weight"].T + self.w[p + "self_attn.q_proj.bias"]).reshape(len(tokens), nh, dh)
            k = (h @ self.w[p + "self_attn.k_proj.weight"].T + self.w[p + "self_attn.k_proj.bias"]).reshape(len(tokens), nkvh, dh)
            v = (h @ self.w[p + "self_attn.v_proj.weight"].T + self.w[p + "self_attn.v_proj.bias"]).reshape(len(tokens), nkvh, dh)
            q, k = rope_ref(q, pos, theta), rope_ref(k, pos, theta)
            attn = attention_ref(q, k, v, scale).reshape(len(tokens), hs)
            x = x + attn @ self.w[p + "self_attn.o_proj.weight"].T

            h = rms(x, self.w[p + "post_attention_layernorm.weight"])
            gate = h @ self.w[p + "mlp.gate_proj.weight"].T
            up = h @ self.w[p + "mlp.up_proj.weight"].T
            act = up * (gate / (1.0 + np.exp(-gate)))
            x = x + act @ self.w[p + "mlp.down_proj.weight"].T

        x = rms(x, self.w["model.norm.weight"])
        return x @ self.w["lm_head.weight"].T


def test_model():
    config = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 100,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "eos_token_id": 99,
    }
    oracle = TinyQwen2(config)

    native = Qwen2Native()
    assert native.create(config), "model creation failed"

    for name, array in oracle.w.items():
        address = array.ctypes.data
        ok = native.load_weight(name, address, array.shape, int(F32))
        assert ok, f"failed to load {name}"

    rng = np.random.default_rng(3)
    prompts = [
        [1],
        [5, 9, 17],
        [2, 4, 6, 8, 10, 12, 14],
    ]

    for prompt in prompts:
        native.reset()
        token = native.infer_next_token(prompt)
        expected = int(np.argmax(oracle.logits(np.array(prompt, dtype=np.int64))[-1]))
        check(f"model.prefill prompt={len(prompt)}", [token], [expected], 0.0)

        # Continue two more steps: this is where a broken cache or a broken RoPE
        # position shows up.
        history = list(prompt)
        for step in range(2):
            history.append(token)
            token = native.infer_next_token(history)
            expected = int(np.argmax(oracle.logits(np.array(history, dtype=np.int64))[-1]))
            check(f"model.decode prompt={len(prompt)} step={step}", [token], [expected], 0.0)

    # KV-cache equivalence: the incremental path must equal the one-shot prefill.
    sequence = [3, 1, 4, 1, 5, 9, 2, 6]
    native.reset()
    one_shot = native.infer_next_token(sequence)
    native.reset()
    incremental = None
    for end in range(1, len(sequence) + 1):
        incremental = native.infer_next_token(sequence[:end])
    assert incremental == one_shot, f"cache mismatch: {incremental} != {one_shot}"
    print("  ok  kv-cache equivalence (incremental == one-shot prefill)")

    native.destroy()


def write_safetensors(path, tensors, dtype=BF16):
    """Write a minimal safetensors file so the loader can be exercised for real."""
    header = {"__metadata__": {"format": "pt"}}
    blobs = []
    offset = 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(to_bits(array, dtype)).astype("<u2").tobytes()
        header[name] = {
            "dtype": "BF16" if dtype == BF16 else ("F16" if dtype == F16 else "F32"),
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)

    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def test_safetensors_model():
    """End-to-end: bf16 safetensors on disk -> Python loader -> C++ bf16 model."""
    import tempfile

    config = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 100,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "torch_dtype": "bfloat16",
        "eos_token_id": 99,
    }
    oracle = TinyQwen2(config)
    # The oracle must see exactly the weights that were written to disk.
    for name, array in oracle.w.items():
        oracle.w[name] = quantize(array, BF16)

    with tempfile.TemporaryDirectory() as directory:
        write_safetensors(os.path.join(directory, "model.safetensors"), oracle.w)
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle)

        model = llaisys.models.Qwen2(directory)
        prompt = [5, 9, 17, 21]
        tokens = model.generate(prompt, max_new_tokens=1)
        expected = int(np.argmax(oracle.logits(np.array(prompt, dtype=np.int64))[-1]))
        check("safetensors.bf16-generate", [tokens[-1]], [expected], 0.0)
        del model


def test_cache_growth():
    """A sequence longer than the initial KV reservation must still be correct."""
    config = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 100,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "eos_token_id": 99,
    }
    oracle = TinyQwen2(config)

    incremental_model = Qwen2Native()
    one_shot_model = Qwen2Native()
    for model in (incremental_model, one_shot_model):
        assert model.create(config)
        for name, array in oracle.w.items():
            assert model.load_weight(name, array.ctypes.data, array.shape, int(F32))

    sequence = [int(token) for token in np.random.default_rng(5).integers(0, 99, 260)]

    one_shot_model.reset()
    one_shot = one_shot_model.infer_next_token(sequence)

    incremental_model.reset()
    incremental = None
    for end in range(1, len(sequence) + 1):
        incremental = incremental_model.infer_next_token(sequence[:end])

    assert incremental == one_shot, f"cache growth mismatch: {incremental} != {one_shot}"
    expected = int(np.argmax(oracle.logits(np.array(sequence, dtype=np.int64))[-1]))
    check("kv-cache growth (260 tokens)", [incremental], [expected], 0.0)

    incremental_model.destroy()
    one_shot_model.destroy()


def reference_probs(logits, temperature, top_k, top_p):
    """Distribution the sample operator must reproduce."""
    values = np.asarray(logits, dtype=np.float64)
    scaled = (values - values.max()) / temperature
    order = np.argsort(-scaled, kind="stable")
    if 1 < top_k < len(values):
        order = order[:top_k]
    weights = np.exp(scaled[order])
    probs = weights / weights.sum()
    if 0.0 < top_p < 1.0:
        cumulative = np.cumsum(probs)
        cut = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        cut = min(cut, len(probs))
        order, probs = order[:cut], probs[:cut]
        probs = probs / probs.sum()
    full = np.zeros(len(values))
    full[order] = probs
    return full


def sample_counts(vals, token_id, temperature, top_k, top_p, draws, seed):
    state = ctypes.c_uint64(seed)
    counts = np.zeros(int(np.prod(vals.shape())), dtype=np.int64)
    for _ in range(draws):
        llaisys.Ops.sample(token_id, vals, temperature, top_k, top_p, state)
        counts[ctypes.c_int64.from_address(token_id.data_ptr()).value] += 1
    return counts


def test_sample():
    rng = np.random.default_rng(11)
    n = 32
    draws = 20000

    logits = (rng.standard_normal(n) * 1.5).astype(np.float32)
    vals = make_tensor((n,), F32, logits)
    token_id = make_tensor((1,), I64)

    cases = [
        ("temperature 1.0, no truncation", 1.0, 0, 1.0),
        ("temperature 0.7 + top_k 8", 0.7, 8, 1.0),
        ("top_p 0.9", 1.0, 0, 0.9),
        ("temperature 1.3 + top_k 8 + top_p 0.85", 1.3, 8, 0.85),
    ]
    for name, temperature, top_k, top_p in cases:
        expected = reference_probs(logits, temperature, top_k, top_p)
        counts = sample_counts(vals, token_id, temperature, top_k, top_p, draws, 2024)
        observed = counts / draws

        # 5 sigma per category. With 32 categories and 20k draws the false
        # positive rate is far below one in ten thousand runs.
        sigma = np.sqrt(expected * (1.0 - expected) / draws)
        tolerance = 5.0 * sigma + 1.0 / draws
        worst = np.max(np.abs(observed - expected) - tolerance)
        if worst > 0:
            index = int(np.argmax(np.abs(observed - expected) - tolerance))
            raise AssertionError(
                f"sample[{name}]: token {index} observed {observed[index]:.4f}, "
                f"expected {expected[index]:.4f} (tolerance {tolerance[index]:.4f})"
            )
        assert np.all(counts[expected == 0.0] == 0), f"sample[{name}]: drew from outside the support"
        print(f"  ok  sample[{name}]")

    # Same seed replays the same sequence, different seeds do not.
    replay_a = sample_counts(vals, token_id, 1.0, 0, 1.0, 8, 777)
    replay_b = sample_counts(vals, token_id, 1.0, 0, 1.0, 8, 777)
    other = sample_counts(vals, token_id, 1.0, 0, 1.0, 8, 778)
    assert np.array_equal(replay_a, replay_b), "sample: same seed produced a different sequence"
    assert not np.array_equal(replay_a, other), "sample: different seeds produced the same sequence"
    print("  ok  sample[seeded replay]")

    # Degenerate settings must agree with argmax, which is what the assignment's
    # correctness test relies on.
    best = int(np.argmax(logits))
    for name, temperature, top_k in (("temperature 0", 0.0, 50), ("top_k 1", 1.0, 1)):
        got = set(sample_counts(vals, token_id, temperature, top_k, 1.0, 64, 5).nonzero()[0].tolist())
        assert got == {best}, f"sample[{name}] returned {got}, greedy would return {best}"
        print(f"  ok  sample[{name} == argmax]")

    # Half precision logits take the same path.
    for dtype in (F16, BF16):
        half_vals = make_tensor((n,), dtype, to_bits(logits, dtype))
        reference = quantize(logits, dtype)
        counts = sample_counts(half_vals, token_id, 1.0, 4, 1.0, 200, 31)
        got = set(counts.nonzero()[0].tolist())
        allowed = set(np.argsort(-reference, kind="stable")[:4].tolist())
        assert got <= allowed, f"sample[{dtype.name}]: {got - allowed} outside top-k"
        print(f"  ok  sample[{dtype.name.lower()} logits]")


def test_model_sampling():
    """The model path must pass temperature/top-k/top-p through to the op."""
    config = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 100,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "eos_token_id": 99,
    }
    oracle = TinyQwen2(config)
    model = Qwen2Native()
    assert model.create(config)
    for name, array in oracle.w.items():
        assert model.load_weight(name, array.ctypes.data, array.shape, int(F32))

    prompt = [5, 9, 17]
    steps = 6

    def greedy():
        model.reset()
        history = list(prompt)
        for _ in range(steps):
            history.append(model.infer_next_token(history))
        return history

    def sampled(temperature, top_k, top_p, seed):
        model.reset()
        history = list(prompt)
        state = ctypes.c_uint64(seed)
        for _ in range(steps):
            history.append(
                model.infer_next_token_sample(history, temperature, top_k, top_p, state)
            )
        return history

    baseline = greedy()
    assert sampled(0.8, 50, 0.9, 1234) == sampled(0.8, 50, 0.9, 1234), "model sampling is not reproducible"
    print("  ok  model sampling[seeded replay]")

    assert sampled(0.0, 50, 0.9, 1234) == baseline, "temperature 0 must equal greedy"
    assert sampled(0.8, 1, 0.9, 1234) == baseline, "top_k 1 must equal greedy"
    print("  ok  model sampling[greedy equivalence]")

    runs = {tuple(sampled(0.8, 50, 0.9, seed)) for seed in range(4)}
    assert len(runs) > 1, "four different seeds produced identical text"
    assert all(0 <= token < config["vocab_size"] for run in runs for token in run), "token out of range"
    print("  ok  model sampling[different seeds differ]")

    model.destroy()


if __name__ == "__main__":
    print("tensor API")
    test_tensor()
    print("ops")
    test_ops()
    print("sample")
    test_sample()
    print("model")
    test_model()
    print("model sampling")
    test_model_sampling()
    print("kv-cache growth")
    test_cache_growth()
    print("safetensors loader")
    test_safetensors_model()
    print("\033[92mAll self-checks passed\033[0m")
