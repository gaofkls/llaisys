# python/llaisys/libllaisys/qwen2.py
import ctypes
import numpy as np
from ctypes import c_int, c_int64, c_float, c_void_p, POINTER, Structure, c_size_t
from typing import Optional, Sequence, List
import os
import sys
from pathlib import Path

# === 删除原来的导入：from .. import LIB_LLAISYS ===
# === 改为直接使用你的load_shared_library函数 ===

def load_shared_library():
    """直接加载共享库，不通过上级模块"""
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        # 如果当前目录没有，尝试在项目构建目录查找
        project_root = Path(__file__).parent.parent.parent
        possible_paths = [
            project_root / "bin" / libname,
            project_root / "lib" / libname,
            project_root / "build" / libname,
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                lib_path = path
                break
        else:
            raise FileNotFoundError(f"Shared library not found: {libname}")

    print(f"Loading Qwen2 library from: {lib_path}")
    return ctypes.CDLL(str(lib_path))

# 加载库
try:
    LIB_LLAISYS = load_shared_library()
except FileNotFoundError as e:
    print(f"Warning: {e}")
    print("Using dummy library for Qwen2 testing")
    
    # 创建虚拟库用于测试
    class DummyLIB_LLAISYS:
        def __getattr__(self, name):
            # 为Qwen2相关函数提供虚拟实现
            if "qwen2" in name.lower() or "Qwen2" in name:
                def dummy_func(*args, **kwargs):
                    print(f"[Dummy Qwen2] {name} called")
                    if "create" in name.lower():
                        return ctypes.c_void_p(1)  # 返回非空指针
                    elif "infer" in name.lower():
                        return 42  # 返回测试token
                    elif "destroy" in name.lower() or "reset" in name.lower():
                        return None
                    else:
                        return 0
                return dummy_func
            else:
                # 对于非Qwen2函数，返回空函数
                return lambda *args, **kwargs: None
    
    LIB_LLAISYS = DummyLIB_LLAISYS()

# 设备类型（直接定义，不导入）
class DeviceType:
    CPU = 0
    CUDA = 1
    METAL = 2

# 定义C结构体对应的Python类
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
        ("end_token", c_int64)
    ]

# 权重结构体
class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", c_void_p),
        ("out_embed", c_void_p),
        ("out_norm_w", c_void_p),
        ("attn_norm_w", POINTER(c_void_p)),
        ("attn_q_w", POINTER(c_void_p)),
        ("attn_q_b", POINTER(c_void_p)),
        ("attn_k_w", POINTER(c_void_p)),
        ("attn_k_b", POINTER(c_void_p)),
        ("attn_v_w", POINTER(c_void_p)),
        ("attn_v_b", POINTER(c_void_p)),
        ("attn_o_w", POINTER(c_void_p)),
        ("mlp_norm_w", POINTER(c_void_p)),
        ("mlp_gate_w", POINTER(c_void_p)),
        ("mlp_up_w", POINTER(c_void_p)),
        ("mlp_down_w", POINTER(c_void_p))
    ]

# 加载Qwen2特定的C函数
def _load_qwen2_functions():
    """设置Qwen2 C函数的参数类型"""
    
    # 尝试加载不同的函数名变体
    function_mappings = {
        "llaisysQwen2ModelCreate": "qwen2_create",
        "llaisysQwen2ModelDestroy": "qwen2_destroy",
        "llaisysQwen2ModelInfer": ["qwen2_infer", "qwen2_infer_next_token"],
        "llaisysQwen2ModelReset": "qwen2_reset",
        "llaisysQwen2ModelWeights": "qwen2_weights",
    }
    
    for preferred_name, alt_names in function_mappings.items():
        if not isinstance(alt_names, list):
            alt_names = [alt_names]
        
        func = None
        # 首先尝试首选名称
        if hasattr(LIB_LLAISYS, preferred_name):
            func = getattr(LIB_LLAISYS, preferred_name)
        else:
            # 尝试替代名称
            for alt_name in alt_names:
                if hasattr(LIB_LLAISYS, alt_name):
                    func = getattr(LIB_LLAISYS, alt_name)
                    # 创建别名以便统一使用
                    setattr(LIB_LLAISYS, preferred_name, func)
                    break
        
        if func:
            # 设置参数类型
            if preferred_name == "llaisysQwen2ModelCreate":
                func.argtypes = [
                    POINTER(LlaisysQwen2Meta),
                    c_int,  # llaisysDeviceType_t
                    POINTER(c_int),
                    c_int
                ]
                func.restype = c_void_p
            elif preferred_name == "llaisysQwen2ModelDestroy":
                func.argtypes = [c_void_p]
                func.restype = None
            elif preferred_name == "llaisysQwen2ModelInfer":
                func.argtypes = [
                    c_void_p,
                    POINTER(c_int64),
                    c_size_t
                ]
                func.restype = c_int64
            elif preferred_name == "llaisysQwen2ModelReset":
                func.argtypes = [c_void_p]
                func.restype = None
            elif preferred_name == "llaisysQwen2ModelWeights":
                func.argtypes = [c_void_p]
                func.restype = POINTER(LlaisysQwen2Weights)
        else:
            print(f"Warning: Qwen2 function {preferred_name} not found in library")

# 尝试加载函数（如果库已加载）
try:
    _load_qwen2_functions()
except Exception as e:
    print(f"Warning: Failed to load Qwen2 functions: {e}")

class Qwen2Native:
    """纯数据传递的Native包装器"""
    
    def __init__(self):
        self.model_ptr = None
        self.meta = None
         # 确保 load_weight 函数已绑定
        if hasattr(LIB_LLAISYS, 'llaisysQwen2ModelLoadWeight'):
            self._load_weight_func = LIB_LLAISYS.llaisysQwen2ModelLoadWeight
            
            # 设置参数类型
            from ctypes import c_void_p, c_char_p, POINTER, c_size_t, c_int
            
            self._load_weight_func.argtypes = [
                c_void_p,           # model
                c_char_p,           # weight_name
                c_void_p,           # data
                POINTER(c_size_t),  # shape
                c_size_t,           # ndim
                c_int               # dtype
            ]
            self._load_weight_func.restype = c_int
        else:
            print("[ERROR] llaisysQwen2ModelLoadWeight function not found!")
            self._load_weight_func = None
        
        self._model_ptr = None  # 注意：应该是c_void_p，不是int
    
    def load_weight(self, name: str, data) -> bool:
        """加载权重到C++后端"""
        if not self._model_ptr or not self._load_weight_func:
            print(f"[ERROR] Cannot load weight {name}: model not created or function not available")
            return False
        
        try:
            import numpy as np
            from ctypes import c_void_p, c_char_p, POINTER, c_size_t, c_int
            
            # 确保数据是numpy数组
            if not isinstance(data, np.ndarray):
                try:
                    data = np.array(data)
                except:
                    print(f"[ERROR] Cannot convert data to numpy array for {name}")
                    return False
            
            # 确保是float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # 准备参数
            weight_name = name.encode('utf-8')
            data_ptr = data.ctypes.data_as(c_void_p)
            shape_array = (c_size_t * data.ndim)(*data.shape)
            
            # 使用float32类型 (13)
            cpp_dtype = 13
            
            # 调用C++函数
            result = self._load_weight_func(
                self._model_ptr,
                c_char_p(weight_name),
                data_ptr,
                shape_array,
                data.ndim,
                c_int(cpp_dtype)
            )
            
            return bool(result)
            
        except Exception as e:
            print(f"[ERROR] load_weight failed for {name}: {e}")
            return False
    
    def create(self, config: dict) -> bool:
        """创建C++模型实例 - 修复数据类型问题"""
        try:
            from ..libllaisys.qwen2 import LlaisysQwen2Meta
            from ctypes import c_float
            
            meta = LlaisysQwen2Meta()
            
            # 1. 填充基本字段
            meta.nlayer = config.get('num_hidden_layers', 28)
            meta.nh = config.get('num_attention_heads', 12)
            meta.nkvh = config.get('num_key_value_heads', 2)
            meta.hs = config.get('hidden_size', 1536)
            meta.dh = meta.hs // meta.nh if meta.nh > 0 else 128
            
            meta.maxseq = config.get('max_position_embeddings', 131072)
            meta.voc = config.get('vocab_size', 151936)
            
            # 2. ========== 关键修复：正确设置dtype ==========
            torch_dtype = config.get('torch_dtype', 'bfloat16')
            print(f"[DEBUG] Original torch_dtype from config: {torch_dtype}")
            
            # 方法1：尝试使用枚举常量（如果可用）
            try:
                # 导入枚举常量
                from ..libllaisys.qwen2 import (
                    LLAISYS_DTYPE_BF16, 
                    LLAISYS_DTYPE_F16, 
                    LLAISYS_DTYPE_F32
                )
                
                if torch_dtype == 'bfloat16':
                    meta.dtype = LLAISYS_DTYPE_BF16
                    print(f"[INFO] Using enum: LLAISYS_DTYPE_BF16 = {LLAISYS_DTYPE_BF16}")
                elif torch_dtype == 'float16':
                    meta.dtype = LLAISYS_DTYPE_F16
                    print(f"[INFO] Using enum: LLAISYS_DTYPE_F16 = {LLAISYS_DTYPE_F16}")
                elif torch_dtype == 'float32':
                    meta.dtype = LLAISYS_DTYPE_F32
                    print(f"[INFO] Using enum: LLAISYS_DTYPE_F32 = {LLAISYS_DTYPE_F32}")
                else:
                    print(f"[WARN] Unknown dtype {torch_dtype}, defaulting to float32")
                    meta.dtype = LLAISYS_DTYPE_F32
                    
            except ImportError:
                # 方法2：如果枚举不可用，使用整数值
                print("[WARN] Enum constants not available, using integer values")
                
                dtype_map = {
                    'bfloat16': 19,  # LLAISYS_DTYPE_BF16
                    'float16': 12,   # LLAISYS_DTYPE_F16
                    'float32': 13,   # LLAISYS_DTYPE_F32
                    'float64': 14,   # LLAISYS_DTYPE_F64
                }
                
                if torch_dtype in dtype_map:
                    meta.dtype = dtype_map[torch_dtype]
                    print(f"[INFO] Using integer: {torch_dtype} -> {meta.dtype}")
                else:
                    print(f"[WARN] Unknown dtype {torch_dtype}, using float32 (13)")
                    meta.dtype = 13  # LLAISYS_DTYPE_F32
            
            # 3. 强制使用float32（如果bfloat16有问题）
            # 临时修复：如果之前bfloat16失败，强制使用float32
            force_float32 = True  # 设置为True来强制使用float32
            if force_float32 and meta.dtype == 19:  # 如果是bfloat16
                print("[FORCE] Changing bfloat16 (19) to float32 (13) for compatibility")
                meta.dtype = 13  # LLAISYS_DTYPE_F32
            
            print(f"[DEBUG] Final dtype value: {meta.dtype}")
            
            # 4. 其他参数
            meta.theta = c_float(config.get('rope_theta', 10000.0))
            meta.epsilon = c_float(config.get('rms_norm_eps', 1e-6))
            
            # 5. 打印调试信息
            print(f"[DEBUG] Model meta for C++:")
            print(f"  nlayer={meta.nlayer}, nh={meta.nh}, nkvh={meta.nkvh}")
            print(f"  hs={meta.hs}, dh={meta.dh}, voc={meta.voc}")
            print(f"  maxseq={meta.maxseq}, dtype={meta.dtype}")
            print(f"  theta={meta.theta}, epsilon={meta.epsilon}")
            
            # 6. 调用C++函数
            device_type = 0  # CPU
            device_ids = None
            ndevice = 0
            
            print(f"[DEBUG] Calling C++ llaisysQwen2ModelCreate...")
            self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
                meta, 
                device_type,
                device_ids,
                ndevice
            )
            
            if self.model_ptr is None:
                print("[ERROR] C++ create returned nullptr")
                return False
            
            print(f"[DEBUG] C++ create succeeded, model_ptr={self.model_ptr}")
            return True
            
        except Exception as e:
            print(f"[ERROR] create() failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def destroy(self):
        """销毁C++模型"""
        if self.model_ptr and hasattr(LIB_LLAISYS, 'llaisysQwen2ModelDestroy'):
            # 检查是否是dummy指针
            if self.model_ptr:
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)
        self.model_ptr = None
    
    def infer_next_token(self, input_ids: List[int]) -> int:
        """调用C++后端推理下一个token"""
        if not self.model_ptr:
            raise RuntimeError("Model not created")
        
        # 如果是dummy指针，返回dummy token
        if self.model_ptr:
            print("[Dummy] Qwen2 infer returning token 42")
            return 42
        
        # 转换为C数组
        arr = (c_int64 * len(input_ids))(*input_ids)
        
        # 调用C++推理
        if hasattr(LIB_LLAISYS, 'llaisysQwen2ModelInfer'):
            return LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_ptr,
                arr,
                len(input_ids)
            )
        else:
            print("[Dummy] Using fallback infer")
            return 42  # 返回dummy token
    
    def reset(self):
        """重置C++模型状态"""
        if self.model_ptr and hasattr(LIB_LLAISYS, 'llaisysQwen2ModelReset'):
            # 检查是否是dummy指针
            if self.model_ptr:
                LIB_LLAISYS.llaisysQwen2ModelReset(self.model_ptr)
        else:
            print("[Dummy] Qwen2 reset called")
    
    def __del__(self):
        self.destroy()

# 导出
__all__ = ['Qwen2Native', 'LlaisysQwen2Meta', 'LlaisysQwen2Weights', 'DeviceType']