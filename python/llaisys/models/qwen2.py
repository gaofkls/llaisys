from typing import Sequence, Optional, List, Union
from pathlib import Path
import json
import numpy as np

from ..libllaisys import LIB_LLAISYS
from ..libllaisys.qwen2 import Qwen2Native


class Qwen2:
    """纯数据加载和调用的Qwen2包装器"""
    
    def __init__(self, model_path, device=None):
        """
        初始化Qwen2模型
        
        Args:
            model_path: 模型路径
            device: 设备类型（可选参数，为了兼容测试脚本）
        """
        # 如果传入了device参数，记录但不使用（C++后端可能还不支持）
        if device is not None:
            print(f"Note: Device parameter '{device}' provided but may not be used")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        print(f"Loading Qwen2 model from {model_path}")
        
        # 1. 加载配置
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 诊断代码块 - 应该移除或作为单独函数存在
        print("=" * 60)
        print("立即诊断：检查C++后端问题")
        print("=" * 60)
        
        # 1. 检查C++库
        print("\n1. 检查C++库...")
        try:
            # 检查LIB_LLAISYS是否已加载
            if LIB_LLAISYS is None:
                print("  ✗ LIB_LLAISYS is None")
            else:
                print(f"  ✓ LIB_LLAISYS 对象: {type(LIB_LLAISYS)}")
                
                # 检查关键函数是否存在
                key_functions = [
                    'llaisysQwen2ModelCreate',
                    'llaisysQwen2ModelDestroy',
                    'llaisysQwen2ModelLoadWeight',
                    'llaisysQwen2ModelInfer',
                    'llaisysQwen2ModelReset'
                ]
                
                for func_name in key_functions:
                    if hasattr(LIB_LLAISYS, func_name):
                        func = getattr(LIB_LLAISYS, func_name)
                        print(f"  ✓ {func_name}: {hex(id(func)) if func else 'None'}")
                    else:
                        print(f"  ✗ 缺少函数: {func_name}")
                        
        except Exception as e:
            print(f"  ✗ 检查C++库时出错: {e}")
        
        # 2. 检查配置
        print("\n2. 检查配置...")
        print(f"  配置类型: {type(self.config)}")
        print(f"  配置内容前5项:")
        for i, (key, value) in enumerate(list(self.config.items())[:5]):
            print(f"    {key}: {value}")
        
        # 检查必需字段
        required_fields = [
            'num_hidden_layers',
            'hidden_size', 
            'num_attention_heads',
            'vocab_size',
            'max_position_embeddings'
        ]
        
        for field in required_fields:
            if field in self.config:
                print(f"  ✓ {field}: {self.config[field]}")
            else:
                print(f"  ✗ 缺少必需字段: {field}")
        
        # 3. 尝试导入C结构体
        print("\n3. 检查C结构体导入...")
        try:
            # 尝试导入LlaisysQwen2Meta
            from ..libllaisys.qwen2 import LlaisysQwen2Meta
            
            # 创建一个测试实例
            test_meta = LlaisysQwen2Meta()
            print(f"  ✓ 可以创建 LlaisysQwen2Meta")
            
            # 尝试设置字段
            test_meta.nlayer = 24
            test_meta.nh = 32
            test_meta.hs = 2048
            print(f"  ✓ 可以设置结构体字段")
            
        except ImportError as e:
            print(f"  ✗ 无法导入 LlaisysQwen2Meta: {e}")
        except Exception as e:
            print(f"  ✗ 创建结构体时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. 检查数据类型配置
        print("\n4. 检查数据类型...")
        torch_dtype = self.config.get('torch_dtype', 'unknown')
        print(f"  torch_dtype: {torch_dtype}")
        
        # 5. 打印完整配置（调试用）
        print("\n5. 完整配置摘要:")
        for key in sorted(self.config.keys()):
            value = self.config[key]
            if isinstance(value, (int, float, str, bool)):
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) <= 3:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value)}")
        
        print("\n" + "=" * 60)
        print("诊断完成")
        print("=" * 60 + "\n")
        
        ##################################################################
        # === 立即诊断代码到这里结束 ===
        ##################################################################    
            
        # 2. 创建C++模型
        self._native = Qwen2Native()
        if not self._native.create(self.config):
            raise RuntimeError("Failed to create model in C++ backend")
        
        # 3. 加载所有权重到C++后端
        self._load_all_weights(model_path)
        
        print(f"Qwen2 model loaded: {self.config.get('num_hidden_layers', 'N/A')} layers")
    
    def _load_config(self, model_path: Path) -> dict:
        """加载模型配置"""
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_all_weights(self, model_path: Path):
        """加载所有权重文件到C++后端"""
        import safetensors
        
        # 找到所有safetensors文件
        weight_files = list(model_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        
        print(f"Loading {len(weight_files)} weight files to C++ backend...")
        
        # 加载每个文件
        for file_path in weight_files:
            self._load_weight_file(file_path)
    
    def _load_weight_file(self, file_path: Path):
        """加载单个权重文件到C++后端"""
        import safetensors
        
        print(f"  Loading {file_path.name}...")
        
        with safetensors.safe_open(str(file_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                try:
                    np_array = f.get_tensor(key).numpy()
                    
                    # 添加更多数据类型验证和转换
                    if np_array.dtype in [np.float16, np.bfloat16]:
                        # 转换为float32以避免C++后端数据类型错误
                        np_array = np_array.astype(np.float32)
                    elif np_array.dtype == np.int64:
                        # 某些整数类型可能需要转换为int32
                        np_array = np_array.astype(np.int32)
                    elif np_array.dtype not in [np.float32, np.int32, np.int8, np.uint8]:
                        # 确保C++后端支持的数据类型
                        print(f"    Converting {key} from {np_array.dtype} to supported type")
                        if np.issubdtype(np_array.dtype, np.floating):
                            np_array = np_array.astype(np.float32)
                        elif np.issubdtype(np_array.dtype, np.integer):
                            np_array = np_array.astype(np.int32)
                    
                    success = self._native.load_weight(key, np_array)
                    
                    if not success:
                        print(f"    Warning: Failed to load weight {key}")
                    else:
                        print(f"    Loaded: {key} {np_array.shape}")
                        
                except Exception as e:
                    print(f"    Error loading {key}: {e}")
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        """
        生成文本 - 所有推理在C++后端完成
        
        Args:
            inputs: 输入token IDs
            max_new_tokens: 最大生成token数
            top_k: top-k采样（C++端实现）
            top_p: top-p采样（C++端实现）
            temperature: 温度（C++端实现）
            
        Returns:
            生成的token IDs
        """
        if max_new_tokens is None:
            max_new_tokens = 512
        
        # 重置C++模型状态
        self._native.reset()
        
        # 生成循环
        generated = []
        current_input = list(inputs)
        
        for i in range(max_new_tokens):
            try:
                # 调用C++后端推理下一个token
                # 注意：top_k, top_p, temperature参数在C++端处理
                next_token = self._native.infer_next_token(current_input)
                
                # 检查结束符
                if next_token == self._native.meta.eos_token_id:
                    print(f"  [Stop] EOS token at step {i}")
                    break
                
                generated.append(next_token)
                
                # 更新输入（C++端应该使用KV-Cache）
                # 这里我们传入完整的历史，让C++端决定如何处理
                current_input.append(next_token)
                
                # 显示进度
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{max_new_tokens} tokens")
                    
            except Exception as e:
                print(f"  Error at step {i}: {e}")
                break
        
        # 返回所有token
        return list(inputs) + generated
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> str:
        """
        生成文本的简化接口
        
        注意：需要外部tokenizer，这里不包含
        """
        # 这里需要外部tokenizer将文本转换为token IDs
        # 为了演示，假设inputs已经是token IDs
        raise NotImplementedError(
            "This method requires an external tokenizer. "
            "Use generate() with token IDs directly."
        )
    
    def reset(self):
        """重置C++模型状态"""
        self._native.reset()
    
    def __del__(self):
        """清理C++资源"""
        if hasattr(self, '_native'):
            self._native.destroy()


# 简单的使用示例
if __name__ == "__main__":
    # 假设有tokenizer将文本转换为token IDs
    def simple_tokenizer(text: str) -> List[int]:
        """简单的字符级tokenizer（仅用于演示）"""
        return [ord(c) for c in text]
    
    # 加载模型
    model = Qwen2("./models/DeepSeek-R1-Distill-Qwen-1.5B")
    
    # 生成
    prompt = "Hello"
    input_ids = simple_tokenizer(prompt)
    output_ids = model.generate(input_ids, max_new_tokens=20)
    
    # 转换回文本
    output_text = ''.join(chr(tid) for tid in output_ids if tid < 65536)
    print(f"Input: {prompt}")
    print(f"Output: {output_text}")