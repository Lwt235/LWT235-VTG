# 修复使用 LoRA 适配器时 generation_config.json 加载错误

## 问题描述

当设置 `use_temporal_tokens=True` 并使用 LoRA 适配器进行推理时，会出现以下错误：

```
OSError: outputs/sft/checkpoint-final does not appear to have a file named generation_config.json
```

## 问题原因

代码尝试从 LoRA 适配器目录加载 `generation_config.json`，但 LoRA 适配器通常不包含这个文件。这个文件应该从基础模型加载。

### LoRA 适配器目录结构

```
./outputs/sft/checkpoint-final/     # LoRA 适配器
├── adapter_config.json             # 适配器配置（包含基础模型路径）
├── adapter_model.safetensors       # LoRA 权重
├── tokenizer.json                  # 分词器文件
└── ...                             # ✗ 没有 generation_config.json
```

### 基础模型目录结构

```
./ckpts/Qwen3-VL-2B-Instruct/      # 基础模型
├── config.json
├── generation_config.json          # ✓ 生成配置文件在这里
├── model.safetensors
└── ...
```

## 解决方案

修改了 `inference/video_infer.py` 中的 `_load_model()` 方法，让它能智能选择正确的路径：

```python
# 修改前：
self.generation_config = GenerationConfig.from_pretrained(
    model_path_str  # 总是使用模型路径，对 LoRA 适配器来说是错误的
)

# 修改后：
# 如果使用 LoRA 适配器，从基础模型加载；否则从模型路径加载
config_path = base_model_path if is_lora_adapter and base_model_path else model_path_str
self.generation_config = GenerationConfig.from_pretrained(
    config_path  # 对 LoRA 和非 LoRA 模型都正确
)
```

## 修改内容

1. **inference/video_infer.py**（核心修复）：
   - 第 113 行：初始化 `base_model_path = None`
   - 第 207-212 行：根据是否是 LoRA 适配器选择正确的路径

2. **tests/test_inference.py**（测试覆盖）：
   - 添加了测试用例验证修复是否正确工作

3. **文档**：
   - 详细的修复说明和使用示例

## 使用示例

现在这段代码可以正常工作了：

```python
from inference import VideoTemporalInference

# 使用 LoRA 适配器和时间标记初始化
engine = VideoTemporalInference(
    model_path="./outputs/sft/checkpoint-final",  # LoRA 适配器路径
    use_temporal_tokens=True,                      # 启用时间标记
)

# 运行推理
result = engine.predict(
    video_path="./data/videos/example.mp4",
    query="the paper bill has an image of a woman holding up a paper",
    video_start=404.2,
    video_end=429.1,
    duration=24.9,
)

print(f"片段: {result['start_time']:.2f}秒 - {result['end_time']:.2f}秒")
```

## 影响

- ✅ 修复了报告的错误
- ✅ 没有破坏性变更（向后兼容）
- ✅ 最小化代码修改（仅 4 行）
- ✅ 通过安全检查（CodeQL 0 个警报）
- ✅ 添加了测试和文档

## 工作原理

当检测到是 LoRA 适配器时：
1. 从 `adapter_config.json` 读取 `base_model_name_or_path` 字段
2. 使用这个基础模型路径加载 `generation_config.json`
3. 从适配器目录加载 LoRA 权重和分词器

当是普通模型时：
1. 直接从模型路径加载所有内容，包括 `generation_config.json`

## 相关文件

- `inference/video_infer.py` - 核心修复
- `tests/test_inference.py` - 测试用例
- `docs/fix_generation_config_lora.md` - 英文详细文档
- `SUMMARY.md` - 修复总结
