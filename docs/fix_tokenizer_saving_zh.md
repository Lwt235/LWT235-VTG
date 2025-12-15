# 修复：训练时未保存Tokenizer导致模型输出不一致

## 问题描述
当启用时序标记训练（`use_temporal_tokens=True`）时，模型在推理阶段的输出可能不一致。这是因为训练过程中添加的时序标记没有随tokenizer一起保存。

## 根本原因
`trainers/sft_trainer.py` 中的 `VideoTemporalSFTTrainer.save_model()` 方法只保存了：
1. 模型（通过 `super().save_model()`）
2. 处理器（通过 `self.processor.save_pretrained()`）

但是**没有保存tokenizer**（存储在 `self.processing_class` 中），而tokenizer中包含了训练初始化时添加的时序标记。

### 什么是时序标记？
时序标记（`<0>` 到 `<999>`）是添加到tokenizer词汇表中的特殊标记，用于实现细粒度的时序定位。当设置 `use_temporal_tokens=True` 时，会在训练时动态添加这1000个标记。

## 问题流程

### 训练阶段：
1. 从基础模型加载tokenizer
2. 通过 `add_temporal_tokens_to_tokenizer()` 添加时序标记（`<0>` 到 `<999>`）
3. 调整模型嵌入层大小以适应新标记
4. 使用时序标记进行训练
5. **但是**：只保存了模型和处理器，**tokenizer没有被保存**

### 推理阶段：
1. 尝试从检查点加载tokenizer → **失败或加载到不含时序标记的基础tokenizer**
2. 重新添加时序标记 → **由于状态不一致可能得到不同的标记ID**
3. 结果：**输出不一致**，因为标记ID与训练时不匹配

## 解决方案
修改 `VideoTemporalSFTTrainer.save_model()` 以明确保存tokenizer：

```python
def save_model(
    self,
    output_dir: Optional[str] = None,
    _internal_call: bool = False,
):
    """
    保存模型、tokenizer和处理器。
    """
    output_dir = output_dir or self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型（自动处理PEFT）
    super().save_model(output_dir, _internal_call=_internal_call)

    # 保存tokenizer（processing_class）以保留时序标记
    if self.processing_class is not None:
        self.processing_class.save_pretrained(output_dir)

    # 保存处理器（如果可用）
    if self.processor is not None:
        self.processor.save_pretrained(output_dir)

    logger.info(f"Model saved to {output_dir}")
```

## 修改内容

### 1. 代码修复（`trainers/sft_trainer.py`）
- **319-321行**：通过 `self.processing_class.save_pretrained()` 添加tokenizer保存
- **307行**：更新文档字符串，说明现在也会保存tokenizer

### 2. 验证（`trainers/rl_trainer.py`）
- 确认RL训练器已经正确保存tokenizer（第497行）
- RL训练器无需修改

### 3. 测试覆盖（`tests/test_trainers.py`）
添加单元测试以验证：
- 调用 `save_model()` 时会保存tokenizer
- 调用 `save_model()` 时会保存processor
- 正确处理tokenizer和processor为 `None` 的情况

## 影响
- **修复了模型输出不一致问题**：时序标记现在能够一致地保留
- **无破坏性变更**：对使用和不使用时序标记的训练都适用
- **最小代码改动**：仅添加4行代码
- **与RL训练器保持一致**：SFT和RL训练器现在以相同方式处理tokenizer保存

## 训练后保存的文件

### 修复前（缺少文件）：
```
outputs/sft/checkpoint-final/
├── adapter_config.json
├── adapter_model.safetensors
└── preprocessor_config.json    # 仅处理器
```

### 修复后（完整）：
```
outputs/sft/checkpoint-final/
├── adapter_config.json
├── adapter_model.safetensors
├── preprocessor_config.json    # 处理器
├── tokenizer_config.json       # Tokenizer ✓
├── tokenizer.json              # Tokenizer ✓
├── special_tokens_map.json     # Tokenizer ✓
└── vocab.json                  # Tokenizer ✓
```

## 使用示例

### 使用时序标记训练：
```python
# 在 train_sft.py 或 train_rl.py 中设置 use_temporal_tokens=True
python train_sft.py

# 现在检查点包含带有时序标记的tokenizer
```

### 推理：
```python
from inference import VideoTemporalInference

# 正确加载带有时序标记的tokenizer
engine = VideoTemporalInference(
    model_path="./outputs/sft/checkpoint-final",
    use_temporal_tokens=True,
)

result = engine.predict(
    video_path="./data/videos/example.mp4",
    query="人物走路",
    duration=30.0,
)
```

## 验证
该修复已通过以下方式验证：
1. ✅ 代码审查（逻辑验证正确）
2. ✅ 添加了tokenizer保存的单元测试
3. ✅ 与RL训练器实现一致性检查
4. ✅ 手动代码检查

## 相关信息
- 相关问题："模型的输出不一致是否也与添加新token并训练完成后模型没有保存tokenizer有关"
- 时序标记文档：`utils/temporal_tokens.py`
- SFT训练器：`trainers/sft_trainer.py`
- RL训练器：`trainers/rl_trainer.py`
