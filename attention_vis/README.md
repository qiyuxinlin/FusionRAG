# 注意力可视化工具

这是一个通用的语言模型注意力可视化工具，支持给定任意句子和模型，可视化其注意力分布。

## 功能特性

- ✅ 支持任意Hugging Face模型或本地模型
- ✅ 可视化平均注意力（所有层和头的平均）
- ✅ 可视化特定层的注意力
- ✅ 可视化特定层特定头的注意力
- ✅ 一次性可视化所有层
- ✅ 一次性可视化特定层的所有头
- ✅ 支持命令行接口和Python API
- ✅ 自动处理自回归模型的未来token遮蔽
- ✅ 支持GPU加速

## 安装依赖

```bash
pip install torch transformers matplotlib seaborn numpy
```

## 快速开始

### 方法1: 使用命令行接口

```bash
# 可视化平均注意力
python run_attention_vis.py --model /mnt/data/models/Qwen2-0.5B-Instruct --text "This should only be set if you understand what it means" --mode avg

python run_attention_vis.py --model gpt2 --text "This should only be set if you understand what it means" --mode avg

# 可视化特定层（第5层）
python run_attention_vis.py --model gpt2 --text "Hello world" --mode layer --layer 5

# 可视化特定头（第5层第3个头）
python run_attention_vis.py --model gpt2 --text "Hello world" --mode head --layer 5 --head 3

# 可视化所有层
python run_attention_vis.py --model gpt2 --text "Hello world" --mode all-layers

# 可视化特定层的所有头
python run_attention_vis.py --model gpt2 --text "Hello world" --mode all-heads --layer 5
```

### 方法2: 使用Python API

```python
from visualize_attention import AttentionVisualizer

# 创建可视化器
visualizer = AttentionVisualizer("gpt2")

# 输入文本
text = "The quick brown fox jumps over the lazy dog."

# 可视化平均注意力
visualizer.plot_average_attention(text, output_path="attention.png")

# 可视化特定层
visualizer.plot_layer_attention(text, layer_idx=5, output_path="layer5.png")

# 可视化特定头
visualizer.plot_head_attention(text, layer_idx=5, head_idx=3, output_path="head.png")

# 可视化所有层
visualizer.plot_all_layers(text, output_path="all_layers.png")

# 可视化特定层的所有头
visualizer.plot_all_heads(text, layer_idx=5, output_path="all_heads.png")
```

## 命令行参数详解

### 必需参数

- `--model`, `-m`: 模型名称或路径
  - 可以是Hugging Face模型ID（如`gpt2`, `meta-llama/Llama-2-7b-hf`）
  - 也可以是本地模型路径

- `--text`, `-t`: 要分析的文本

- `--mode`: 可视化模式
  - `avg`: 平均注意力（所有层和头）
  - `layer`: 特定层的平均注意力
  - `head`: 特定层特定头的注意力
  - `all-layers`: 所有层的注意力
  - `all-heads`: 特定层所有头的注意力
  - `info`: 仅显示模型信息

### 可选参数

- `--layer`, `-l`: 层索引（用于`layer`、`head`、`all-heads`模式）
- `--head`, `-h`: 头索引（用于`head`模式）
- `--output`, `-o`: 输出图片路径（默认自动生成）
- `--no-mask`: 不遮蔽未来tokens（用于编码器模型如BERT）
- `--device`: 运行设备（`auto`, `cuda`, `cpu`）
- `--dtype`: 模型数据类型（`float16`, `float32`, `bfloat16`）
- `--figsize`: 图片大小，格式为`width,height`（默认`12,10`）

## 使用示例

### 示例1: 查看模型信息

```bash
python run_attention_vis.py --model gpt2 --text "test" --mode info
```

输出:
```
模型信息:
  模型名称: gpt2
  层数: 12
  每层注意力头数: 12
  设备: cuda:0
```

### 示例2: 使用本地模型

```bash
python run_attention_vis.py \
  --model /path/to/your/local/model \
  --text "Your custom text here" \
  --mode avg \
  --output my_attention.png
```

### 示例3: 使用编码器模型（BERT等）

对于BERT等编码器模型，使用`--no-mask`参数来显示完整的注意力矩阵（不遮蔽未来tokens）:

```bash
python run_attention_vis.py \
  --model bert-base-uncased \
  --text "Hello world" \
  --mode avg \
  --no-mask
```

### 示例4: 分析长文本

```bash
python run_attention_vis.py \
  --model gpt2 \
  --text "The transformer architecture has revolutionized NLP." \
  --mode avg \
  --figsize 16,14
```

### 示例5: 在CPU上运行

```bash
python run_attention_vis.py \
  --model gpt2 \
  --text "Hello world" \
  --mode avg \
  --device cpu
```

### 示例6: 使用完整精度

```bash
python run_attention_vis.py \
  --model gpt2 \
  --text "Hello world" \
  --mode avg \
  --dtype float32
```

## Python API详细说明

### 创建可视化器

```python
from visualize_attention import AttentionVisualizer

visualizer = AttentionVisualizer(
    model_name="gpt2",        # 模型名称或路径
    device="auto",            # 设备: "auto", "cuda", "cpu"
    torch_dtype=torch.float16 # 数据类型
)
```

### 获取模型信息

```python
info = visualizer.get_model_info()
print(f"层数: {info['num_layers']}")
print(f"注意力头数: {info['num_heads']}")
```

### 可视化方法

#### 1. 平均注意力

```python
visualizer.plot_average_attention(
    text="Your text here",
    output_path="attention.png",
    mask_future=True,  # 是否遮蔽未来tokens
    figsize=(12, 10)   # 图片大小
)
```

#### 2. 特定层注意力

```python
visualizer.plot_layer_attention(
    text="Your text here",
    layer_idx=5,       # 层索引
    output_path="layer5.png",
    mask_future=True,
    figsize=(12, 10)
)
```

#### 3. 特定头注意力

```python
visualizer.plot_head_attention(
    text="Your text here",
    layer_idx=5,       # 层索引
    head_idx=3,        # 头索引
    output_path="head.png",
    mask_future=True,
    figsize=(12, 10)
)
```

#### 4. 所有层注意力

```python
visualizer.plot_all_layers(
    text="Your text here",
    output_path="all_layers.png",
    mask_future=True,
    max_cols=4  # 每行最多显示的子图数量
)
```

#### 5. 所有头注意力

```python
visualizer.plot_all_heads(
    text="Your text here",
    layer_idx=5,
    output_path="all_heads.png",
    mask_future=True,
    max_cols=4
)
```

## 运行完整示例

我们提供了一个包含多个示例的脚本:

```bash
python example.py
```

这将运行所有示例并生成多个可视化图片。

## 文件说明

- `visualize_attention.py`: 核心可视化工具类
- `run_attention_vis.py`: 命令行接口
- `example.py`: 使用示例
- `attention_vis.py`: RE2注意力可视化（特定用途）
- `README.md`: 本文档

## 注意事项

1. **内存使用**: 大模型可能需要较多GPU内存。如果遇到OOM错误，可以:
   - 使用`--device cpu`在CPU上运行
   - 使用`--dtype float16`减少内存占用
   - 缩短输入文本长度

2. **自回归 vs 编码器模型**:
   - 自回归模型（GPT系列）: 使用`mask_future=True`（默认）
   - 编码器模型（BERT系列）: 使用`mask_future=False`或`--no-mask`

3. **可视化大小**: 对于较长的文本，建议增加`figsize`参数以获得更清晰的可视化效果

4. **模型加载**: 首次运行会从Hugging Face下载模型，请确保网络连接正常

## 常见问题

**Q: 如何可视化本地模型?**

A: 将`--model`参数设置为本地模型路径:
```bash
python run_attention_vis.py --model /path/to/model --text "test" --mode avg
```

**Q: 如何减少内存使用?**

A: 使用以下选项:
```bash
python run_attention_vis.py --model gpt2 --text "test" --mode avg --dtype float16 --device cuda
```

**Q: 如何可视化BERT模型?**

A: 对于编码器模型，添加`--no-mask`参数:
```bash
python run_attention_vis.py --model bert-base-uncased --text "test" --mode avg --no-mask
```

**Q: 如何选择特定的GPU?**

A: 使用环境变量:
```bash
CUDA_VISIBLE_DEVICES=0 python run_attention_vis.py --model gpt2 --text "test" --mode avg
```

## 高级用法

### 批量处理多个文本

```python
from visualize_attention import AttentionVisualizer

visualizer = AttentionVisualizer("gpt2")

texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]

for i, text in enumerate(texts):
    visualizer.plot_average_attention(
        text,
        output_path=f"attention_{i}.png"
    )
```

### 比较不同层的注意力

```python
visualizer = AttentionVisualizer("gpt2")
text = "Compare different layers."

# 可视化多个层
for layer_idx in [0, 5, 11]:  # 第一层、中间层、最后一层
    visualizer.plot_layer_attention(
        text,
        layer_idx=layer_idx,
        output_path=f"layer_{layer_idx}.png"
    )
```

### 分析注意力模式

```python
from visualize_attention import AttentionVisualizer

visualizer = AttentionVisualizer("gpt2")
text = "The cat sat on the mat."

# 获取注意力权重
tokens, attentions = visualizer.get_attention_weights(text)

# 分析注意力权重
import torch
import numpy as np

# 平均注意力
avg_attention = torch.stack(attentions).mean(dim=0).mean(dim=1)[0]

# 找出每个token最关注的token
for i, token in enumerate(tokens):
    max_attention_idx = avg_attention[i].argmax().item()
    max_attention_val = avg_attention[i, max_attention_idx].item()
    print(f"{token} -> {tokens[max_attention_idx]} (权重: {max_attention_val:.3f})")
```

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT License
