# Per-Head Statistics 功能指南

## 概述

Per-head statistics 功能允许对每一层的每个attention head单独计算steering vectors，而不是对所有heads一起统计。这提供了更精细的控制粒度。

## 两种统计模式对比

### 模式 1: Per-Layer (默认，`PER_HEAD_STATS="false"`)

**统计粒度**: 每层所有heads一起统计

**工作方式**:
1. 对每层的所有heads一起计算平均激活: `[num_heads, head_dim]`
2. 计算该层的steering vector: `[num_heads, head_dim]`
3. 对整层的激活做PCA投影: `[seq_len, num_heads * head_dim]`
4. 投影steering vector到流形上

**优点**:
- ✅ 计算速度快（每层1次PCA）
- ✅ 捕获heads之间的协同关系
- ✅ 更稳定（数据量大）

**缺点**:
- ❌ 粒度较粗，无法针对单个head
- ❌ 假设所有heads有相似的统计特性

**适用场景**:
- 大多数情况下的默认选择
- 当认为同一层的heads功能相似时
- 需要快速计算时

### 模式 2: Per-Head (`PER_HEAD_STATS="true"`)

**统计粒度**: 每层的每个head单独统计

**工作方式**:
1. 对每个head单独计算平均激活: `[head_dim]`
2. 为每个head单独计算steering vector: `[head_dim]`
3. 对每个head的激活单独做PCA投影: `[seq_len, head_dim]`
4. 每个head的steering vector单独投影

**优点**:
- ✅ 最精细的控制粒度
- ✅ 每个head独立优化
- ✅ 适应不同heads的不同特性
- ✅ 可以发现head-level的差异

**缺点**:
- ❌ 计算量大（每层28个heads × 28层 = 784次PCA）
- ❌ 数据量较小时可能不稳定
- ❌ 增加存储空间

**适用场景**:
- 需要精细控制每个head时
- 研究不同heads的作用差异
- 已知不同heads有明显不同功能时

## 配置方法

### 在 `run_fusionrag_steering_muliti.sh` 中

```bash
# 禁用 per-head 统计（默认）
PER_HEAD_STATS="false"

# 启用 per-head 统计
PER_HEAD_STATS="true"
```

### 直接调用 `compute_kv_distribution_stats.py`

```bash
# 禁用 per-head 统计（默认）
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --model_name Qwen2.5-7B-Instruct \
    --use_all_samples \
    --output_path ./kv_stats/steering.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7

# 启用 per-head 统计
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --model_name Qwen2.5-7B-Instruct \
    --use_all_samples \
    --output_path ./kv_stats/steering_perhead.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7 \
    --per_head_stats
```

## 输出格式差异

### Per-Layer 格式 (`PER_HEAD_STATS="false"`)

```python
{
    'key_steering': {
        'layer_0': {
            'steering_vector': tensor,  # [num_heads, head_dim]
            'bge_mean': tensor,         # [num_heads, head_dim]
            'no_prep_mean': tensor      # [num_heads, head_dim]
        },
        ...
    },
    'metadata': {
        'per_head_stats': False,
        ...
    }
}
```

### Per-Head 格式 (`PER_HEAD_STATS="true"`)

```python
{
    'key_steering': {
        'layer_0': {
            'steering_vector': tensor,  # [num_heads, head_dim] (整层的)
            'bge_mean': tensor,
            'no_prep_mean': tensor,
            'per_head': {                # 新增: 每个head的统计
                'head_0': {
                    'steering_vector': tensor,  # [head_dim]
                    'bge_mean': tensor,
                    'no_prep_mean': tensor
                },
                'head_1': {...},
                ...
                'head_27': {...}
            }
        },
        ...
    },
    'metadata': {
        'per_head_stats': True,
        ...
    }
}
```

## 文件命名约定

启用per-head统计后，文件名会自动添加`_perhead`后缀：

```bash
# Per-layer（默认）
2wikimqa_steering_manifold.pt
2wikimqa_to_musique_steering_manifold.pt

# Per-head（启用）
2wikimqa_steering_manifold_perhead.pt
2wikimqa_to_musique_steering_manifold_perhead.pt
```

## 计算成本对比

假设：
- 28层
- 每层28个heads
- 每个head维度128

### Per-Layer 模式

```
总PCA次数: 28层 × 2(key+value) = 56次
每次PCA维度: num_heads * head_dim = 28 × 128 = 3584维
总计算量: 56次 × 3584维
```

### Per-Head 模式

```
总PCA次数: 28层 × 28heads × 2(key+value) = 1568次
每次PCA维度: head_dim = 128维
总计算量: 1568次 × 128维
```

**时间对比**: Per-head模式大约需要 **5-10倍** 的计算时间

## PCA 效果对比

### Per-Layer PCA

**输入**: `[seq_len, num_heads * head_dim]` = `[seq_len, 3584]`

**典型输出**:
```
PCA: 10 components capture 72% variance
Dimension reduction: 3584 -> 10
```

**有效维度**: ~10 (远小于3584)

### Per-Head PCA

**输入**: `[seq_len, head_dim]` = `[seq_len, 128]`

**典型输出**:
```
PCA: 5 components capture 71% variance
Dimension reduction: 128 -> 5
```

**有效维度**: ~5 (远小于128)

## 使用建议

### 1. 默认使用 Per-Layer

对于大多数应用，per-layer模式已经足够：

```bash
PER_HEAD_STATS="false"
USE_MANIFOLD_PROJECTION="true"
PCA_VARIANCE_THRESHOLD="0.7"
```

### 2. 研究场景使用 Per-Head

如果你需要：
- 分析不同heads的作用
- 发现head-level的模式
- 最精细的控制

```bash
PER_HEAD_STATS="true"
USE_MANIFOLD_PROJECTION="true"
PCA_VARIANCE_THRESHOLD="0.7"
```

### 3. 快速验证时禁用 Manifold Projection

如果只是想看per-head的差异（不需要投影）：

```bash
PER_HEAD_STATS="true"
USE_MANIFOLD_PROJECTION="false"
```

这样会更快，但steering vector未经过降噪。

## 实验建议

### 实验1: 对比per-layer vs per-head

```bash
# 1. 计算per-layer steering
PER_HEAD_STATS="false"
bash run_fusionrag_steering_muliti.sh

# 2. 计算per-head steering
PER_HEAD_STATS="true"
bash run_fusionrag_steering_muliti.sh

# 3. 比较结果
# - 性能差异
# - 计算时间差异
# - 存储空间差异
```

### 实验2: 分析head差异

启用per-head后，可以分析：

```python
import torch

stats = torch.load('steering_manifold_perhead.pt')

# 查看layer 0的所有heads的steering vector范数
layer_0_key = stats['key_steering']['layer_0']['per_head']

head_norms = []
for head_idx in range(28):
    steering = layer_0_key[f'head_{head_idx}']['steering_vector']
    norm = torch.norm(steering).item()
    head_norms.append(norm)

print("Head steering vector norms:")
for i, norm in enumerate(head_norms):
    print(f"Head {i}: {norm:.4f}")

# 分析哪些heads的steering vector最大（最需要纠正）
import numpy as np
top_heads = np.argsort(head_norms)[-5:]
print(f"\nTop 5 heads with largest steering: {top_heads}")
```

### 实验3: 选择性head steering

基于per-head分析结果，可以只对重要的heads应用steering（需要修改应用代码）。

## 理论依据

### Per-Layer Manifold

假设: 同一层的所有heads存在于一个共享的低维流形M上

```
r_layer = mean(KV_bge) - mean(KV_no_prep)  # [num_heads * head_dim]
r_M = P_M @ r_layer  # 投影到共享流形
```

### Per-Head Manifold

假设: 每个head有自己独立的低维流形M_h

```
for each head h:
    r_h = mean(KV_bge_h) - mean(KV_no_prep_h)  # [head_dim]
    r_M_h = P_M_h @ r_h  # 投影到head专属流形
```

**关键区别**:
- Per-layer假设heads共享流形结构
- Per-head允许每个head有独立的流形

## 何时使用哪种模式？

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 标准实验 | Per-Layer | 平衡效果和速度 |
| 快速验证 | Per-Layer | 计算快 |
| 精细调优 | Per-Head | 最大控制 |
| 研究分析 | Per-Head | 发现head差异 |
| 数据量小 | Per-Layer | 更稳定 |
| 数据量大 | 两者皆可 | 都有足够数据 |
| 资源受限 | Per-Layer | 节省计算 |
| 追求极致 | Per-Head | 最优性能 |

## 故障排查

### 问题1: Per-head模式PCA失败

**可能原因**: 某些head的数据量不足

**解决方案**: 增加样本数量或降低PCA阈值

```bash
PCA_VARIANCE_THRESHOLD="0.6"  # 从0.7降低到0.6
```

### 问题2: Per-head计算太慢

**解决方案**:
1. 使用更少的样本进行快速验证
2. 或者回退到per-layer模式

### 问题3: 内存不足

**原因**: Per-head模式存储28倍的统计数据

**解决方案**:
1. 增加系统内存
2. 或只保存steering_vector，删除bge_mean和no_prep_mean

## 总结

**Per-Head Statistics** 是一个强大的功能，提供了最精细的控制粒度：

✅ **优势**:
- 每个head独立优化
- 发现head-level差异
- 最大化控制能力

⚠️ **代价**:
- 计算时间增加5-10倍
- 存储空间增加
- 数据需求更高

**推荐使用策略**:
1. 默认使用 per-layer 模式
2. 需要深入分析时使用 per-head 模式
3. 根据实验结果决定是否值得使用 per-head

通过这个功能，你可以更深入地理解和控制transformer中每个attention head的行为。
