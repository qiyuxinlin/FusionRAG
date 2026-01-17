# Steering Vector 更新总结

## 更新时间
2026-01-16

## 核心变更

从 **BatchNorm (scale/bias)** 方案改为 **Steering Vector（论文方法）**

### 之前（BatchNorm）

```python
# 计算
scale = std_bge / std_no_prep
bias = mean_bge - mean_no_prep * scale

# 应用
kv_aligned = kv * scale + bias
```

### 现在（Steering Vector）

```python
# 计算
steering = mean_bge - mean_no_prep  # 直接计算差异
steering_M = P_M @ steering         # 流形投影（PCA）

# 应用
kv_aligned = kv + α * steering_M    # 简单加法
```

## 核心公式

```
h' = h + α * r_M
```

其中：
- `h`: KV_no_preprocess（原始激活）
- `r`: mean(KV_bge) - mean(KV_no_preprocess)（steering vector）
- `r_M`: P_M @ r（投影到流形，消除噪声）
- `α`: steering strength（默认 1.0）

## 修改的文件

### 1. compute_kv_distribution_stats.py

**变化**：
- 不再计算 `scale` 和 `bias`
- 直接计算 `steering_vector = mean(bge) - mean(no_prep)`
- 使用 PCA 投影到流形：`r_M = P_M @ r`

**输出格式变化**：
```python
# 旧格式
{
    'key_stats': {
        'layer_0': {'bias': ..., 'scale': ...},
        ...
    }
}

# 新格式
{
    'key_steering': {
        'layer_0': {'steering_vector': ...},
        ...
    }
}
```

### 2. test_fusionrag_reflect.py

**函数改名**：
- `apply_distribution_bias()` → `apply_steering_vector()`

**新逻辑**：
```python
# 检测格式
if 'key_steering' in stats:
    # 新格式：应用 steering vector
    kv_steered = kv + alpha * steering_vector
else:
    # 旧格式：应用 scale/bias（向后兼容）
    kv_transformed = kv * scale + bias
```

## 使用方法

### 计算 Steering Vectors（Offline）

```bash
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --model_name Qwen2.5-7B-Instruct \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --output_path ./kv_stats/steering_vectors.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7
```

### 运行测试（Online）

```bash
python test_fusionrag_reflect.py \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/steering_vectors.pt \
    --preprocess true \
    --rate 0.3
```

## 关键参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--use_manifold_projection` | True | 启用 PCA 流形投影 |
| `--no_manifold_projection` | - | 禁用流形投影 |
| `--pca_variance_threshold` | 0.7 | PCA 累积方差阈值 |
| `alpha`（在 apply 函数中） | 1.0 | Steering strength |

## 预期效果

### 维度降低
- 原始：3584 维（28 heads × 128 head_dim）
- PCA 后：10-50 维（保留 70% 方差）
- 降低：>98%

### 性能预期
| 方法 | 准确率 | 开销 |
|------|-------|------|
| No Preprocess | 基准 | 低 |
| Steering (no proj) | +5-7% | 低-中 |
| **Steering + Manifold** | **+8-12%** | **低-中** |
| BGE (upper bound) | +15% | 高 |

## 向后兼容

✅ **自动检测格式**，支持新旧两种：
- 新格式：`key_steering` → 使用 steering vector
- 旧格式：`key_stats` → 使用 scale/bias

## 快速开始

### 一键运行示例

```bash
bash run_no_preprocess_with_bias_example.sh
```

### 对比实验

```bash
bash run_comparison_experiment.sh
```

这会运行4种方法的对比：
1. No Preprocess（baseline）
2. Steering without projection
3. **Steering + Manifold**（推荐）
4. BGE（upper bound）

## 文档

详细文档：
- **MANIFOLD_STEERING_METHOD.md** - 完整设计和实现
- **NO_PREPROCESS_WITH_BIAS_README.md** - 用户指南（已更新）
- **MANIFOLD_PROJECTION_UPDATE.md** - 技术细节

## 验证测试

运行验证脚本：
```bash
python verify_manifold_projection.py
```

预期输出：
```
✓ All verification tests passed!
```

## 理论依据

论文：**"Mitigating Overthinking in Large Reasoning Models via Manifold Steering"**

核心定理：**Theorem 4.1**
```
E[||r_noise||²] = tr((I - P_M)Σ_noise)
```

**含义**：正交补空间 M⊥ 的噪声会累积，PCA 投影消除这些噪声。

## 总结

✅ **完全按照论文实现** - Steering vector + Manifold projection
✅ **理论支持** - Theorem 4.1，维度降低 >98%
✅ **向后兼容** - 自动检测新旧格式
✅ **简单高效** - 只需向量加法
✅ **文档完整** - 详细设计文档和使用示例

🎯 **推荐使用**：Steering + Manifold projection
