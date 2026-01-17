# Manifold Projection Update - 实现说明

## 更新日期
2026-01-16

## 更新概述

基于论文 "Mitigating Overthinking in Large Reasoning Models via Manifold Steering" 的理论，对 `no_preprocess_with_bias` 方法进行了升级，添加了 **PCA 流形投影** 功能，以消除高维空间中的干扰噪声。

## 理论基础

### 问题：高维空间的干扰噪声

在原始的 BatchNorm 方法中，我们直接在高维空间（d 维，通常 d = num_heads × head_dim = 28 × 128 = 3584）中计算偏置向量：

```python
bias = mean_bge - mean_no_preprocess * scale
```

**问题**：根据论文 Theorem 4.1，当维度 d 远大于样本数 N 时，高维空间中的噪声会累积：

```
E[||r_noise||²] = tr((I - P_M)Σ_noise)
```

其中 `(I - P_M)` 是正交补空间 M⊥ 的投影算子。这意味着大量噪声来自于与真实信号正交的维度。

### 解决方案：流形投影

**核心洞察**（Low-dimensional Manifold Hypothesis）：
- 神经网络的激活值实际上存在于一个低维流形 M ⊂ R^d
- 有效维度 d_eff << d（论文实验显示 k=10 即可捕获 >70% 方差）

**解决方法**：
1. 使用 PCA 识别主流形 M（top-k 主成分）
2. 计算投影矩阵：`P_M = U_eff @ U_eff^T`
3. 将偏置向量投影到流形：`bias_projected = P_M @ bias`

**效果**：消除正交补空间 M⊥ 中的噪声，只保留主流形上的信号。

## 实现细节

### 1. 新增方法

在 `KVDistributionAnalyzer` 类中新增两个方法：

#### `compute_manifold_projection()`
```python
def compute_manifold_projection(self, activations: torch.Tensor,
                                variance_threshold: float = 0.7) -> torch.Tensor:
    """
    使用 PCA 计算流形投影矩阵

    输入：activations [n_samples, n_features] - 所有样本的激活值
    输出：P_M [n_features, n_features] - 投影矩阵
    """
    # 1. 执行 PCA
    pca = PCA()
    pca.fit(activations_np)

    # 2. 找到累积方差 >= threshold 的最小 k
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumsum_variance, variance_threshold) + 1

    # 3. 获取 top-k 主成分
    U_eff = torch.tensor(pca.components_[:n_components].T)  # [n_features, k]

    # 4. 计算投影矩阵
    P_M = U_eff @ U_eff.T  # [n_features, n_features]

    return P_M
```

#### `project_to_manifold()`
```python
def project_to_manifold(self, vector: torch.Tensor,
                       projection_matrix: torch.Tensor) -> torch.Tensor:
    """
    将向量投影到流形

    输入：vector [num_heads, head_dim]
    输出：projected_vector [num_heads, head_dim]
    """
    # Flatten -> Project -> Reshape
    vector_flat = vector.reshape(-1)
    projected_flat = projection_matrix @ vector_flat
    projected = projected_flat.reshape(vector.shape)
    return projected
```

### 2. 修改的方法

#### `compute_distribution_stats()` 主要变化

**新增参数**：
- `use_manifold_projection: bool = True` - 是否启用流形投影
- `pca_variance_threshold: float = 0.7` - PCA 方差阈值

**数据收集阶段（新增）**：
```python
# 收集原始激活值用于 PCA
if use_manifold_projection:
    # Flatten: [num_heads, seq_len, head_dim] -> [seq_len, num_heads * head_dim]
    bge_key_flat = bge_key_layer.transpose(0, 1).reshape(seq_len, -1)
    no_prep_key_flat = no_prep_key_layer.transpose(0, 1).reshape(seq_len, -1)

    # 合并两个方法的数据（找共享流形）
    key_raw_activations[layer_name].append(
        torch.cat([bge_key_flat, no_prep_key_flat], dim=0)
    )
```

**统计聚合阶段（修改）**：
```python
# 原始计算（不变）
key_scale = bge_key_std_agg / (no_prep_key_std_agg + 1e-8)
key_bias = bge_key_mean_agg - no_prep_key_mean_agg * key_scale

# 应用流形投影（新增）
if use_manifold_projection:
    # 1. 合并所有样本的激活值
    key_activations_all = torch.cat(key_raw_activations[layer_name], dim=0)

    # 2. 计算投影矩阵
    key_projection_matrix = self.compute_manifold_projection(
        key_activations_all, variance_threshold=pca_variance_threshold
    )

    # 3. 投影 bias 和 scale
    key_bias = self.project_to_manifold(key_bias, key_projection_matrix)
    key_scale = self.project_to_manifold(key_scale, key_projection_matrix)
```

### 3. 新增命令行参数

```bash
--use_manifold_projection       # 启用流形投影（默认：True）
--no_manifold_projection        # 禁用流形投影
--pca_variance_threshold 0.7    # PCA 方差阈值（默认：0.7）
```

### 4. 输出变化

统计文件的 metadata 中新增两个字段：
```python
'metadata': {
    ...
    'use_manifold_projection': bool,
    'pca_variance_threshold': float
}
```

## 使用方法

### 启用流形投影（推荐，默认）

```bash
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --model_name Qwen2.5-7B-Instruct \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --output_path ./kv_stats/musique_manifold_proj.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7
```

### 禁用流形投影（使用原始 BatchNorm）

```bash
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --model_name Qwen2.5-7B-Instruct \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --output_path ./kv_stats/musique_no_proj.pt \
    --no_manifold_projection
```

## 预期效果

### 维度降低

原始维度：`d = num_heads × head_dim = 28 × 128 = 3584`

PCA 后（k 个主成分，70% 方差）：
- 预期 k ≈ 10-50（取决于数据）
- 维度降低 98%+
- 消除 >95% 的噪声维度

### 输出示例

```
Layer 0 - Key cache:
  PCA: 15 components capture 72.34% variance
  Dimension reduction: 3584 -> 15

Layer 0 - Value cache:
  PCA: 12 components capture 71.89% variance
  Dimension reduction: 3584 -> 12
```

### 性能对比预期

| 方法 | 维度 | 噪声 | 准确率预期 |
|------|------|------|------------|
| No Preprocess (baseline) | - | - | 基准 |
| BatchNorm (原始) | 3584 | 高 | +5% |
| **BatchNorm + Manifold** | 10-50 | 低 | **+8-10%** |
| BGE 召回 (upper bound) | - | - | +15% |

## 理论支持

### Theorem 4.1（论文）

对于随机投影 r ∈ R^d，如果 r 不在流形 M 上，则：

```
E[||r_other||²] = tr((I - P_M)Σ_noise)
```

当 d >> N 时，噪声项会很大。通过投影到 M：

```
r_M = P_M @ r
E[||r_M - r_true||²] ≈ 0  (如果 r_true ∈ M)
```

### Corollary 4.2（论文）

如果真实信号 r_true 存在于 k 维流形 M，则投影后的估计误差仅取决于 M 的有效维度 k，而不是原始维度 d。

## 向后兼容性

- 默认启用流形投影（推荐）
- 可通过 `--no_manifold_projection` 禁用，回退到原始 BatchNorm 方法
- online 阶段（test_fusionrag_reflect.py）无需修改，自动使用统计文件中的投影后向量

## 实验建议

### 对比实验

1. **No Preprocess (baseline)**
   ```bash
   RECALL_METHOD="bge"
   PREPROCESS="false"
   ```

2. **BatchNorm without Projection**
   ```bash
   # 计算统计（无投影）
   python compute_kv_distribution_stats.py ... --no_manifold_projection

   # 运行测试
   RECALL_METHOD="no_preprocess_with_bias"
   KV_STATS_PATH="./kv_stats/no_proj.pt"
   ```

3. **BatchNorm with Manifold Projection（推荐）**
   ```bash
   # 计算统计（有投影）
   python compute_kv_distribution_stats.py ... --use_manifold_projection

   # 运行测试
   RECALL_METHOD="no_preprocess_with_bias"
   KV_STATS_PATH="./kv_stats/manifold_proj.pt"
   ```

4. **BGE 召回 (upper bound)**
   ```bash
   RECALL_METHOD="bge"
   PREPROCESS="true"
   ```

### 评估指标

- **准确率**：答案正确率
- **速度**：每个问题的处理时间
- **内存**：峰值内存使用
- **与 BGE 的差距**：相对于 upper bound 的性能差距

## 文件变更总结

### 修改的文件
1. `compute_kv_distribution_stats.py`
   - 新增 `compute_manifold_projection()` 方法
   - 新增 `project_to_manifold()` 方法
   - 修改 `compute_distribution_stats()` 添加 PCA 流程
   - 新增命令行参数 `--use_manifold_projection`, `--pca_variance_threshold`

2. `NO_PREPROCESS_WITH_BIAS_README.md`
   - 更新"方法原理"章节，添加流形投影说明
   - 更新使用示例，添加新参数
   - 新增"优势"第4点：流形投影消除噪声

### 新增文件
3. `MANIFOLD_PROJECTION_UPDATE.md`（本文档）
   - 详细说明实现细节和使用方法

### 无需修改的文件
- `test_fusionrag_reflect.py` - online 阶段无需修改
- `run_fusionrag_sweep.sh` - 无需修改
- `run_no_preprocess_with_bias_example.sh` - 无需修改

## 依赖项

新增依赖：`sklearn` (已在原代码中添加)

```python
from sklearn.decomposition import PCA
```

确保环境中安装了 scikit-learn：
```bash
pip install scikit-learn
```

## 参考文献

- "Mitigating Overthinking in Large Reasoning Models via Manifold Steering"
- Theorem 4.1: Interference noise in orthogonal complement
- Section 4: Low-dimensional Manifold Hypothesis
- Experimental results: k=10 captures >70% variance

## 更新日志

- **2026-01-16**: 添加 PCA 流形投影支持
  - 实现 `compute_manifold_projection()` 方法
  - 实现 `project_to_manifold()` 方法
  - 修改统计计算流程，应用投影到 bias 和 scale
  - 添加命令行参数支持
  - 更新文档
