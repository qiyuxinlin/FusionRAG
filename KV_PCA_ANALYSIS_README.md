# KV Cache PCA 可视化分析方案

## 📋 方案概述

本方案通过 **PCA (主成分分析)** 可视化不同层的 Key 和 Value 向量特征分布，对比 **No Preprocess** 和 **BGE** 两种模式的 KV cache 差异。

### 核心目标

1. **理解 FusionRAG 的工作机制**: 通过可视化看 BGE 预处理如何改变 KV cache
2. **层级分析**: 观察不同层的特征分布差异
3. **量化对比**: 计算 No Preprocess 和 BGE 模式的 L2 距离
4. **降维可视化**: 将高维 KV 向量投影到 2D 空间，直观展示分布

---

## 🔬 分析维度

### 1. **模式对比维度**

| 模式 | 描述 | KV Cache 来源 |
|------|------|--------------|
| **No Preprocess** | 单文档 KV | `/kv_cache/{example_id}_{chunk_id}_*.pt` |
| **BGE** | 融合多文档 KV | `/preprocess_kv_cache_global_topk10_bge/{example_id}_{chunk_id}_*.pt` |

**关键差异**:
- No Preprocess: 只包含当前文档的 KV
- BGE: 包含 system + top10 相似文档 + 当前文档的融合 KV

### 2. **层级分析维度**

分析 **6 层**（均匀分布在 28 层中）:
- **Layer 0** (底层): 词嵌入层附近，特征较原始
- **Layer 5-6** (浅层): 局部特征
- **Layer 11-12** (中层): 中级语义特征
- **Layer 16-17** (中深层): 高级语义特征
- **Layer 22-23** (深层): 抽象特征
- **Layer 27** (顶层): 最终表示

### 3. **K vs V 对比**

- **Key (K)**: 用于 attention 计算相似度
- **Value (V)**: 用于加权聚合生成输出

观察 K 和 V 的分布差异，理解它们在 attention 机制中的不同角色。

---

## 📊 可视化输出

### 生成的图表

#### 1. **PCA 散点图** (每个样本一张)

**文件**: `pca_key_example{id}_chunk{id}.png`, `pca_value_example{id}_chunk{id}.png`

**内容**:
- 多层子图网格（例如 2×3 或 3×4）
- 每个子图显示一层的 PCA 投影
- **蓝色点**: No Preprocess 模式的 token
- **红色点**: BGE 模式的 token
- 标题显示 L2 距离

**示例布局**:
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Layer 0     │ Layer 5     │ Layer 11    │ Layer 16    │
│ L2: 0.45    │ L2: 0.62    │ L2: 0.88    │ L2: 1.23    │
│ 蓝=No Prep  │             │             │             │
│ 红=BGE      │             │             │             │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Layer 22    │ Layer 27    │             │             │
│ L2: 1.45    │ L2: 1.67    │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

**解读要点**:
- **重叠程度**: 两种模式的点云重叠越多 → 差异越小
- **分离程度**: 两种模式的点云分离 → 差异明显
- **分布形状**:
  - 聚集 → 特征集中
  - 分散 → 特征多样
- **L2 距离**: 量化两种模式的平均差异

#### 2. **L2 距离趋势图**

**文件**: `l2_distance_trends.png`

**内容**:
- 左图: Key cache 的 L2 距离随层数变化
- 右图: Value cache 的 L2 距离随层数变化
- 多条曲线代表不同样本

**示例**:
```
L2 Distance
    ↑
2.0 │           ╱───
    │         ╱
1.5 │       ╱
    │     ╱
1.0 │   ╱
    │ ╱
0.5 │╱
    └─────────────────→ Layer
    0   5  11  16  22  27
```

**预期趋势**:
- **递增**: 深层差异 > 浅层差异（常见）
- **平稳**: 各层差异相近
- **波动**: 某些层差异特别大

#### 3. **PCA 方差解释比例**

**文件**: `pca_variance_explained.png`

**内容**:
- 左图: Key 的前2个主成分解释的方差比例
- 右图: Value 的前2个主成分解释的方差比例

**解读**:
- **高方差比例** (>0.8): 2D 投影保留了大部分信息，可视化可信
- **低方差比例** (<0.5): 2D 投影损失较多信息，需谨慎解读

#### 4. **统计摘要 JSON**

**文件**: `summary_statistics.json`

**内容**:
```json
[
  {
    "example_id": 0,
    "chunk_id": 1,
    "layers": {
      "0": {
        "key_l2_dist": 0.4523,
        "val_l2_dist": 0.3891,
        "key_variance_explained": [0.6234, 0.1876],
        "val_variance_explained": [0.5891, 0.2103]
      },
      ...
    }
  }
]
```

---

## 🚀 使用方法

### 快速开始

```bash
# 1. 添加执行权限
chmod +x run_kv_pca_analysis.sh

# 2. 运行分析（默认分析样本 0, 1, 2）
bash run_kv_pca_analysis.sh
```

### 自定义配置

编辑 `run_kv_pca_analysis.sh`:

```bash
# 分析更多样本
SAMPLE_IDS="0 1 2 3 4 5"

# 分析 system prompt (chunk_id=0)
CHUNK_ID="0"

# 指定要分析的层
LAYERS="0 7 14 21 27"  # 取消注释并指定

# 增加采样 token 数（更精确但更慢）
MAX_TOKENS="1000"
```

### Python 直接调用

```bash
/home/shm/anaconda3/envs/fusionrag/bin/python visualize_kv_pca.py \
    --sample_ids 0 1 2 \
    --chunk_id 1 \
    --layers 0 7 14 21 27 \
    --max_tokens 500 \
    --output_dir ./my_analysis
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cache_dir` | `/mnt/data3/tmp/fusionrag` | KV cache 根目录 |
| `--dataset` | `musique` | 数据集名称 |
| `--model_name` | `Qwen2.5-7B-Instruct` | 模型名称 |
| `--sample_ids` | `[0]` | 要分析的样本ID列表 |
| `--chunk_id` | `1` | Chunk ID (1=文档, 0=system) |
| `--layers` | `None` | 指定层（默认自动选6层）|
| `--max_layers` | `28` | 模型总层数 |
| `--max_tokens` | `500` | 每层采样token数 |
| `--output_dir` | `./kv_pca_analysis` | 输出目录 |

---

## 🔍 分析示例与解读

### 示例 1: 观察层级特征演化

**观察目标**: 不同层的 KV 分布如何变化

**方法**:
```bash
# 分析所有层（均匀采样6层）
bash run_kv_pca_analysis.sh
```

**预期结果**:
- **浅层** (Layer 0-5):
  - 蓝色和红色点云大部分重叠
  - L2 距离较小 (< 0.5)
  - 说明浅层特征差异小

- **深层** (Layer 20-27):
  - 蓝色和红色点云分离明显
  - L2 距离较大 (> 1.0)
  - 说明深层特征差异大

**结论**: BGE 融合对深层影响更大，浅层保持相对稳定

---

### 示例 2: Key vs Value 对比

**观察目标**: Key 和 Value 的差异模式是否不同

**方法**: 对比同一层的 Key 和 Value 图

**预期结果**:
- **Key**: 可能差异更大（用于相似度计算，融合影响显著）
- **Value**: 可能差异较小（用于内容聚合，相对稳定）

**如果 Key L2 > Value L2**:
→ BGE 融合主要改变了 attention 的相似度计算，而不是聚合的内容

**如果 Value L2 > Key L2**:
→ BGE 融合主要改变了聚合的内容表示

---

### 示例 3: 多样本一致性

**观察目标**: 不同样本的差异模式是否一致

**方法**:
```bash
# 分析多个样本
SAMPLE_IDS="0 1 2 3 4 5 6 7 8 9"
```

**查看**: `l2_distance_trends.png`

**预期结果**:
- **一致趋势**: 所有样本的 L2 距离曲线形状相似
  → BGE 融合的影响模式稳定

- **不一致**: 不同样本曲线差异大
  → 融合效果因样本而异（可能与文档相似度有关）

---

## 📐 技术细节

### PCA 降维流程

1. **提取特征**:
   ```
   KV Cache: [num_layers, batch, seq_len, hidden_dim]
   选择 layer_i: [1, seq_len, 128]
   去除 batch: [seq_len, 128]
   采样 tokens: [500, 128]
   ```

2. **合并数据**:
   ```
   no_prep: [500, 128]
   bge:     [500, 128]
   combined: [1000, 128]
   ```

3. **PCA 投影**:
   ```
   PCA.fit_transform(combined)
   → [1000, 2]
   ```

4. **分离回原模式**:
   ```
   no_prep_pca: [500, 2]  (前500个点)
   bge_pca:     [500, 2]  (后500个点)
   ```

### L2 距离计算

```python
# 对每个 token，计算欧氏距离
l2_dist = ||no_prep_kv - bge_kv||_2

# 平均所有 token 的距离
mean_l2 = mean(l2_dist)
```

### 采样策略

**为什么采样**:
- 完整文档可能有数百到上千个 token
- PCA 计算复杂度: O(n²d) where n=样本数，d=维度
- 采样 500 个 token 平衡精度和效率

**采样方法**:
```python
# 随机采样（保证代表性）
indices = torch.randperm(seq_len)[:max_tokens]
sampled_kv = kv_cache[indices]
```

---

## 🎯 预期发现与假设验证

### 假设 1: 深层差异大于浅层

**验证方法**: 查看 `l2_distance_trends.png`

**如果成立**:
- 曲线呈递增趋势
- 说明 BGE 融合的影响在深层累积放大

**如果不成立**:
- 曲线平坦或波动
- 说明各层独立处理融合信息

---

### 假设 2: BGE 形成独特的聚类

**验证方法**: 查看 PCA 散点图

**如果成立**:
- 红色点形成独立的聚类区域
- 说明 BGE 融合创造了新的特征空间

**如果不成立**:
- 红色和蓝色点混合
- 说明 BGE 只是在原有空间中调整，未创造新特征

---

### 假设 3: Key 比 Value 受影响更大

**验证方法**: 对比同层的 Key 和 Value L2 距离

**如果成立**:
- key_l2_dist > val_l2_dist (大部分层)
- 说明融合主要改变 attention 权重分配

**如果不成立**:
- val_l2_dist >= key_l2_dist
- 说明融合也显著改变内容表示

---

## 🔧 故障排除

### 问题 1: FileNotFoundError

**错误**: `KV cache not found: /mnt/data3/tmp/.../0_1_key.pt`

**原因**: 指定的样本或 chunk 没有 KV cache

**解决**:
```bash
# 检查可用的样本
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache/ | grep "_1_key.pt" | head -10

# 使用存在的样本 ID
SAMPLE_IDS="0 1 2"
```

---

### 问题 2: 内存不足

**错误**: `CUDA out of memory` 或 `MemoryError`

**原因**: 加载多个大的 KV cache

**解决**:
```bash
# 减少采样 token 数
MAX_TOKENS="200"

# 一次只分析一个样本
SAMPLE_IDS="0"

# 或使用更少的层
LAYERS="0 14 27"
```

---

### 问题 3: PCA 方差解释率低

**现象**: `pca_variance_explained.png` 显示 < 40%

**原因**: KV 特征高度分散，2D 投影损失信息多

**解决**:
- 仍可参考 L2 距离趋势图
- 或尝试 3D PCA（需修改代码）
- 或使用 t-SNE（计算较慢，但更适合非线性结构）

---

## 📚 扩展分析建议

### 1. 增加对比模式

修改代码支持更多模式对比:
```python
# 对比 BGE vs REPEAT_SELF vs RANDOM
modes = ['no_preprocess', 'bge', 'repeat_self', 'random']
```

### 2. t-SNE 可视化

对于高度非线性的特征分布，t-SNE 可能更合适:
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30)
transformed = tsne.fit_transform(features)
```

### 3. 层间相似度热图

计算不同层之间 KV 分布的相似度:
```python
# Compute cosine similarity between layers
similarity_matrix = cosine_similarity(layer_features)
sns.heatmap(similarity_matrix)
```

### 4. Token 级别分析

不采样，分析特定 token（如问题中的关键词）的 KV 变化

---

## 📖 参考资料

- **PCA**: [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- **KV Cache**: [Transformer KV Cache 机制](https://huggingface.co/docs/transformers/kv_cache)
- **FusionRAG**: 论文和代码库

---

## 🎓 论文写作建议

### 可以用的图表

1. **Figure X: Layer-wise KV Distribution Evolution**
   - 使用 `pca_key/value_example*.png` 的某一层作为代表
   - 说明: "PCA visualization shows BGE preprocessing creates distinct feature clusters in deeper layers"

2. **Figure Y: L2 Distance Trends Across Layers**
   - 使用 `l2_distance_trends.png`
   - 说明: "The impact of BGE fusion accumulates in deeper layers, with L2 distance increasing from X to Y"

3. **Table Z: Statistical Summary**
   - 基于 `summary_statistics.json`
   - 列出关键层的 L2 距离和方差解释率

### 可能的结论

**如果深层差异显著**:
> "Our analysis reveals that FusionRAG's document fusion primarily affects deeper layers of the model (Layers 16-27), where the L2 distance between fused and non-fused KV caches increases by X%. This suggests that semantic integration occurs at higher abstraction levels."

**如果 Key 差异大于 Value**:
> "Interestingly, Key vectors exhibit larger distributional shifts than Value vectors (L2 distance: X.XX vs Y.YY), indicating that fusion primarily modulates attention patterns rather than altering the content representation itself."

---

## ✅ 检查清单

运行分析前确认:
- [ ] KV cache 文件存在 (no_preprocess 和 bge 两个目录)
- [ ] 有足够的内存 (建议 16GB+)
- [ ] 安装了依赖: `torch`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- [ ] 输出目录有写权限

分析完成后检查:
- [ ] 生成了所有预期的图表文件
- [ ] `summary_statistics.json` 包含所有样本
- [ ] PCA 方差解释率 > 40% (否则需注意解读)
- [ ] L2 距离趋势合理（无异常突变）

---

**Happy Analyzing! 🎉**
