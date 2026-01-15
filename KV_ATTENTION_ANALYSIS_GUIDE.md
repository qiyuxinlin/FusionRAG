# KV Cache 和注意力分析工具使用指南

## 概述

这套工具用于深入分析 FusionRAG 中不同召回方法的机理差异，包括：

1. **KV Cache 统计分析** - 对比不同方法生成的 KV cache 数值特性
2. **KV Cache 相似度分析** - 计算不同方法 KV cache 之间的相似度
3. **注意力可视化** - 可视化模型在生成时的注意力分布

## 工具列表

### 1. `analyze_sample_kv.py` - KV Cache 分析工具

**功能**：
- 从结果CSV中筛选符合条件的样本
- 加载并对比不同方法的KV cache
- 计算统计特性（均值、方差、范数、稀疏性等）
- 计算相似度矩阵（余弦相似度、L2距离、MSE）
- 生成可视化图表

**使用方法**：

```bash
# 基础用法（使用默认配置）
python analyze_sample_kv.py

# 自定义CSV路径
python analyze_sample_kv.py \
    --bge_csv /path/to/bge/results.csv \
    --random_csv /path/to/random/results.csv \
    --no_prep_csv /path/to/no_prep/results.csv

# 选择不同的筛选条件
python analyze_sample_kv.py \
    --condition bge_correct_noprep_wrong_random_correct \
    --max_samples 10
```

**筛选条件**：

- `bge_correct_noprep_wrong_random_correct`: BGE答对、no_preprocess答错、Random答对
  - **核心假设验证**：如果Random和BGE都对、no_prep错，说明preprocess起作用，但可能不是因为语义相似度

- `bge_correct_random_wrong`: BGE答对、Random答错
  - **语义相似度重要性验证**：如果BGE对但Random错，说明召回相似文档确实有用

- `all_methods_correct`: 三种方法都对
  - **Baseline分析**：这些是简单样本

- `all_methods_wrong`: 三种方法都错
  - **难例分析**：这些是困难样本

**输出**：

生成的文件会保存在 `kv_analysis_output/` 目录：

- `sample_<id>_kv_analysis.png`: KV cache对比可视化（6个子图）
  - Key范数分布
  - Value范数分布
  - 相似度矩阵热图
  - 统计特性对比（Key和Value）
  - MSE对比

- `sample_<id>_report.json`: 详细分析报告（JSON格式）
  - 问题信息
  - 三种方法的预测结果
  - KV统计特性
  - 相似度矩阵

---

### 2. `visualize_attention.py` - 注意力可视化工具

**功能**：
- 对指定样本进行推理
- Hook模型的attention层，记录注意力权重
- 分析模型关注了哪些输入token
- 对比不同文档区域获得的注意力

**使用方法**：

```bash
# 分析单个样本
python visualize_attention.py \
    --sample_id 10 \
    --method bge \
    --output_dir ./attention_analysis_output

# 对比不同方法
for method in bge random no_prep; do
    python visualize_attention.py \
        --sample_id 10 \
        --method $method
done
```

**输出**：

生成的文件会保存在 `attention_analysis_output/` 目录：

- `sample_<id>_<method>_attention_heatmap.png`: 注意力热图
  - 展示生成的每个token对输入的注意力分布
  - 显示3个代表性层（首层、中间层、末层）

- `sample_<id>_<method>_doc_attention.png`: 文档注意力分布图
  - 展示每层对不同文档区域的注意力占比

- `sample_<id>_<method>_attention_report.json`: 注意力分析报告

---

## 完整分析流程

### Step 1: 找到有趣的样本

```bash
# 运行KV分析工具，找到符合条件的样本
python analyze_sample_kv.py \
    --condition bge_correct_noprep_wrong_random_correct \
    --max_samples 5
```

**查看输出**：
- 工具会打印出符合条件的样本列表
- 例如: `Sample #25: What is the capital of the county...`

### Step 2: 深入分析KV cache

查看生成的 `sample_25_kv_analysis.png`，重点关注：

**1. 相似度矩阵（右上角热图）**
- 如果 `sim(BGE, Random) ≈ sim(BGE, Repeat_Self) ≈ 0.95+`
  - **验证假设：三种方法的KV高度相似**
  - 说明召回方法对KV cache影响不大
  - 支持"位置适应"假设

**2. 统计特性对比（底部柱状图）**
- 对比 `norm_mean`, `std`, `sparsity`
- 如果三种方法数值接近
  - 说明KV cache在数值分布上相似
  - 进一步支持假设

**3. MSE对比（右下角）**
- 如果 MSE 都很小（< 1e-4）
  - 说明KV cache几乎相同
  - 性能差异可能来自其他因素

### Step 3: 注意力分析

```bash
# 对同一样本运行三种方法的注意力分析
python visualize_attention.py --sample_id 25 --method bge
python visualize_attention.py --sample_id 25 --method random
python visualize_attention.py --sample_id 25 --method no_prep
```

**对比三个热图**，重点关注：

**1. 跨文档注意力**
- 模型是否真的关注融合的其他文档？
- BGE召回的相似文档是否获得更多注意力？

**2. 位置效应**
- 同样内容在不同位置的注意力权重是否不同？
- No_preprocess是否因为位置问题无法关注到关键信息？

**3. 层间差异**
- 浅层 vs 深层的注意力模式
- 某些层是否更依赖跨文档信息？

---

## 关键假设的验证路径

### 假设：FusionRAG的性能主要来自Positional Adaptation，而非Cross-Document Attention

#### 验证路径 1: KV Cache 相似度

**预期结果（支持假设）**：
```
sim(BGE, Random) ≈ 0.95+
sim(BGE, Repeat_Self) ≈ 0.95+
sim(Random, Repeat_Self) ≈ 0.95+
```

**结论**：KV cache高度相似，召回方法影响不大

**反驳假设的结果**：
```
sim(BGE, Random) < 0.8
sim(BGE, Repeat_Self) > 0.9
```
说明BGE确实产生了不同的KV表示

---

#### 验证路径 2: 注意力分布

**预期结果（支持假设）**：
```
跨文档注意力占比 < 10%
主要注意力集中在问题和当前文档
BGE vs Random 的注意力分布差异不大
```

**结论**：模型并未充分利用融合的其他文档

**反驳假设的结果**：
```
BGE方法下，相似文档获得 20%+ 注意力
Random方法下，随机文档获得 < 5% 注意力
```
说明语义相似度确实影响了模型行为

---

#### 验证路径 3: 统计特性

**预期结果（支持假设）**：
```
三种方法的norm_mean、std、sparsity都接近
```

**结论**：KV cache在数值特性上相似

---

## 输出文件解读

### KV Analysis Report (JSON)

```json
{
  "sample_idx": 25,
  "question": "What is the capital...",
  "kv_similarity": {
    "bge_vs_random": {
      "key_cosine_similarity": 0.9876,  // 接近1说明高度相似
      "value_cosine_similarity": 0.9823,
      "key_mse": 1.23e-05,  // 越小说明越接近
      "value_mse": 2.45e-05
    },
    ...
  },
  "kv_statistics": {
    "bge": {
      "key_norm_mean": 12.34,
      "key_std": 5.67,
      "key_sparsity": 0.12  // 稀疏性
    },
    ...
  }
}
```

**关键指标解读**：

| 指标 | 阈值 | 含义 |
|------|------|------|
| **Cosine Similarity** | > 0.95 | 高度相似 |
| | 0.8 - 0.95 | 中等相似 |
| | < 0.8 | 差异明显 |
| **MSE** | < 1e-4 | 几乎相同 |
| | 1e-4 - 1e-2 | 有差异但不大 |
| | > 1e-2 | 差异显著 |
| **Norm Mean** | 相对差异 < 10% | 分布相似 |
| **Sparsity** | 绝对差异 < 0.05 | 稀疏性相似 |

---

## 常见问题

### Q1: 找不到符合条件的样本怎么办？

**A**: 尝试其他筛选条件，例如：
```bash
python analyze_sample_kv.py --condition bge_correct_random_wrong
```

或者查看CSV文件，手动选择感兴趣的样本。

---

### Q2: KV cache文件路径不对怎么办？

**A**: 修改脚本顶部的配置区域：

```python
# 在 analyze_sample_kv.py 中修改
BGE_CACHE_DIR = "/your/custom/path/preprocess_kv_cache_global_topk10_bge"
RANDOM_CACHE_DIR = "/your/custom/path/preprocess_kv_cache_global_topk10_random"
NO_PREP_CACHE_DIR = "/your/custom/path/kv_cache"
```

---

### Q3: 注意力可视化工具报错怎么办？

**A**: 常见问题：
1. **GPU内存不足**: 使用CPU或减少模型精度
   ```python
   torch_dtype=torch.float32  # 改为float32
   ```

2. **Hook没有捕获到attention**: 检查模型架构，调整hook匹配逻辑
   ```python
   # 在 AttentionHook.register_hooks() 中调整匹配条件
   ```

3. **输出attentions需要特殊配置**: 某些模型需要在config中设置
   ```python
   config.output_attentions = True
   ```

---

### Q4: 如何批量分析多个样本？

**A**: 使用bash循环：

```bash
# 分析前10个符合条件的样本
python analyze_sample_kv.py --max_samples 10 > samples.log

# 从log中提取样本ID
grep "Sample #" samples.log | awk '{print $2}' | cut -d':' -f1 | cut -d'#' -f2 > sample_ids.txt

# 批量运行注意力分析
while read sample_id; do
    for method in bge random no_prep; do
        python visualize_attention.py --sample_id $sample_id --method $method
    done
done < sample_ids.txt
```

---

## 进一步分析

### 定量对比实验

创建一个汇总脚本，统计所有样本的平均相似度：

```bash
# 运行多个样本
python analyze_sample_kv.py --max_samples 20

# 提取所有JSON报告中的相似度数据
python -c "
import json
import glob
import numpy as np

files = glob.glob('kv_analysis_output/sample_*_report.json')
similarities = []

for f in files:
    with open(f) as fp:
        data = json.load(fp)
        sim = data['kv_similarity']['bge_vs_random']['key_cosine_similarity']
        similarities.append(sim)

print(f'平均相似度: {np.mean(similarities):.4f}')
print(f'标准差: {np.std(similarities):.4f}')
print(f'中位数: {np.median(similarities):.4f}')
"
```

---

## 论文分析建议

基于这些工具的输出，你可以在论文中包括：

### 图表1: KV Cache相似度分布
- 横轴：样本编号
- 纵轴：余弦相似度
- 三条曲线：BGE vs Random, BGE vs Repeat_Self, Random vs Repeat_Self
- **结论**：如果三条线都接近1，支持"位置适应"假设

### 图表2: 统计特性对比箱线图
- 对比20个样本的 norm_mean, std, sparsity
- 三个方法的箱线图并排
- **结论**：如果箱线图重叠度高，说明分布相似

### 图表3: 注意力热图对比
- 并排展示BGE、Random、No_Preprocess的attention heatmap
- 突出显示跨文档注意力占比
- **结论**：如果BGE和Random的热图相似，支持假设

### 图表4: 文档注意力分布饼图
- 展示模型注意力在不同区域的占比
- 对比BGE召回的相似文档 vs Random召回的文档
- **结论**：如果占比差异不大，说明语义相似度作用有限

---

**创建时间**: 2026-01-14
**版本**: 1.0
