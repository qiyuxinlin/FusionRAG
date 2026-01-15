# KV Cache t-SNE 分析 - 使用指南

## 📝 t-SNE vs PCA

### 主要区别

| 特性 | t-SNE | PCA |
|------|-------|-----|
| **优势** | 保留局部结构，聚类效果好 | 保留全局结构，有理论解释 |
| **适用场景** | 发现数据聚类和模式 | 降维和方差分析 |
| **速度** | 慢（每层1-2秒） | 快（毫秒级） |
| **可解释性** | 低（轴无明确含义） | 高（主成分有方差贡献率） |
| **参数** | 需调整perplexity | 几乎无参数 |
| **可复现性** | 需设置随机种子 | 确定性算法 |

### 何时使用t-SNE？

✅ **推荐使用t-SNE的场景**：
1. **发现聚类**：想看No Preprocess和BGE是否形成明显分离的簇
2. **局部结构**：关心相似样本之间的局部关系
3. **视觉探索**：需要直观的可视化效果
4. **补充PCA**：PCA方差解释率低时，t-SNE可能提供更好的可视化

❌ **不推荐使用t-SNE的场景**：
1. **需要定量分析**：t-SNE的距离不能直接解释
2. **时间紧迫**：样本多或层多时，t-SNE很慢
3. **需要可复现**：虽然设置了随机种子，但结果仍有随机性

### 建议工作流

**第一步：用PCA快速探索**（5分钟）
```bash
bash run_kv_pca_quick.sh 5
```
- 快速了解整体趋势
- 查看方差解释率
- 确定重点层和样本

**第二步：用t-SNE深入分析**（10-15分钟）
```bash
bash run_kv_tsne_quick.sh 5
```
- 重点关注PCA中发现的有趣层
- 寻找更清晰的聚类模式
- 验证BGE融合的影响

---

## 🚀 快速开始

### 方式 1: 快速分析（指定样本数量）

```bash
# 分析前5个样本（默认）
bash run_kv_tsne_quick.sh 5

# 分析前3个样本
bash run_kv_tsne_quick.sh 3

# 分析10个样本，保存到指定目录
bash run_kv_tsne_quick.sh 10 ./my_tsne_output

# 分析5个样本，使用perplexity=40
bash run_kv_tsne_quick.sh 5 ./output 40
```

### 方式 2: 范围分析（指定样本范围）

```bash
# 分析样本 0-5
bash run_kv_tsne_range.sh 0 5

# 分析样本 10-15
bash run_kv_tsne_range.sh 10 15

# 分析样本 0-3，perplexity=20
bash run_kv_tsne_range.sh 0 3 ./output 20
```

### 方式 3: 直接调用Python（完全控制）

```bash
/home/shm/anaconda3/envs/fusionrag/bin/python visualize_kv_tsne.py \
    --sample_ids 0 1 2 \
    --chunk_id 1 \
    --layers 0 7 14 21 27 \
    --max_tokens 500 \
    --perplexity 30 \
    --output_dir ./my_custom_tsne
```

---

## ⚙️ 参数详解

### perplexity（困惑度）

**最重要的t-SNE参数**，控制局部/全局结构的平衡。

```bash
# 小perplexity (5-15): 强调局部结构
bash run_kv_tsne_quick.sh 5 ./output 10

# 中等perplexity (20-30): 平衡 [默认推荐]
bash run_kv_tsne_quick.sh 5 ./output 30

# 大perplexity (40-50): 强调全局结构
bash run_kv_tsne_quick.sh 5 ./output 50
```

**选择建议**：
- **token数量 < 100**: perplexity=10-15
- **token数量 100-500**: perplexity=20-30（默认）
- **token数量 > 500**: perplexity=30-50

**注意**：perplexity必须小于样本数量（代码会自动调整）

### max_tokens（采样token数）

控制每层分析多少个token：

```bash
# 快速验证（减少token）
python visualize_kv_tsne.py --sample_ids 0 1 --max_tokens 200

# 标准分析（默认）
python visualize_kv_tsne.py --sample_ids 0 1 --max_tokens 500

# 高精度（增加token）
python visualize_kv_tsne.py --sample_ids 0 1 --max_tokens 1000
```

**权衡**：
- token多 → 更准确，但t-SNE更慢
- token少 → 更快，但可能丢失细节

### layers（分析的层）

```bash
# 默认8层（快速但全面）
LAYERS="0 5 11 16 18 22 25 27"

# 只分析首中尾3层（最快）
python visualize_kv_tsne.py --sample_ids 0 --layers 0 14 27

# 分析所有28层（最详细，但很慢）
python visualize_kv_tsne.py --sample_ids 0 --layers 0 1 2 3 ... 27
```

---

## 📊 输出文件说明

### 1. t-SNE 散点图

**文件名**：
- `tsne_key_example{N}_chunk1.png`
- `tsne_value_example{N}_chunk1.png`

**内容**：多个子图，每个子图显示一层的 2D t-SNE 投影
- 🔵 蓝色点 = No Preprocess
- 🔴 红色点 = BGE

**如何解读**：
- **明显分离的簇** → BGE融合对该层影响大
- **混合在一起** → BGE融合对该层影响小
- **聚类紧密度** → 方法内部的一致性
- **簇间距离** → 两种方法的差异程度

### 2. L2 距离趋势图

**文件名**：`l2_distance_trends.png`

**内容**：与PCA相同，展示L2距离随层数的变化

**用途**：与t-SNE散点图配合，定量验证可视化结果

### 3. 统计摘要

**文件名**：`summary_statistics.json`

**内容**：L2距离数值（不包含方差解释率，因为t-SNE无此概念）

---

## 🎯 常见使用场景

### 场景 1: 快速验证（3个样本）

**目标**：快速看看t-SNE能否发现PCA未发现的模式

```bash
# 先跑PCA（30秒）
bash run_kv_pca_quick.sh 3 ./pca_out

# 再跑t-SNE（3分钟）
bash run_kv_tsne_quick.sh 3 ./tsne_out

# 对比两种方法的可视化
```

**预期时间**：3-5分钟

---

### 场景 2: 标准分析（5-10个样本）

**目标**：获得可信的t-SNE聚类结果

```bash
# 中等样本，默认perplexity
bash run_kv_tsne_quick.sh 5
```

**预期时间**：5-10分钟

**建议**：论文使用这个规模

---

### 场景 3: 调参实验

**目标**：找到最佳的perplexity参数

```bash
# 尝试不同perplexity
bash run_kv_tsne_quick.sh 3 ./tsne_p10 10
bash run_kv_tsne_quick.sh 3 ./tsne_p30 30
bash run_kv_tsne_quick.sh 3 ./tsne_p50 50

# 对比结果，选择最清晰的聚类
```

**预期时间**：10-15分钟

---

### 场景 4: 特定层深入分析

**目标**：详细分析PCA报告中发现的关键层（如Layer 18, 27）

```bash
# 只分析关键层，但增加样本和token数
python visualize_kv_tsne.py \
    --sample_ids 0 1 2 3 4 5 \
    --layers 18 25 27 \
    --max_tokens 800 \
    --perplexity 40 \
    --output_dir ./critical_layers_tsne
```

**预期时间**：5-8分钟

---

## ⚡ 性能优化

### 如果t-SNE太慢

```bash
# 1. 减少样本数
bash run_kv_tsne_quick.sh 3  # 而不是10

# 2. 减少层数
python visualize_kv_tsne.py --sample_ids 0 1 --layers 0 14 27

# 3. 减少token采样
python visualize_kv_tsne.py --sample_ids 0 1 --max_tokens 200

# 4. 组合优化
bash run_kv_tsne_quick.sh 3  # 3样本
# 并修改脚本：MAX_TOKENS="300", LAYERS="0 14 27"
```

### 如果内存不足

```bash
# 减少max_tokens
MAX_TOKENS="200"

# 或一次只分析1个样本
bash run_kv_tsne_range.sh 0 0  # 只分析样本0
```

---

## 🔍 t-SNE 结果解读

### 好的t-SNE结果

✅ **清晰的簇分离**：
- 蓝色点（No Preprocess）和红色点（BGE）形成两个明显的簇
- 簇内点聚集紧密，簇间距离大
- 说明BGE融合确实改变了特征分布

✅ **与L2距离一致**：
- L2距离大的层，t-SNE图上簇分离也明显
- 验证了定量和可视化的一致性

### 需要注意的结果

⚠️ **完全混合**：
- 蓝红点完全混在一起
- 可能原因1：该层BGE影响很小（L2距离应该也小）
- 可能原因2：perplexity设置不当

⚠️ **多个子簇**：
- 每种方法内部形成多个簇
- 可能原因：token之间本身就有不同的功能（如开头/中间/结尾token）

⚠️ **异常点**：
- 远离簇中心的孤立点
- 可能是特殊token（如[SEP], [CLS]）或采样artifact

---

## 📚 参数速查表

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `样本数量` | 分析多少个样本 | 5 | 3-10 |
| `--perplexity` | t-SNE困惑度 | 30 | 10-50 |
| `--layers` | 分析哪些层 | 8层 | 3-10层 |
| `--max_tokens` | 每层采样token数 | 500 | 200-800 |
| `--chunk_id` | 分析哪个chunk | 1 | 1 (文档) |

---

## 🐛 常见问题

### Q1: "perplexity must be less than n_samples"

**原因**：样本数太少（token数 < perplexity）

**解决**：
```bash
# 自动调整（代码会处理）
# 或手动降低perplexity
bash run_kv_tsne_quick.sh 5 ./output 10  # 使用更小的perplexity
```

### Q2: t-SNE结果每次运行都不一样

**原因**：t-SNE有随机性（尽管设置了random_state=42）

**解决**：
- 关注整体模式（簇的分离），而非具体坐标
- 多次运行，观察一致的模式
- 结合L2距离数值进行解释

### Q3: t-SNE图上没有看到明显的簇分离

**可能原因**：
1. **该层BGE影响确实小**：查看L2距离验证
2. **perplexity不合适**：尝试调整（10, 30, 50）
3. **token采样不足**：增加max_tokens
4. **两种方法本身就相似**：这是正常的发现

---

## ✅ 快速参考

```bash
# 最简单：分析3个样本
bash run_kv_tsne_quick.sh 3

# 标准分析：5个样本
bash run_kv_tsne_quick.sh 5

# 调参实验：尝试不同perplexity
bash run_kv_tsne_quick.sh 3 ./out_p10 10
bash run_kv_tsne_quick.sh 3 ./out_p30 30
bash run_kv_tsne_quick.sh 3 ./out_p50 50

# 范围分析：样本 10-15
bash run_kv_tsne_range.sh 10 15

# 自定义：完全控制
python visualize_kv_tsne.py \
    --sample_ids 0 1 2 \
    --layers 0 14 27 \
    --max_tokens 500 \
    --perplexity 30 \
    --output_dir ./output
```

---

## 🎓 PCA + t-SNE 联合分析推荐流程

```bash
# 第一步：PCA快速扫描（5分钟）
bash run_kv_pca_quick.sh 10 ./pca_results

# 查看 pca_results/PCA_ANALYSIS_REPORT.md 或 l2_distance_trends.png
# 识别关键层（如Layer 18, 27）和异常样本

# 第二步：t-SNE深入分析关键层（10分钟）
python visualize_kv_tsne.py \
    --sample_ids 0 1 2 3 4 \
    --layers 18 25 27 \
    --max_tokens 600 \
    --perplexity 30 \
    --output_dir ./tsne_critical_layers

# 第三步：对比两种方法的可视化
# PCA: 看全局趋势和方差解释
# t-SNE: 看局部聚类和簇分离

# 第四步：结合L2距离数值，写分析报告
```

---

**祝分析顺利！🎉**

**提示**：t-SNE更慢但可能发现PCA看不到的局部结构。两种方法互补使用效果最佳！
