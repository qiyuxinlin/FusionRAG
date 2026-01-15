# KV Cache 可视化工具集

## 📦 工具总览

FusionRAG项目现在提供完整的降维可视化工具集，支持PCA和t-SNE，单方法和多方法对比。

### 工具列表

| 工具 | 用途 | 推荐场景 |
|------|------|----------|
| **原版PCA** | no_preprocess vs bge 固定对比 | 快速验证BGE效果 |
| **多方法PCA** | 任意方法灵活对比（2-8个） | Ablation实验分析 |
| **原版t-SNE** | no_preprocess vs bge，发现聚类 | 补充PCA，深层分析 |
| **多方法t-SNE** | 任意方法t-SNE对比（2-8个） | 多方法聚类对比 ✨新

---

## 🎯 工具选择指南

### 何时使用原版PCA？

```bash
bash run_kv_pca_quick.sh 10
```

✅ **适合场景**：
- 快速验证BGE融合的效果
- 只关心baseline vs 主方法的对比
- 已经习惯使用原版脚本

❌ **不适合场景**：
- 需要对比多种ablation方法
- 需要灵活选择对比方法

---

### 何时使用多方法PCA？

```bash
bash run_kv_pca_multi.sh bge random repeat_self 5
```

✅ **适合场景**：
- **Ablation实验**：对比所有召回方法
- **方法筛选**：找出最优/最差的方法
- **论文撰写**：需要展示多方法对比
- **灵活探索**：临时组合不同方法

❌ **不适合场景**：
- 只需要固定的baseline对比（用原版更快）

---

### 何时使用t-SNE？

```bash
bash run_kv_tsne_quick.sh 5
```

✅ **适合场景**：
- PCA方差解释率低（<30%）
- 想发现更清晰的聚类模式
- 需要漂亮的可视化效果
- 补充PCA的不足

❌ **不适合场景**：
- 时间紧迫（t-SNE慢10倍）
- 需要定量分析（t-SNE无方差解释率）

---

### 何时使用多方法t-SNE？✨新

```bash
bash run_kv_tsne_multi.sh bge random repeat_self 3
```

✅ **适合场景**：
- **多方法聚类对比**：想看3种以上方法的聚类模式
- **补充多方法PCA**：PCA已完成，想用t-SNE进一步验证
- **发现局部结构**：关心相似样本之间的局部关系
- **Ablation可视化**：需要更直观的多方法对比图

❌ **不适合场景**：
- 只对比2种方法（用原版t-SNE更快）
- 时间非常紧迫（多方法+t-SNE=双重慢）

**建议工作流**：
```bash
# 1. 先用多方法PCA快速扫描
bash run_kv_pca_multi.sh bge random repeat_self 5

# 2. 再用多方法t-SNE深入关键层
bash run_kv_tsne_multi.sh bge random repeat_self 3
```

---

## 🚀 快速开始

### 方案 1: 最简单 - 使用预设

```bash
# 对比baseline和bge（最常用）
bash run_kv_pca_compare_preset.sh baseline 5

# 对比所有ablation方法
bash run_kv_pca_compare_preset.sh ablation_all 5

# 可用预设列表
bash run_kv_pca_compare_preset.sh xxx  # 查看所有预设
```

### 方案 2: 灵活 - 自定义方法

```bash
# 对比2种方法
bash run_kv_pca_multi.sh bge random

# 对比3种方法，分析10个样本
bash run_kv_pca_multi.sh bge random repeat_self 10

# 对比4种方法，保存到指定目录
bash run_kv_pca_multi.sh bge random repeat_self fixed_doc 5 ./my_output
```

### 方案 3: 完全控制 - Python直接调用

**PCA多方法**：
```bash
python visualize_kv_pca_multi.py \
    --methods no_preprocess bge random \
    --sample_ids 0 1 2 3 4 \
    --layers 0 5 11 16 18 22 25 27 \
    --max_tokens 500 \
    --output_dir ./custom_analysis
```

**t-SNE多方法** ✨新：
```bash
python visualize_kv_tsne_multi.py \
    --methods bge random repeat_self \
    --sample_ids 0 1 2 \
    --layers 0 18 27 \
    --max_tokens 500 \
    --perplexity 30 \
    --output_dir ./custom_tsne
```

---

## 📚 详细文档

- **原版PCA使用指南**：无专门文档（原始脚本，直观易用）
- **多方法PCA使用指南**：[MULTI_METHOD_PCA_GUIDE.md](./MULTI_METHOD_PCA_GUIDE.md)
- **原版t-SNE使用指南**：[KV_TSNE_USAGE_GUIDE.md](./KV_TSNE_USAGE_GUIDE.md)
- **多方法t-SNE使用指南** ✨新：[TSNE_MULTI_METHOD_GUIDE.md](./TSNE_MULTI_METHOD_GUIDE.md)
- **PCA vs t-SNE对比**：[PCA_vs_TSNE_COMPARISON.md](./PCA_vs_TSNE_COMPARISON.md)

---

## 🗂️ 文件清单

### 核心Python脚本

```
visualize_kv_pca.py         # 原版PCA（no_preprocess vs bge）
visualize_kv_pca_multi.py   # 多方法PCA（任意方法对比）
visualize_kv_tsne.py        # 原版t-SNE（no_preprocess vs bge）
visualize_kv_tsne_multi.py  # 多方法t-SNE（任意方法对比）✨新
```

### Shell运行脚本

#### 原版PCA
```
run_kv_pca_quick.sh         # 快速分析（指定样本数）
run_kv_pca_range.sh         # 范围分析（指定样本范围）
run_kv_pca_analysis.sh      # 完整分析
```

#### 多方法PCA
```
run_kv_pca_multi.sh         # 多方法对比（自定义方法）
run_kv_pca_compare_preset.sh # 预设对比（常用组合）
```

#### 原版t-SNE
```
run_kv_tsne_quick.sh        # 快速分析
run_kv_tsne_range.sh        # 范围分析
```

#### 多方法t-SNE ✨新
```
run_kv_tsne_multi.sh        # 多方法对比（自定义方法）
run_kv_tsne_compare_preset.sh # 预设对比（常用组合）
```

### 文档
```
MULTI_METHOD_PCA_GUIDE.md   # 多方法PCA详细指南
TSNE_MULTI_METHOD_GUIDE.md  # 多方法t-SNE详细指南 ✨新
KV_TSNE_USAGE_GUIDE.md      # 原版t-SNE使用指南
PCA_vs_TSNE_COMPARISON.md   # PCA与t-SNE对比
PCA_TOOLS_README.md         # 本文档（工具总览）
```

---

## 🎨 可用方法列表

| 方法名 | 说明 | 适用工具 |
|--------|------|----------|
| `no_preprocess` | 无预处理（baseline） | 所有工具 |
| `bge` | BGE相似度召回（主方法） | 所有工具 |
| `random` | 随机召回 | 多方法PCA |
| `repeat_self` | 重复自身 | 多方法PCA |
| `fixed_doc` | 固定文档 | 多方法PCA |
| `random_docs` | 随机文档 | 多方法PCA |
| `random_text` | 随机文本 | 多方法PCA |
| `bge_shuffled` | BGE打乱顺序 | 多方法PCA |

**注意**：原版PCA工具只支持`no_preprocess`和`bge`的对比。

---

## 💡 典型工作流

### 工作流 1: 论文实验（推荐）

```bash
# Step 1: 快速验证baseline效果（5分钟）
bash run_kv_pca_quick.sh 10 ./pca_baseline

# Step 2: 完整ablation对比（10分钟）
bash run_kv_pca_compare_preset.sh ablation_all 10 ./pca_ablation

# Step 3: 关键层t-SNE深入分析（15分钟）
python visualize_kv_tsne.py \
    --sample_ids 0 1 2 3 4 \
    --methods no_preprocess bge \
    --layers 18 25 27 \
    --output_dir ./tsne_critical

# Step 4: 整理结果
# - pca_baseline/: baseline对比图（论文正文）
# - pca_ablation/: ablation对比图（论文补充材料）
# - tsne_critical/: 关键层t-SNE图（可选，增强可视化）
```

**预期时间**：30-40分钟

---

### 工作流 2: 快速探索（时间紧迫）

```bash
# 只用原版PCA，最快
bash run_kv_pca_quick.sh 5

# 5分钟搞定，得到：
# - L2距离趋势
# - PCA散点图
# - 统计摘要
```

**预期时间**：5分钟

---

### 工作流 3: 深入研究（充足时间）

```bash
# Step 1: PCA全面扫描（15分钟）
bash run_kv_pca_quick.sh 20 ./pca_full

# Step 2: 多方法ablation（20分钟）
bash run_kv_pca_compare_preset.sh ablation_all 20 ./ablation_full

# Step 3: 特定方法组合（10分钟）
bash run_kv_pca_multi.sh bge random repeat_self 15 ./specific_comparison

# Step 4: t-SNE补充（30分钟）
bash run_kv_tsne_quick.sh 10 ./tsne_full

# Step 5: 调参实验（可选，30分钟）
bash run_kv_tsne_quick.sh 5 ./tsne_p10 10   # perplexity=10
bash run_kv_tsne_quick.sh 5 ./tsne_p30 30   # perplexity=30
bash run_kv_tsne_quick.sh 5 ./tsne_p50 50   # perplexity=50
```

**预期时间**：1.5-2小时

---

## 📊 输出对比

### 原版PCA输出

```
kv_pca_analysis/
├── pca_key_example0_chunk1.png          # 2种方法（蓝红）
├── pca_value_example0_chunk1.png
├── l2_distance_trends.png               # 2条曲线
├── pca_variance_explained.png
└── summary_statistics.json
```

### 多方法PCA输出

```
kv_pca_multi_bge_random_repeat_self/
├── pca_key_example0_chunk1_bge_random_repeat_self.png    # 3种方法（不同颜色）
├── pca_value_example0_chunk1_bge_random_repeat_self.png
├── l2_distance_trends_bge_random_repeat_self.png         # 多子图
└── summary_statistics.json
```

### t-SNE输出

```
kv_tsne_analysis/
├── tsne_key_example0_chunk1.png         # t-SNE投影（保留局部结构）
├── tsne_value_example0_chunk1.png
├── l2_distance_trends.png               # 与PCA相同
└── summary_statistics.json
```

---

## 🎓 最佳实践建议

### 1. 先PCA后t-SNE

```bash
# ✅ 推荐：先用PCA快速扫描
bash run_kv_pca_quick.sh 10

# 查看PCA结果，识别关键层

# ✅ 然后用t-SNE深入关键层
python visualize_kv_tsne.py --sample_ids 0 1 2 --layers 18 27
```

### 2. 预设优先，自定义补充

```bash
# ✅ 推荐：先用预设快速对比
bash run_kv_pca_compare_preset.sh ablation3

# ✅ 如果需要特定组合，再自定义
bash run_kv_pca_multi.sh bge fixed_doc random_text
```

### 3. 样本数量权衡

```bash
# 快速验证：3-5个样本
bash run_kv_pca_multi.sh bge random 3

# 标准分析：5-10个样本
bash run_kv_pca_multi.sh bge random 10

# 论文级别：10-20个样本
bash run_kv_pca_multi.sh bge random 20
```

### 4. 分批分析大规模实验

```bash
# ❌ 不推荐：一次对比8种方法（图太拥挤）
bash run_kv_pca_multi.sh no_preprocess bge random repeat_self fixed_doc random_docs random_text bge_shuffled

# ✅ 推荐：分批对比
bash run_kv_pca_multi.sh no_preprocess bge random repeat_self 10 ./batch1
bash run_kv_pca_multi.sh bge fixed_doc random_docs random_text 10 ./batch2
```

---

## 🐛 故障排查

### 问题 1: "Method xxx KV cache not found"

**原因**：该方法的KV cache尚未生成

**检查**：
```bash
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
```

**解决**：先运行对应的实验生成KV cache

---

### 问题 2: 多方法对比时颜色难以区分

**解决**：
```bash
# 减少方法数量（≤4种最佳）
bash run_kv_pca_multi.sh bge random repeat_self

# 或增大图片尺寸（修改Python脚本中的figsize）
```

---

### 问题 3: t-SNE运行太慢

**解决**：
```bash
# 减少样本数
bash run_kv_tsne_quick.sh 3

# 减少层数
python visualize_kv_tsne.py --sample_ids 0 1 --layers 0 27

# 减少token采样
python visualize_kv_tsne.py --sample_ids 0 1 --max_tokens 200
```

---

## ✅ 快速参考卡片

### 我想快速验证BGE效果
```bash
bash run_kv_pca_quick.sh 5
```

### 我想对比所有ablation方法
```bash
bash run_kv_pca_compare_preset.sh ablation_all 10
```

### 我想自定义对比3种方法
```bash
bash run_kv_pca_multi.sh bge random repeat_self 5
```

### 我想用t-SNE发现聚类
```bash
bash run_kv_tsne_quick.sh 5
```

### 我想完全控制所有参数
```bash
python visualize_kv_pca_multi.py --methods ... --sample_ids ... --layers ...
```

### 我想查看所有可用预设
```bash
bash run_kv_pca_compare_preset.sh xxx
```

---

## 📞 更多帮助

- **多方法PCA详细教程**：查看 [MULTI_METHOD_PCA_GUIDE.md](./MULTI_METHOD_PCA_GUIDE.md)
- **t-SNE详细教程**：查看 [KV_TSNE_USAGE_GUIDE.md](./KV_TSNE_USAGE_GUIDE.md)
- **PCA vs t-SNE选择**：查看 [PCA_vs_TSNE_COMPARISON.md](./PCA_vs_TSNE_COMPARISON.md)

---

**祝分析顺利！** 🎉

现在你拥有了完整的KV Cache可视化分析工具集！
