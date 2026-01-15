# t-SNE 多方法对比可视化 - 使用指南

## 🎉 新功能

现在t-SNE可视化也支持**多方法对比**了！仿照PCA multi的设计，可以灵活选择任意方法进行t-SNE对比分析。

## 📦 工具对比

### 原版t-SNE vs 多方法t-SNE

| 特性 | 原版 `visualize_kv_tsne.py` | 新版 `visualize_kv_tsne_multi.py` |
|------|---------------------------|----------------------------------|
| **对比方法数** | 固定2个 (no_preprocess vs bge) | 2-8个（灵活） |
| **方法选择** | 硬编码 | 命令行参数 |
| **脚本** | `run_kv_tsne_quick.sh` | `run_kv_tsne_multi.sh` |
| **适用场景** | 快速验证baseline vs bge | 任意方法对比 |

**建议**：
- ✅ 快速验证BGE效果：用原版 `run_kv_tsne_quick.sh`
- ✅ 多方法ablation对比：用新版 `run_kv_tsne_multi.sh`
- ✅ 保留两者，各有用途

---

## 🚀 快速开始

### 方式 1: 预设对比（最简单）

```bash
# 对比baseline和bge
bash run_kv_tsne_compare_preset.sh baseline 3

# 对比3种ablation方法
bash run_kv_tsne_compare_preset.sh ablation3 5

# 对比所有方法
bash run_kv_tsne_compare_preset.sh ablation_all 5
```

**可用预设**：
- `baseline` - no_preprocess vs bge
- `ablation2` - bge vs random
- `ablation3` - bge vs random vs repeat_self
- `ablation4` - bge vs random vs repeat_self vs fixed_doc
- `ablation_all` - 所有6种方法
- `random_methods` - 所有随机相关方法
- `bge_variants` - BGE及其变体

### 方式 2: 自定义方法对比（灵活）

```bash
# 对比2种方法
bash run_kv_tsne_multi.sh bge random

# 对比3种方法，5个样本
bash run_kv_tsne_multi.sh bge random repeat_self 5

# 自定义输出目录
bash run_kv_tsne_multi.sh bge random 3 ./my_tsne_output
```

### 方式 3: Python直接调用（完全控制）

```bash
python visualize_kv_tsne_multi.py \
    --methods no_preprocess bge random \
    --sample_ids 0 1 2 \
    --layers 0 11 18 27 \
    --max_tokens 500 \
    --perplexity 30 \
    --output_dir ./custom_tsne
```

---

## 🎨 支持的方法

与PCA multi完全相同：

| 方法名 | 说明 |
|--------|------|
| `no_preprocess` | 无预处理（baseline） |
| `bge` | BGE相似度召回 |
| `random` | 随机召回 |
| `repeat_self` | 重复自身 |
| `fixed_doc` | 固定文档 |
| `random_docs` | 随机文档 |
| `random_text` | 随机文本 |
| `bge_shuffled` | BGE打乱顺序 |

---

## 📊 输出文件

### 文件格式
```
tsne_key_example{N}_chunk1_{method1}_{method2}_...png
tsne_value_example{N}_chunk1_{method1}_{method2}_...png
l2_distance_trends_{method1}_{method2}_...png
summary_statistics.json
```

### 与PCA的区别

| 输出 | PCA | t-SNE |
|------|-----|-------|
| 散点图 | PC1/PC2轴，有方差解释率 | Dim1/Dim2轴，无物理意义 |
| L2距离趋势 | ✅ 相同 | ✅ 相同 |
| 统计摘要 | 包含方差解释率 | 只有L2距离 |

---

## 💡 典型使用场景

### 场景 1: 快速验证（3个样本，2种方法）

**目标**：看BGE和baseline是否有明显的聚类分离

```bash
bash run_kv_tsne_compare_preset.sh baseline 3
```

**预期时间**：3-5分钟

**查看**：
- Layer 0：两种方法应该混在一起（浅层影响小）
- Layer 27：两种方法应该形成两个簇（深层影响大）

---

### 场景 2: Ablation实验（5个样本，3-4种方法）

**目标**：对比多种召回策略的聚类效果

```bash
# 对比3种方法
bash run_kv_tsne_compare_preset.sh ablation3 5

# 或对比4种方法
bash run_kv_tsne_compare_preset.sh ablation4 5
```

**预期时间**：10-15分钟

**查看**：
- 哪些方法形成相似的簇？
- 哪些方法与BGE差异最大？

---

### 场景 3: 深入分析关键层（更多样本，特定层）

**目标**：详细分析PCA发现的关键层（如Layer 18, 27）

```bash
python visualize_kv_tsne_multi.py \
    --methods no_preprocess bge random \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 \
    --layers 18 27 \
    --max_tokens 600 \
    --perplexity 40 \
    --output_dir ./critical_layers_tsne
```

**预期时间**：10-15分钟

---

## ⚙️ 参数说明

### perplexity（困惑度）

**最重要的t-SNE参数**：

```bash
# 小perplexity (10-15): 强调局部结构
bash run_kv_tsne_multi.sh bge random 3 ./out_p10

# 中等perplexity (20-30): 平衡 [默认推荐]
bash run_kv_tsne_multi.sh bge random 3 ./out_p30

# 大perplexity (40-50): 强调全局结构
bash run_kv_tsne_multi.sh bge random 3 ./out_p50
```

### max_tokens（采样token数）

```bash
# 快速验证（200 tokens）
python visualize_kv_tsne_multi.py --methods bge random --max_tokens 200

# 标准分析（500 tokens，默认）
python visualize_kv_tsne_multi.py --methods bge random --max_tokens 500

# 高精度（800 tokens）
python visualize_kv_tsne_multi.py --methods bge random --max_tokens 800
```

### layers（分析的层）

```bash
# 默认8层（推荐）
LAYERS="0 5 11 16 18 22 25 27"

# 只分析关键层（最快）
python visualize_kv_tsne_multi.py --layers 0 18 27

# 分析所有28层（最详细，但很慢）
python visualize_kv_tsne_multi.py --layers 0 1 2 ... 27
```

---

## 🎯 PCA + t-SNE 联合工作流

### 推荐流程

```bash
# Step 1: 用PCA快速扫描（5分钟）
bash run_kv_pca_multi.sh bge random repeat_self 10

# 查看PCA结果，识别关键层和异常样本

# Step 2: 用t-SNE深入分析关键层（10分钟）
python visualize_kv_tsne_multi.py \
    --methods bge random repeat_self \
    --sample_ids 0 1 2 3 4 \
    --layers 18 25 27 \
    --perplexity 30

# Step 3: 对比两种方法
# - PCA: 看全局趋势、L2距离、方差解释率
# - t-SNE: 看局部聚类、簇分离程度
```

---

## 📈 解读t-SNE结果

### 好的t-SNE结果

✅ **清晰的簇分离**：
```
Layer 27 - Value

   🔵🔵🔵      🟠🟠🟠      🟢🟢🟢
  🔵🔵🔵🔵    🟠🟠🟠🟠    🟢🟢🟢🟢
   🔵🔵       🟠🟠        🟢🟢

🔵 = no_preprocess
🟠 = bge
🟢 = random
```
说明三种方法产生了不同的KV分布。

✅ **与L2距离一致**：
- L2距离大的方法，t-SNE图上簇间距离也大
- 验证了定量和可视化的一致性

### 需要注意的结果

⚠️ **完全混合**：
- 所有方法的点混在一起
- 可能原因1：该层影响很小（正常，如Layer 0）
- 可能原因2：perplexity不合适

⚠️ **多个子簇**：
- 每种方法内部形成多个簇
- 可能原因：token本身有不同功能（开头/中间/结尾）

---

## 🐛 常见问题

### Q1: t-SNE运行很慢

**原因**：t-SNE时间复杂度高

**解决**：
```bash
# 减少样本数（≤5）
bash run_kv_tsne_multi.sh bge random 3

# 减少层数
python visualize_kv_tsne_multi.py --layers 0 18 27

# 减少token采样
python visualize_kv_tsne_multi.py --max_tokens 200
```

### Q2: 某些样本的KV cache不存在

**现象**：
```
Warning: KV cache not found for bge: /path/to/5_1_key.pt
⚠️ Warning: Only 1 method loaded successfully for example 5. Skipping comparison.
```

**说明**：
- ✅ 脚本会自动跳过该样本
- ✅ 使用成功加载的方法继续分析
- ✅ 不会导致程序崩溃

### Q3: t-SNE结果每次运行都不一样

**原因**：t-SNE有随机性（尽管设置了random_state=42）

**解决**：
- 关注整体模式（簇的分离），而非具体坐标
- 多次运行，观察一致的模式
- 结合L2距离数值进行解释

### Q4: 如何选择perplexity？

**规则**：
- **token数 < 100**: perplexity=10-15
- **token数 100-500**: perplexity=20-30（默认）
- **token数 > 500**: perplexity=30-50

**调参实验**：
```bash
# 尝试不同perplexity
bash run_kv_tsne_multi.sh bge random 3 ./p10
# 然后修改脚本或直接调用Python改变perplexity
```

---

## 📚 相关文档

- **多方法PCA使用指南**：`MULTI_METHOD_PCA_GUIDE.md`
- **PCA vs t-SNE对比**：`PCA_vs_TSNE_COMPARISON.md`
- **工具总览**：`PCA_TOOLS_README.md`

---

## ✅ 快速参考

```bash
# 最简单：预设对比
bash run_kv_tsne_compare_preset.sh baseline 3
bash run_kv_tsne_compare_preset.sh ablation3 5

# 灵活：自定义方法
bash run_kv_tsne_multi.sh bge random 3
bash run_kv_tsne_multi.sh bge random repeat_self 5

# 完全控制：Python直接调用
python visualize_kv_tsne_multi.py \
    --methods bge random \
    --sample_ids 0 1 2 \
    --layers 0 18 27 \
    --perplexity 30

# 检查可用方法
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/ | grep preprocess
```

---

## 💪 性能提示

### t-SNE预期时间

| 配置 | 预期时间 |
|------|----------|
| 3样本 × 2方法 × 8层 | 3-5分钟 |
| 5样本 × 3方法 × 8层 | 10-15分钟 |
| 10样本 × 4方法 × 8层 | 25-30分钟 |

### 优化建议

1. **样本数量优先控制**：3-5个样本足够发现模式
2. **层数适当选择**：8层已经很全面，不必分析所有28层
3. **token采样平衡**：500默认值已经足够，不必增加到1000+
4. **perplexity使用默认**：30是平衡的选择，不必频繁调参

---

**祝分析顺利！** 🎉

现在你可以用t-SNE灵活对比任意方法，发现PCA无法显示的局部聚类结构！
