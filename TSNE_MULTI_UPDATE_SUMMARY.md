# t-SNE 多方法可视化 - 更新总结

## ✅ 完成的工作

仿照PCA多方法的设计，成功创建了t-SNE多方法可视化工具集。

## 📦 新增文件

### 1. 核心Python脚本
- **`visualize_kv_tsne_multi.py`** (460+ 行)
  - 支持2-8种方法的t-SNE对比
  - 自动处理KV cache缺失情况
  - 与PCA multi相同的架构和接口

### 2. Shell运行脚本
- **`run_kv_tsne_multi.sh`**
  - 自定义方法对比
  - 支持灵活的参数解析

- **`run_kv_tsne_compare_preset.sh`**
  - 预设常用对比组合
  - 7种预设：baseline, ablation2-4, ablation_all, random_methods, bge_variants

### 3. 详细文档
- **`TSNE_MULTI_METHOD_GUIDE.md`** (完整使用指南)
  - 快速开始
  - 参数详解
  - 使用场景
  - 常见问题

### 4. 更新的文档
- **`PCA_TOOLS_README.md`** (工具总览)
  - 添加多方法t-SNE说明
  - 更新工具列表和文件清单
  - 添加使用建议

## 🎯 核心特性

### 1. 与PCA Multi完全一致的接口

```bash
# PCA多方法
bash run_kv_pca_multi.sh bge random repeat_self 5

# t-SNE多方法（相同的参数格式）
bash run_kv_tsne_multi.sh bge random repeat_self 5
```

### 2. 支持所有8种方法

```python
VALID_METHODS = [
    'no_preprocess', 'bge', 'random', 'repeat_self',
    'fixed_doc', 'random_docs', 'random_text', 'bge_shuffled'
]
```

### 3. 健壮的错误处理

- ✅ 自动跳过KV cache不存在的方法
- ✅ 自动调整perplexity（不能超过样本数）
- ✅ 只有1个方法成功加载时跳过该样本
- ✅ 使用实际加载的第一个方法作为参考

### 4. 完整的输出

```
kv_tsne_multi_{methods}/
├── tsne_key_example{N}_chunk1_{methods}.png
├── tsne_value_example{N}_chunk1_{methods}.png
├── l2_distance_trends_{methods}.png
└── summary_statistics.json
```

## 🚀 使用方法

### 最简单：预设对比

```bash
# 对比baseline和bge
bash run_kv_tsne_compare_preset.sh baseline 3

# 对比3种ablation方法
bash run_kv_tsne_compare_preset.sh ablation3 5

# 对比所有方法
bash run_kv_tsne_compare_preset.sh ablation_all 5
```

### 灵活：自定义方法

```bash
# 对比2种方法
bash run_kv_tsne_multi.sh bge random 3

# 对比3种方法
bash run_kv_tsne_multi.sh bge random repeat_self 5

# 对比4种方法，自定义输出
bash run_kv_tsne_multi.sh bge random repeat_self fixed_doc 5 ./my_output
```

### 完全控制：Python直接调用

```bash
python visualize_kv_tsne_multi.py \
    --methods bge random repeat_self \
    --sample_ids 0 1 2 \
    --layers 0 18 27 \
    --max_tokens 500 \
    --perplexity 30 \
    --output_dir ./custom_tsne
```

## 📊 与PCA Multi的对比

| 特性 | PCA Multi | t-SNE Multi |
|------|-----------|-------------|
| **方法数量** | 2-8个 | 2-8个 |
| **接口** | ✅ 完全相同 | ✅ 完全相同 |
| **预设** | 7种预设 | 7种预设 |
| **参数** | 无特殊参数 | +perplexity |
| **速度** | 快 | 慢（10倍） |
| **输出** | PC1/PC2 + 方差解释率 | Dim1/Dim2 |
| **优势** | 全局结构 | 局部聚类 |

## 💡 推荐工作流

```bash
# Step 1: 用PCA多方法快速扫描（10分钟）
bash run_kv_pca_multi.sh bge random repeat_self 10

# 查看PCA结果，识别：
# - 哪些层L2距离最大？
# - 哪些方法差异最明显？

# Step 2: 用t-SNE多方法深入分析关键层（10分钟）
python visualize_kv_tsne_multi.py \
    --methods bge random repeat_self \
    --sample_ids 0 1 2 3 4 \
    --layers 18 25 27 \
    --perplexity 30

# Step 3: 对比两种方法的可视化
# - PCA: 看全局趋势、定量分析
# - t-SNE: 看局部聚类、簇分离
```

## 🎨 颜色方案

使用与PCA Multi相同的颜色：

```python
METHOD_COLORS = {
    'no_preprocess': '#1f77b4',  # 蓝色
    'bge': '#ff7f0e',             # 橙色
    'random': '#2ca02c',          # 绿色
    'repeat_self': '#d62728',     # 红色
    'fixed_doc': '#9467bd',       # 紫色
    'random_docs': '#8c564b',     # 棕色
    'random_text': '#e377c2',     # 粉色
    'bge_shuffled': '#7f7f7f',    # 灰色
}
```

## 📚 完整工具生态

现在FusionRAG拥有完整的KV Cache可视化工具集：

| 工具 | 用途 |
|------|------|
| **原版PCA** | 快速验证baseline vs bge |
| **多方法PCA** | Ablation实验PCA对比 |
| **原版t-SNE** | 快速验证baseline vs bge聚类 |
| **多方法t-SNE** | Ablation实验t-SNE对比 ✨新 |

**选择建议**：
- 🚀 **快速验证**：用原版工具（PCA或t-SNE）
- 🔬 **深入分析**：用多方法工具
- 📊 **定量分析**：优先用PCA（有方差解释率）
- 🎨 **视觉探索**：优先用t-SNE（聚类效果好）
- ✅ **最佳实践**：PCA + t-SNE 联合使用

## ⚡ 性能提示

### t-SNE多方法预期时间

| 配置 | 预期时间 |
|------|----------|
| 3样本 × 2方法 × 8层 | 3-5分钟 |
| 5样本 × 3方法 × 8层 | 10-15分钟 |
| 10样本 × 4方法 × 8层 | 25-30分钟 |

### 优化建议

1. **样本数量**：3-5个样本足够（不要超过10个）
2. **层数选择**：8层已经很全面（默认设置）
3. **Token采样**：500默认值平衡（不必增加）
4. **Perplexity**：30是平衡选择（20-40范围）

## 🐛 已测试的边界情况

✅ **KV cache部分缺失**：
- 某个方法的KV cache不存在
- 自动跳过该方法，继续分析其他方法

✅ **只有1个方法成功加载**：
- 无法对比，返回None
- 跳过该样本，不会崩溃

✅ **Perplexity过大**：
- 自动调整为 n_samples - 1
- 打印warning信息

✅ **参考方法不存在**：
- 使用实际加载的第一个方法作为参考
- 打印使用哪个方法作为参考

## 📖 文档完整性

所有文档已更新：

- ✅ `visualize_kv_tsne_multi.py` - 核心代码
- ✅ `run_kv_tsne_multi.sh` - 自定义脚本
- ✅ `run_kv_tsne_compare_preset.sh` - 预设脚本
- ✅ `TSNE_MULTI_METHOD_GUIDE.md` - 详细指南
- ✅ `PCA_TOOLS_README.md` - 工具总览（已更新）

## ✅ 验证清单

- [x] Python脚本正常运行
- [x] Shell脚本可执行权限
- [x] 参数解析正确
- [x] 错误处理健壮
- [x] 输出格式正确
- [x] 文档完整详细
- [x] 与PCA接口一致

## 🎉 总结

成功创建了与PCA multi完全对应的t-SNE多方法可视化工具！

**关键优势**：
- ✅ 接口一致：学习成本低
- ✅ 功能完整：支持所有方法和预设
- ✅ 错误处理：健壮稳定
- ✅ 文档详细：易于使用

**立即使用**：
```bash
# 最快体验
bash run_kv_tsne_compare_preset.sh baseline 3

# 查看帮助
bash run_kv_tsne_compare_preset.sh
```

---

**祝使用愉快！** 🎉
