# 多方法 KV Cache PCA 对比分析 - 使用指南

## 🎯 新功能概述

现在你可以灵活选择**任意多个方法**进行KV cache特征分布的PCA对比分析，而不是只能固定对比`no_preprocess`和`bge`。

### 可用方法列表

| 方法名 | 说明 | 目录名 |
|--------|------|--------|
| `no_preprocess` | 无预处理（baseline） | `kv_cache` |
| `bge` | BGE相似度召回 | `preprocess_kv_cache_global_topk10_bge` |
| `random` | 随机召回 | `preprocess_kv_cache_global_topk10_random` |
| `repeat_self` | 重复自身 | `preprocess_kv_cache_global_topk10_repeat_self` |
| `fixed_doc` | 固定文档 | `preprocess_kv_cache_global_topk10_fixed_doc` |
| `random_docs` | 随机文档 | `preprocess_kv_cache_global_topk10_random_docs` |
| `random_text` | 随机文本 | `preprocess_kv_cache_global_topk10_random_text` |
| `bge_shuffled` | BGE打乱顺序 | `preprocess_kv_cache_global_topk10_bge_shuffled` |

---

## 🚀 三种使用方式

### 方式 1: 预设对比（最简单）

使用预定义的常用对比组合：

```bash
# 默认对比：no_preprocess vs bge
bash run_kv_pca_compare_preset.sh baseline

# 对比3种ablation方法
bash run_kv_pca_compare_preset.sh ablation3

# 对比所有方法（6种）
bash run_kv_pca_compare_preset.sh ablation_all

# 对比所有随机相关方法
bash run_kv_pca_compare_preset.sh random_methods
```

**可用预设**：
- `baseline` - no_preprocess vs bge（默认）
- `ablation2` - bge vs random
- `ablation3` - bge vs random vs repeat_self
- `ablation4` - bge vs random vs repeat_self vs fixed_doc
- `ablation_all` - 所有6种方法
- `random_methods` - 所有随机相关方法
- `bge_variants` - BGE及其变体

### 方式 2: 自定义方法对比（灵活）

直接指定要对比的方法：

```bash
# 对比2种方法
bash run_kv_pca_multi.sh no_preprocess bge

# 对比3种方法
bash run_kv_pca_multi.sh no_preprocess bge random

# 对比4种方法
bash run_kv_pca_multi.sh bge random repeat_self fixed_doc

# 对比2种方法，分析10个样本
bash run_kv_pca_multi.sh bge random 10

# 对比方法 + 指定样本数 + 指定输出目录
bash run_kv_pca_multi.sh bge random repeat_self 5 ./my_output
```

### 方式 3: Python直接调用（完全控制）

```bash
python visualize_kv_pca_multi.py \
    --methods no_preprocess bge random \
    --sample_ids 0 1 2 3 4 \
    --layers 0 5 11 16 18 22 25 27 \
    --max_tokens 500 \
    --output_dir ./custom_comparison
```

---

## 📖 详细示例

### 示例 1: 快速对比baseline和bge

```bash
# 最简单的方式
bash run_kv_pca_compare_preset.sh baseline

# 等价于
bash run_kv_pca_multi.sh no_preprocess bge
```

**预期输出**：
- 2种方法（蓝色=no_preprocess，橙色=bge）
- 8层的PCA对比图
- L2距离趋势图

### 示例 2: 对比BGE与3种ablation方法

```bash
# 对比bge, random, repeat_self, fixed_doc
bash run_kv_pca_multi.sh bge random repeat_self fixed_doc 5
```

**预期输出**：
- 4种方法，不同颜色区分
- 参考方法=bge，其他方法的L2距离都是相对于bge计算
- 分析5个样本

### 示例 3: 完整ablation对比（论文用）

```bash
# 对比所有6种方法，分析10个样本
bash run_kv_pca_compare_preset.sh ablation_all 10 ./paper_figures
```

**用途**：
- 论文中展示所有ablation方法的对比
- 全面了解不同召回策略对KV cache的影响

### 示例 4: 只对比随机相关方法

```bash
# 研究随机性的影响
bash run_kv_pca_compare_preset.sh random_methods 5
```

**对比的方法**：
- no_preprocess（baseline）
- random（随机召回）
- random_docs（随机文档）
- random_text（随机文本）

### 示例 5: 分析特定样本和层

```bash
# 只分析样本0-2，Layer 18和27
python visualize_kv_pca_multi.py \
    --methods no_preprocess bge random \
    --sample_ids 0 1 2 \
    --layers 18 27 \
    --max_tokens 800 \
    --output_dir ./critical_layers
```

---

## 📊 输出文件说明

### 1. PCA散点图

**文件名格式**：
```
pca_key_example{N}_chunk1_{method1}_{method2}_...png
pca_value_example{N}_chunk1_{method1}_{method2}_...png
```

**示例**：
```
pca_key_example0_chunk1_no_preprocess_bge_random.png
pca_value_example0_chunk1_no_preprocess_bge_random.png
```

**内容**：
- 多个子图，每个对应一层
- 每种方法用不同颜色的点表示
- 标题中显示L2距离（相对于第一个方法）

**颜色方案**：
- 🔵 no_preprocess - 蓝色
- 🟠 bge - 橙色
- 🟢 random - 绿色
- 🔴 repeat_self - 红色
- 🟣 fixed_doc - 紫色
- 🟤 random_docs - 棕色
- 🩷 random_text - 粉色
- ⚫ bge_shuffled - 灰色

### 2. L2距离趋势图

**文件名格式**：
```
l2_distance_trends_{method1}_{method2}_...png
```

**内容**：
- 对于每个非参考方法，绘制其与参考方法的L2距离趋势
- 2行子图：上排=Key，下排=Value
- 每列对应一个方法

**示例**：如果对比`no_preprocess bge random repeat_self`：
- 参考方法=no_preprocess
- 3列子图：bge, random, repeat_self

### 3. 统计摘要

**文件名**：`summary_statistics.json`

**内容**：
```json
[
  {
    "example_id": 0,
    "chunk_id": 1,
    "methods": ["no_preprocess", "bge", "random"],
    "layers": {
      "0": {
        "l2_distances": {
          "no_preprocess": {"key": 0.0, "val": 0.0},
          "bge": {"key": 1.75, "val": 0.02},
          "random": {"key": 2.31, "val": 0.05}
        },
        "key_variance_explained": [0.116, 0.091],
        "val_variance_explained": [0.137, 0.070]
      },
      ...
    }
  },
  ...
]
```

---

## 🎨 可视化解读

### 2种方法对比

当对比2种方法时（如`no_preprocess`和`bge`）：

```
Layer 0 - Key                    Layer 27 - Key
L2 dist: 1.75                    L2 dist: 17.54

● ● ●  ●                         ●●●●
 ● ● ● ●                         ●●●
● ● ● ●
                                      ■■■■
▲ ▲ ▲ ▲                               ■■■
 ▲ ▲ ▲                                ■■

● = no_preprocess               ● = no_preprocess (聚成一簇)
▲ = bge                         ■ = bge (另一簇)
```

**解读**：
- Layer 0：两种方法混在一起 → BGE影响小
- Layer 27：两种方法明显分离 → BGE影响大

### 多种方法对比（3+）

当对比3种或更多方法时：

```
Layer 27 - Value
L2 vs no_preprocess: bge:48.26, random:52.11, repeat_self:45.32

     ●●●         ■■■■        ▲▲▲       ◆◆◆
    ●●●●        ■■■■■       ▲▲▲▲     ◆◆◆◆
     ●●         ■■■          ▲▲       ◆◆

● = no_preprocess (参考)
■ = bge
▲ = random
◆ = repeat_self
```

**解读**：
- 每种方法形成独立的簇
- 簇间距离反映方法差异
- L2数值量化了差异程度

---

## 💡 常见分析场景

### 场景 1: 验证BGE的有效性

**问题**：BGE融合是否真的改变了KV cache？

```bash
bash run_kv_pca_compare_preset.sh baseline 10
```

**查看**：
- Layer 0-11：应该较混合（浅层影响小）
- Layer 18-27：应该明显分离（深层影响大）
- Value L2距离应该远大于Key L2距离

### 场景 2: 对比所有ablation方法

**问题**：哪种召回方法的KV分布最接近/远离BGE？

```bash
bash run_kv_pca_compare_preset.sh ablation_all 5
```

**查看**：
- L2距离趋势图：哪种方法的曲线与BGE最接近？
- PCA散点图：哪些方法形成相似的簇？

### 场景 3: 研究随机性的影响

**问题**：不同类型的随机方法有何区别？

```bash
bash run_kv_pca_compare_preset.sh random_methods 5
```

**对比**：
- random（随机召回） vs random_docs（随机文档） vs random_text（随机文本）
- 看哪种随机方式的分布最接近no_preprocess

### 场景 4: 论文补充材料

**问题**：需要展示完整的ablation实验结果

```bash
# 生成所有方法的对比图
bash run_kv_pca_compare_preset.sh ablation_all 20 ./paper_supplementary

# 只对比关键层
python visualize_kv_pca_multi.py \
    --methods no_preprocess bge random repeat_self fixed_doc \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 \
    --layers 0 11 18 27 \
    --output_dir ./paper_key_layers
```

---

## ⚙️ 参数详解

### run_kv_pca_multi.sh 参数

```bash
bash run_kv_pca_multi.sh <方法1> <方法2> ... [样本数] [输出目录]
```

**参数识别规则**：
- **方法名**：匹配预定义的方法列表（如`bge`, `random`）
- **数字**：识别为样本数量
- **包含`/`或`.`的字符串**：识别为输出目录

**示例**：
```bash
# 2个方法，默认5样本
bash run_kv_pca_multi.sh bge random

# 3个方法，10个样本
bash run_kv_pca_multi.sh bge random repeat_self 10

# 2个方法，5样本，自定义输出
bash run_kv_pca_multi.sh bge random 5 ./my_comparison

# 4个方法，自定义输出（默认5样本）
bash run_kv_pca_multi.sh bge random repeat_self fixed_doc ./output
```

### Python脚本参数

```bash
python visualize_kv_pca_multi.py \
    --methods <方法列表> \
    --sample_ids <样本ID列表> \
    --layers <层列表> \
    --max_tokens <最大token数> \
    --output_dir <输出目录>
```

**完整示例**：
```bash
python visualize_kv_pca_multi.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --model_name Qwen2.5-7B-Instruct \
    --methods no_preprocess bge random \
    --sample_ids 0 1 2 3 4 \
    --chunk_id 1 \
    --layers 0 5 11 16 18 22 25 27 \
    --max_layers 28 \
    --max_tokens 500 \
    --output_dir ./my_analysis
```

---

## 🔍 L2距离计算说明

### 参考方法

**第一个指定的方法作为参考**（L2=0）

示例：
```bash
bash run_kv_pca_multi.sh no_preprocess bge random
```
- 参考方法 = `no_preprocess`
- `bge`的L2距离 = 相对于`no_preprocess`
- `random`的L2距离 = 相对于`no_preprocess`

### 改变参考方法

```bash
# 使用bge作为参考
bash run_kv_pca_multi.sh bge random repeat_self fixed_doc

# 现在所有L2距离都是相对于bge
```

### 多重对比

如果需要多种参考方法的对比，分别运行：

```bash
# 以no_preprocess为参考
bash run_kv_pca_multi.sh no_preprocess bge random 5 ./ref_noprep

# 以bge为参考
bash run_kv_pca_multi.sh bge random repeat_self 5 ./ref_bge

# 以random为参考
bash run_kv_pca_multi.sh random repeat_self fixed_doc 5 ./ref_random
```

---

## 📈 与原版PCA脚本的对比

| 特性 | 原版 `visualize_kv_pca.py` | 新版 `visualize_kv_pca_multi.py` |
|------|---------------------------|----------------------------------|
| **对比方法数** | 固定2个 | 2-8个（灵活） |
| **方法选择** | 硬编码 | 命令行参数 |
| **L2距离** | 双向对比 | 相对于参考方法 |
| **颜色方案** | 固定蓝红 | 8种预定义颜色 |
| **适用场景** | baseline vs bge | 任意方法对比 |

**建议**：
- ✅ 快速验证：用原版 `run_kv_pca_quick.sh`
- ✅ 多方法对比：用新版 `run_kv_pca_multi.sh`
- ✅ 保留两者，各有用途

---

## 🐛 常见问题

### Q1: "Unknown method: xxx"

**原因**：方法名拼写错误或不存在

**解决**：检查方法名是否在支持列表中
```bash
# 支持的方法
no_preprocess, bge, random, repeat_self, fixed_doc,
random_docs, random_text, bge_shuffled
```

### Q2: "KV cache not found for xxx"

**原因**：该方法的KV cache尚未生成

**解决**：
1. 检查目录是否存在：
   ```bash
   ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
   ```
2. 如果不存在，需要先运行对应的实验生成KV cache

### Q3: 对比3个以上方法时图太拥挤

**解决方案1**：分批对比
```bash
# 第一批
bash run_kv_pca_multi.sh no_preprocess bge random

# 第二批
bash run_kv_pca_multi.sh no_preprocess repeat_self fixed_doc
```

**解决方案2**：只分析关键层
```bash
python visualize_kv_pca_multi.py \
    --methods no_preprocess bge random repeat_self \
    --layers 0 18 27 \  # 只分析3层
    --sample_ids 0 1 2
```

### Q4: 如何快速找到最相似/最不同的方法？

**查看L2距离趋势图**：
```bash
bash run_kv_pca_compare_preset.sh ablation_all 5

# 然后查看
ls ./pca_ablation_all/l2_distance_trends_*.png
```

**查看summary_statistics.json**：
```bash
cat ./pca_ablation_all/summary_statistics.json | grep "l2_distances" -A 10
```

---

## ✅ 快速参考

```bash
# 最简单：预设对比
bash run_kv_pca_compare_preset.sh baseline          # no_preprocess vs bge
bash run_kv_pca_compare_preset.sh ablation3 10      # 3方法，10样本

# 灵活：自定义方法
bash run_kv_pca_multi.sh bge random                 # 2方法
bash run_kv_pca_multi.sh bge random repeat_self 5   # 3方法，5样本

# 完全控制：Python直接调用
python visualize_kv_pca_multi.py \
    --methods no_preprocess bge random \
    --sample_ids 0 1 2 \
    --layers 0 18 27

# 检查可用方法
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/ | grep preprocess
```

---

**祝分析顺利！** 🎉

现在你可以灵活对比任意方法组合，全面了解不同召回策略对KV cache特征分布的影响。
