# 文本段落 KV Cache t-SNE 可视化工具

这个工具用于对给定的任意N个文字段落生成KV cache，并使用t-SNE进行降维可视化对比。

## 文件说明

- `visualize_text_kv_tsne.py` - 核心Python脚本
- `run_text_kv_tsne.sh` - 便捷调用的Shell脚本
- `example_texts.txt` - 示例文本文件（与PCA共用）
- `TEXT_KV_TSNE_README.md` - 本说明文档

## 功能特点

1. **直接从文本生成KV cache**：不需要预先保存的KV cache文件
2. **支持任意数量的文本段落**：可以对比2个或更多文本
3. **灵活的文本输入**：支持命令行参数或文件输入
4. **t-SNE可视化**：对Key和Value cache分别进行t-SNE降维和可视化
5. **多层分析**：可指定分析哪些层，默认选择均匀分布的6层
6. **局部结构保留**：t-SNE比PCA更好地保留局部结构和发现聚类

## t-SNE vs PCA

| 特性 | PCA | t-SNE |
|------|-----|-------|
| 速度 | 快 | 慢（约10-50倍） |
| 全局结构 | 保留 | 不保留 |
| 局部结构 | 部分保留 | 很好保留 |
| 聚类发现 | 较弱 | 很强 |
| 可解释性 | 有方差解释率 | 无方差解释率 |
| 随机性 | 确定性 | 随机性（需要固定random_state） |
| 适用场景 | 快速概览、全局趋势 | 发现聚类、局部模式 |

**选择建议：**
- **快速查看全局分布** → 使用PCA
- **发现文本聚类和局部相似性** → 使用t-SNE
- **需要可解释的方差** → 使用PCA
- **探索数据结构** → 两者都试试

## 使用方法

### 方法1：使用Shell脚本（推荐）

#### 1.1 直接指定文本段落

```bash
cd /home/shm/document/exp/FusionRAG

# 对比两个段落
bash run_text_kv_tsne.sh \
    --texts "这是第一个段落的内容。" "这是第二个段落的内容。"

# 对比多个段落并指定标签
bash run_text_kv_tsne.sh \
    --texts \
        "Artificial intelligence is transforming healthcare." \
        "The weather today is sunny and warm." \
        "Python is a popular programming language." \
    --labels "AI Topic" "Weather Topic" "Programming Topic"
```

#### 1.2 从文件读取文本

```bash
# 使用示例文本文件
bash run_text_kv_tsne.sh --text_file example_texts.txt

# 指定输出目录和GPU
bash run_text_kv_tsne.sh \
    --text_file example_texts.txt \
    --output_dir ./my_tsne_analysis \
    --device cuda:0
```

#### 1.3 指定要分析的层和perplexity

```bash
# 分析特定层
bash run_text_kv_tsne.sh \
    --text_file example_texts.txt \
    --layers 0 5 10 15 20 27

# 使用逗号分隔
bash run_text_kv_tsne.sh \
    --text_file example_texts.txt \
    --layers 0,5,10,15,20,27

# 指定perplexity（推荐范围：5-50）
bash run_text_kv_tsne.sh \
    --text_file example_texts.txt \
    --perplexity 50
```

#### 1.4 查看帮助

```bash
bash run_text_kv_tsne.sh --help
```

### 方法2：直接使用Python脚本

```bash
cd /home/shm/document/exp/FusionRAG

/home/shm/anaconda3/envs/fusionrag/bin/python visualize_text_kv_tsne.py \
    --model_path /mnt/data3/models/Qwen2.5-7B-Instruct \
    --model_name Qwen2.5-7B-Instruct \
    --model_type qwen2 \
    --texts "Text 1" "Text 2" "Text 3" \
    --labels "Passage A" "Passage B" "Passage C" \
    --layers 0 5 10 15 20 27 \
    --perplexity 30 \
    --output_dir ./text_kv_tsne_results
```

## 文本输入格式

### 命令行输入

使用 `--texts` 参数直接指定文本，用空格分隔：
```bash
--texts "Text 1" "Text 2" "Text 3"
```

### 文件输入

使用 `--text_file` 参数从文件读取。支持两种格式：

**格式1：简单文本（每行一个）**
```
This is the first passage.
This is the second passage.
This is the third passage.
```

**格式2：带标签（推荐）**
```
AI_Tech: Artificial intelligence has revolutionized healthcare.
Weather: The weather today is sunny with clear skies.
Medicine: Modern medicine uses personalized treatment approaches.
```

注释行（以 `#` 开头）和空行会被自动忽略。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径（必需） | - |
| `--model_name` | 模型名称（用于参考） | Qwen2.5-7B-Instruct |
| `--model_type` | 模型类型 | qwen2 |
| `--texts` | 文本段落列表 | - |
| `--text_file` | 文本文件路径 | - |
| `--labels` | 文本标签列表 | Text 0, Text 1, ... |
| `--layers` | 要分析的层索引 | 均匀分布的6层 |
| `--all_layers` | 分析所有层 | False |
| `--max_tokens` | 每层最大token数 | 50000 |
| `--perplexity` | t-SNE perplexity | 30 |
| `--output_dir` | 输出目录 | ./text_kv_tsne_analysis |
| `--device` | 设备 | cuda |
| `--torch_dtype` | 数据类型 | float16 |

### Perplexity 参数说明

`perplexity` 是t-SNE的重要参数：
- **推荐范围**：5-50
- **较小值（5-15）**：更关注局部结构，适合发现紧密聚类
- **中等值（30-50）**：平衡局部和全局结构
- **较大值（>50）**：更接近全局视图，但可能失去细节

**经验法则**：
- 样本数 < 100：perplexity = 5-15
- 样本数 100-1000：perplexity = 30-50
- 样本数 > 1000：perplexity = 50-100

## 输出结果

脚本会在输出目录生成以下文件：

1. **tsne_key_cache.png** - Key cache的t-SNE可视化图
   - 每个子图代表一个层
   - 不同颜色代表不同的文本段落
   - 坐标轴为t-SNE 1和t-SNE 2（无物理意义）

2. **tsne_value_cache.png** - Value cache的t-SNE可视化图
   - 结构同Key cache图
   - 显示Value cache的分布情况

3. **summary_statistics.json** - 统计摘要
   - 文本标签信息
   - Perplexity设置

## 示例：快速开始

```bash
# 1. 进入目录
cd /home/shm/document/exp/FusionRAG

# 2. 使用示例文本文件运行
bash run_text_kv_tsne.sh --text_file example_texts.txt

# 3. 查看结果
ls -lh ./text_kv_tsne_analysis/
```

## 工作原理

1. **加载模型**：使用transformers库加载指定的语言模型
2. **生成KV cache**：对每个文本段落进行前向传播，提取所有层的Key和Value cache
3. **特征提取**：从指定层提取特征，将多头注意力展平
4. **t-SNE降维**：对所有文本的特征进行联合t-SNE，降维到2维
5. **可视化**：绘制散点图，不同颜色代表不同文本

## 与PCA版本对比

### 使用方式

完全相同的接口！只需替换脚本名称：

```bash
# PCA版本（快，有方差解释）
bash run_text_kv_pca.sh --text_file example_texts.txt

# t-SNE版本（慢，发现聚类）
bash run_text_kv_tsne.sh --text_file example_texts.txt
```

### 何时使用哪个？

**场景1：快速查看多个文本的KV cache分布**
→ 使用PCA（几秒钟）

**场景2：发现哪些文本在KV cache空间中聚类**
→ 使用t-SNE（几分钟到十几分钟）

**场景3：需要解释方差贡献**
→ 使用PCA

**场景4：探索性分析，发现隐藏模式**
→ 使用t-SNE

**场景5：对比不同预处理方法的效果**
→ 两者都试试，PCA看全局，t-SNE看局部

## 性能说明

t-SNE比PCA慢得多：
- **PCA**：每个样本每层 < 0.1秒
- **t-SNE**：每个样本每层 1-2秒

示例：
- 5个文本，6层，每层500 tokens → t-SNE约需30秒到1分钟
- 5个文本，28层（全部层），每层500 tokens → t-SNE约需2-5分钟

## 常见问题

### Q1: 如何选择perplexity？

从默认值30开始，如果：
- 聚类太分散 → 降低perplexity（10-20）
- 聚类太紧密 → 提高perplexity（40-50）

### Q2: t-SNE结果每次不一样吗？

默认使用固定的random_state=42，结果可重现。

### Q3: t-SNE坐标轴有意义吗？

没有。t-SNE的坐标轴只是降维结果，重点看点的相对位置和聚类。

### Q4: 如何修改模型路径？

在脚本中修改 `MODEL_PATH` 变量，或使用 `--model_path` 参数。

### Q5: 为什么t-SNE比PCA慢？

t-SNE需要迭代优化（默认1000次迭代），而PCA是直接计算。这是t-SNE能保留局部结构的代价。

## 依赖项

```bash
# 在fusionrag环境中已安装
torch
transformers
numpy
matplotlib
seaborn
scikit-learn
tqdm
```

## 相关脚本

- `visualize_text_kv_pca.py` - PCA版本（快速）
- `visualize_text_kv_tsne.py` - t-SNE版本（详细聚类）
- `visualize_kv_pca_multi.py` - 多方法PCA对比（需要预生成cache）
- `visualize_kv_tsne_multi.py` - 多方法t-SNE对比（需要预生成cache）
