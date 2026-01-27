# 文本段落 KV Cache PCA 可视化工具

这个工具用于对给定的任意N个文字段落生成KV cache，并进行PCA可视化对比。

## 文件说明

- `visualize_text_kv_pca.py` - 核心Python脚本
- `run_text_kv_pca.sh` - 便捷调用的Shell脚本
- `example_texts.txt` - 示例文本文件
- `TEXT_KV_PCA_README.md` - 本说明文档

## 功能特点

1. **直接从文本生成KV cache**：不需要预先保存的KV cache文件
2. **支持任意数量的文本段落**：可以对比2个或更多文本
3. **灵活的文本输入**：支持命令行参数或文件输入
4. **PCA可视化**：对Key和Value cache分别进行PCA降维和可视化
5. **多层分析**：可指定分析哪些层，默认选择均匀分布的6层

## 使用方法

### 方法1：使用Shell脚本（推荐）

#### 1.1 直接指定文本段落

```bash
cd /home/shm/document/exp/FusionRAG

# 对比两个段落
bash run_text_kv_pca.sh \
    --texts "这是第一个段落的内容。" "这是第二个段落的内容。"

# 对比多个段落并指定标签
bash run_text_kv_pca.sh \
    --texts \
        "Artificial intelligence is transforming healthcare." \
        "The weather today is sunny and warm." \
        "Python is a popular programming language." \
    --labels "AI Topic" "Weather Topic" "Programming Topic"
```

#### 1.2 从文件读取文本

```bash
# 使用示例文本文件
bash run_text_kv_pca.sh --text_file example_texts.txt

# 指定输出目录和GPU
bash run_text_kv_pca.sh \
    --text_file example_texts.txt \
    --output_dir ./my_analysis \
    --device cuda:0
```

#### 1.3 指定要分析的层

```bash
# 分析特定层（例如：0, 5, 10, 15, 20, 27）
bash run_text_kv_pca.sh \
    --text_file example_texts.txt \
    --layers 0 5 10 15 20 27

# 或者使用逗号分隔
bash run_text_kv_pca.sh \
    --text_file example_texts.txt \
    --layers 0,5,10,15,20,27
```

#### 1.4 查看帮助

```bash
bash run_text_kv_pca.sh --help
```

### 方法2：直接使用Python脚本

```bash
cd /home/shm/document/exp/FusionRAG

/home/shm/anaconda3/envs/fusionrag/bin/python visualize_text_kv_pca.py \
    --model_path /mnt/data3/models/Qwen2.5-7B-Instruct \
    --model_name Qwen2.5-7B-Instruct \
    --model_type qwen2 \
    --texts "Text 1" "Text 2" "Text 3" \
    --labels "Passage A" "Passage B" "Passage C" \
    --layers 0 5 10 15 20 27 \
    --output_dir ./text_kv_pca_results
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
AI技术: Artificial intelligence has revolutionized healthcare.
天气: The weather today is sunny with clear skies.
医学: Modern medicine uses personalized treatment approaches.
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
| `--max_tokens` | 每层最大token数 | 500 |
| `--output_dir` | 输出目录 | ./text_kv_pca_analysis |
| `--device` | 设备 | cuda |
| `--torch_dtype` | 数据类型 | float16 |

## 输出结果

脚本会在输出目录生成以下文件：

1. **pca_key_cache.png** - Key cache的PCA可视化图
   - 每个子图代表一个层
   - 不同颜色代表不同的文本段落
   - 坐标轴显示解释方差比例

2. **pca_value_cache.png** - Value cache的PCA可视化图
   - 结构同Key cache图
   - 显示Value cache的分布情况

3. **summary_statistics.json** - 统计摘要
   - 每层的PCA解释方差比例
   - 文本标签信息

## 示例：快速开始

```bash
# 1. 进入目录
cd /home/shm/document/exp/FusionRAG

# 2. 使用示例文本文件运行
bash run_text_kv_pca.sh --text_file example_texts.txt

# 3. 查看结果
ls -lh ./text_kv_pca_analysis/
```

## 工作原理

1. **加载模型**：使用transformers库加载指定的语言模型
2. **生成KV cache**：对每个文本段落进行前向传播，提取所有层的Key和Value cache
3. **特征提取**：从指定层提取特征，将多头注意力展平
4. **PCA降维**：对所有文本的特征进行联合PCA，降维到2维
5. **可视化**：绘制散点图，不同颜色代表不同文本

## 常见问题

### Q1: 如何修改模型路径？

在脚本中修改 `MODEL_PATH` 变量，或使用 `--model_path` 参数。

### Q2: 如何对比更多文本？

在 `--texts` 中添加更多文本，或在文件中添加更多行。

### Q3: PCA结果如何解释？

- **接近的点**：表示KV cache特征相似，语义或结构相关
- **远离的点**：表示KV cache特征差异大
- **PC1/PC2**：分别显示第一、第二主成分的解释方差比例

### Q4: 如何选择分析的层？

- **早期层（0-5）**：更关注句法特征
- **中期层（10-20）**：句法和语义混合
- **后期层（25-27）**：更关注语义特征

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

## 与原始脚本的区别

| 特性 | visualize_kv_pca_multi.py | visualize_text_kv_pca.py |
|------|--------------------------|--------------------------|
| 输入 | 预生成的KV cache文件 | 原始文本段落 |
| 数据来源 | 数据集 + 预处理方法 | 任意文本 |
| KV cache | 从磁盘加载 | 从模型实时生成 |
| 适用场景 | 对比不同预处理方法 | 分析文本特征差异 |
