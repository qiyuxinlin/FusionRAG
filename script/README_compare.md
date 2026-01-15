# FusionRAG 结果对比工具

## 功能说明

`compare_results.py` 用于比较两个不同方案的 FusionRAG result 文件，分析样本的答对/答错情况。

## 主要功能

- 统计两个方案的准确率对比
- 分析样本分类：
  - 都答对的样本
  - 只有方案1答对的样本
  - 只有方案2答对的样本
  - 都答错的样本
- 生成详细的对比报告

## 使用方法

### 基本用法

```bash
python compare_results.py \
  --file1 <第一个result文件路径> \
  --file2 <第二个result文件路径>
```

### 完整参数

```bash
python compare_results.py \
  --file1 <第一个result文件路径> \
  --file2 <第二个result文件路径> \
  --name1 "方案1名称" \
  --name2 "方案2名称" \
  --output <输出文件路径>
```

### 参数说明

- `--file1`: 第一个 result CSV 文件路径（必需）
- `--file2`: 第二个 result CSV 文件路径（必需）
- `--name1`: 第一个方案的名称（可选，默认为"方案1"）
- `--name2`: 第二个方案的名称（可选，默认为"方案2"）
- `--output`: 输出详细对比结果的文件路径（可选，默认为"comparison_detail.txt"）

## 使用示例

### 示例1: 比较不同 rate 配置

```bash
python compare_results.py \
  --file1 /home/shm/document/exp/FusionRAG/result/fixed_doc/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_fixed_doc/rate_0.0_revert_rope.csv \
  --file2 /home/shm/document/exp/FusionRAG/result/fixed_doc/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_fixed_doc/rate_0.5_revert_rope.csv \
  --name1 "Rate-0.0" \
  --name2 "Rate-0.5" \
  --output comparison_rate.txt
```

### 示例2: 比较不同 TopK 配置

```bash
python compare_results.py \
  --file1 result/topk5_result.csv \
  --file2 result/topk10_result.csv \
  --name1 "TopK-5" \
  --name2 "TopK-10" \
  --output comparison_topk.txt
```

### 示例3: 比较 Baseline 和 FusionRAG

```bash
python compare_results.py \
  --file1 result/baseline.csv \
  --file2 result/fusionrag.csv \
  --name1 "Baseline" \
  --name2 "FusionRAG" \
  --output baseline_vs_fusionrag.txt
```

## 输出说明

### 终端输出

脚本会在终端显示：
- 文件加载信息
- 统计摘要（各类样本数量和占比）
- 两个方案的准确率对比
- 准确率提升/下降情况

### 详细报告文件

脚本会生成一个详细的文本文件，包含：

1. **统计摘要部分**
   - 总样本数
   - 各类样本的数量和占比
   - 准确率对比

2. **详细样本列表**（按类别分组）
   - 只有方案1答对的样本
   - 只有方案2答对的样本
   - 都答对的样本
   - 都答错的样本

每个样本包含：
- 样本索引
- 主问题和子问题
- 标准答案
- 两个方案的预测答案
- 正确性、F1分数、EM分数

## 文件格式要求

输入的 CSV 文件需要包含以下列：
- `Main Question`: 主问题
- `Sub Question`: 子问题
- `Ground Truth`: 标准答案
- `Predicted`: 预测答案
- `Correct`: 是否正确（True/False）
- `F1`: F1分数
- `EM`: 精确匹配分数

## 注意事项

1. 两个文件的样本顺序应该一致
2. 如果样本数量不一致，会发出警告并只比较较短文件的长度
3. 输出文件默认为 UTF-8 编码

## 快速开始

查看示例脚本：
```bash
cat compare_example.sh
```

运行示例：
```bash
./compare_example.sh
```
