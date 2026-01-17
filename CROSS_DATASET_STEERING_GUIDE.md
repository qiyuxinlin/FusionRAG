# 跨数据集 Steering Vectors 使用指南

## 概述

`run_fusionrag_steering.sh` 脚本现在支持**分别指定训练数据集和测试数据集**，实现跨数据集的 steering vector 应用。

---

## 核心配置参数

### 数据集配置

```bash
# 数据配置
TRAIN_DATASET="2wikimqa"              # 用于计算 steering vectors 的数据集（影响 KV cache 路径）
TEST_DATASET="musique"                # 用于实际测试的数据集
TEST_DATA_PATH="./data/result_reflect.json"  # 测试数据的 JSON 文件路径
```

### 参数说明

| 参数 | 作用阶段 | 说明 |
|------|---------|------|
| `TRAIN_DATASET` | Step 0 (统计) | 用于读取 KV cache 计算 steering vectors |
| `TEST_DATASET` | Step 1-N (测试) | 用于读取 KV cache 进行实际测试 |
| `TEST_DATA_PATH` | Step 1-N (测试) | 测试问题的 JSON 文件路径 |

---

## 使用场景

### 场景 1: 跨数据集（2wikimqa → musique）

**目的**: 测试 steering vectors 的泛化能力

```bash
# 配置
TRAIN_DATASET="2wikimqa"              # 在 2wikimqa 上训练
TEST_DATASET="musique"                # 在 musique 上测试
TEST_DATA_PATH="./data/result_reflect.json"  # musique 的测试数据

# 运行
bash run_fusionrag_steering.sh
```

**输出文件名**: `./kv_stats/2wikimqa_to_musique_steering_manifold.pt`

**输出示例**:
```
==========================================
FusionRAG Rate Sweep
==========================================
...
数据集配置: 跨数据集 (2wikimqa → musique)
  训练数据集: 2wikimqa (用于计算 steering vectors)
  测试数据集: musique (用于实际测试)
测试数据路径: ./data/result_reflect.json
...

==========================================
Step 0: 计算 Steering Vectors
==========================================
训练数据集: 2wikimqa
模型: Qwen2.5-7B-Instruct
输出路径: ./kv_stats/2wikimqa_to_musique_steering_manifold.pt
跨数据集模式: 2wikimqa → musique
...
```

### 场景 2: 同数据集（2wikimqa → 2wikimqa）

**目的**: 在同一数据集上验证方法有效性

```bash
# 配置
TRAIN_DATASET="2wikimqa"
TEST_DATASET="2wikimqa"
TEST_DATA_PATH="./data/2wikimqa_reflect.json"

# 运行
bash run_fusionrag_steering.sh
```

**输出文件名**: `./kv_stats/2wikimqa_steering_manifold.pt`

**输出示例**:
```
数据集: 2wikimqa (同数据集训练和测试)
测试数据路径: ./data/2wikimqa_reflect.json
...

==========================================
Step 0: 计算 Steering Vectors
==========================================
训练数据集: 2wikimqa
模型: Qwen2.5-7B-Instruct
输出路径: ./kv_stats/2wikimqa_steering_manifold.pt
...
```

### 场景 3: 反向跨数据集（musique → 2wikimqa）

**目的**: 测试反向泛化

```bash
# 配置
TRAIN_DATASET="musique"
TEST_DATASET="2wikimqa"
TEST_DATA_PATH="./data/2wikimqa_reflect.json"

# 运行
bash run_fusionrag_steering.sh
```

**输出文件名**: `./kv_stats/musique_to_2wikimqa_steering_manifold.pt`

---

## 文件命名规则

脚本会根据配置**自动生成文件名**：

### 跨数据集模式

```bash
{TRAIN_DATASET}_to_{TEST_DATASET}_steering_{manifold|no_proj}.pt
```

**示例**:
- `2wikimqa_to_musique_steering_manifold.pt`
- `musique_to_2wikimqa_steering_no_proj.pt`

### 同数据集模式

```bash
{TRAIN_DATASET}_steering_{manifold|no_proj}.pt
```

**示例**:
- `2wikimqa_steering_manifold.pt`
- `musique_steering_no_proj.pt`

---

## 完整配置示例

### 示例 1: 跨数据集 + 使用所有样本

```bash
#!/bin/bash

# 数据配置
TRAIN_DATASET="2wikimqa"
TEST_DATASET="musique"
TEST_DATA_PATH="./data/result_reflect.json"

# Steering 配置
COMPUTE_STEERING="true"
USE_ALL_SAMPLES="true"           # 自动扫描 2wikimqa 的所有样本
USE_MANIFOLD_PROJECTION="true"
PCA_VARIANCE_THRESHOLD="0.7"
STEERING_ALPHA="1.0"
STEERING_KEY_LAYERS="all"
STEERING_VALUE_LAYERS="all"

# Rate 列表
RATE_LIST=(0.0 0.3 0.5 1.0)
```

**执行流程**:
1. 扫描 2wikimqa 数据集的所有样本
2. 计算 steering vectors（带 Manifold Projection）
3. 保存到 `./kv_stats/2wikimqa_to_musique_steering_manifold.pt`
4. 在 musique 数据集上测试 4 个 rate 值

### 示例 2: 同数据集 + 指定样本 + 只应用浅层

```bash
# 数据配置
TRAIN_DATASET="2wikimqa"
TEST_DATASET="2wikimqa"
TEST_DATA_PATH="./data/2wikimqa_reflect.json"

# Steering 配置
COMPUTE_STEERING="true"
USE_ALL_SAMPLES="false"
STEERING_SAMPLE_IDS="0 1 2 3 4 5 6 7 8 9"  # 只用 10 个样本
USE_MANIFOLD_PROJECTION="true"
PCA_VARIANCE_THRESHOLD="0.7"
STEERING_ALPHA="1.0"
STEERING_KEY_LAYERS="0-9"        # 只对浅层应用
STEERING_VALUE_LAYERS="0-9"

# Rate 列表
RATE_LIST=(0.3)
```

### 示例 3: 使用已有的 Steering Vectors

```bash
# 数据配置
TRAIN_DATASET="2wikimqa"  # 必须与已有文件匹配
TEST_DATASET="musique"    # 必须与已有文件匹配
TEST_DATA_PATH="./data/result_reflect.json"

# Steering 配置
COMPUTE_STEERING="false"  # 不重新计算
KV_STATS_PATH="./kv_stats/2wikimqa_to_musique_steering_manifold.pt"  # 使用已有文件

STEERING_ALPHA="1.0"
STEERING_KEY_LAYERS="all"
STEERING_VALUE_LAYERS="all"
```

---

## 数据路径说明

### KV Cache 路径结构

脚本会自动构建 KV cache 路径：

```
{CACHE_DIR}/{MODEL_NAME}/{DATASET}/kv_cache/
```

**训练阶段**（Step 0）:
```bash
# TRAIN_DATASET="2wikimqa"
/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/2wikimqa/kv_cache/
/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/2wikimqa/preprocess_kv_cache_global_topk10_bge/
```

**测试阶段**（Step 1-N）:
```bash
# TEST_DATASET="musique"
/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache/
```

### 测试数据 JSON 文件

| 数据集 | 文件路径 | 说明 |
|--------|---------|------|
| 2wikimqa | `./data/2wikimqa_reflect.json` | 2wikimqa 测试问题 |
| musique | `./data/result_reflect.json` | musique 测试问题 |

---

## 常见配置组合

### 组合 1: 标准跨数据集实验

```bash
TRAIN_DATASET="2wikimqa"
TEST_DATASET="musique"
TEST_DATA_PATH="./data/result_reflect.json"
USE_ALL_SAMPLES="true"
```

**用途**: 测试 steering vectors 的跨数据集泛化能力

### 组合 2: 同数据集基线

```bash
TRAIN_DATASET="2wikimqa"
TEST_DATASET="2wikimqa"
TEST_DATA_PATH="./data/2wikimqa_reflect.json"
USE_ALL_SAMPLES="true"
```

**用途**: 建立同数据集性能基线

### 组合 3: 快速验证

```bash
TRAIN_DATASET="2wikimqa"
TEST_DATASET="musique"
TEST_DATA_PATH="./data/result_reflect.json"
USE_ALL_SAMPLES="false"
STEERING_SAMPLE_IDS="0 1 2 3 4"  # 只用 5 个样本
RATE_LIST=(0.3)  # 只测试一个 rate
```

**用途**: 快速验证配置是否正确

---

## 实验设计建议

### 完整对比实验

建议运行以下 4 种配置进行对比：

```bash
# 1. 同数据集 2wikimqa (baseline)
TRAIN_DATASET="2wikimqa"
TEST_DATASET="2wikimqa"
TEST_DATA_PATH="./data/2wikimqa_reflect.json"

# 2. 跨数据集 2wikimqa → musique
TRAIN_DATASET="2wikimqa"
TEST_DATASET="musique"
TEST_DATA_PATH="./data/result_reflect.json"

# 3. 同数据集 musique (baseline)
TRAIN_DATASET="musique"
TEST_DATASET="musique"
TEST_DATA_PATH="./data/result_reflect.json"

# 4. 跨数据集 musique → 2wikimqa
TRAIN_DATASET="musique"
TEST_DATASET="2wikimqa"
TEST_DATA_PATH="./data/2wikimqa_reflect.json"
```

**分析问题**:
- 跨数据集性能下降多少？
- 哪个方向的跨数据集效果更好？
- Steering vectors 学到了什么通用特征？

---

## 注意事项

### 1. 数据集名称必须匹配

确保 `TRAIN_DATASET` 和 `TEST_DATASET` 与实际的 KV cache 目录名称一致：

```bash
# 检查目录是否存在
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/2wikimqa/
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
```

### 2. 测试数据路径必须匹配

确保 `TEST_DATA_PATH` 指向正确的测试问题文件：

```bash
# 检查文件是否存在
ls ./data/2wikimqa_reflect.json
ls ./data/result_reflect.json
```

### 3. 跨数据集的样本选择

当使用 `USE_ALL_SAMPLES="true"` 时，脚本会自动扫描 **训练数据集**（`TRAIN_DATASET`）的所有样本：

```bash
# 会扫描 2wikimqa 的样本
TRAIN_DATASET="2wikimqa"
TEST_DATASET="musique"
USE_ALL_SAMPLES="true"
```

### 4. 文件冲突

如果切换数据集配置，注意删除旧的 steering vectors 文件，避免混淆：

```bash
# 删除旧文件
rm ./kv_stats/2wikimqa_to_musique_steering_manifold.pt

# 或重命名备份
mv ./kv_stats/2wikimqa_to_musique_steering_manifold.pt \
   ./kv_stats/2wikimqa_to_musique_steering_manifold_backup.pt
```

---

## 调试技巧

### 验证配置

在运行前，检查脚本输出的配置信息：

```bash
bash run_fusionrag_steering.sh 2>&1 | head -30
```

查看：
- "数据集配置: 跨数据集 (2wikimqa → musique)"
- "训练数据集: 2wikimqa (用于计算 steering vectors)"
- "测试数据集: musique (用于实际测试)"
- "输出路径: ./kv_stats/2wikimqa_to_musique_steering_manifold.pt"

### 仅运行 Step 0

如果只想计算 steering vectors，可以设置一个很小的 `RATE_LIST`：

```bash
RATE_LIST=(0.3)  # 只测试一个 rate，快速完成
```

或者先注释掉测试部分，只运行 Step 0。

---

## 总结

### 关键改进

✅ **分离训练和测试数据集**: 支持跨数据集实验
✅ **智能文件命名**: 自动反映训练→测试关系
✅ **清晰的日志输出**: 明确显示当前配置
✅ **向后兼容**: 支持同数据集模式

### 典型工作流

1. **设置数据集配置**
   ```bash
   TRAIN_DATASET="2wikimqa"
   TEST_DATASET="musique"
   TEST_DATA_PATH="./data/result_reflect.json"
   ```

2. **运行脚本**
   ```bash
   bash run_fusionrag_steering.sh
   ```

3. **检查输出**
   - Steering vectors: `./kv_stats/2wikimqa_to_musique_steering_manifold.pt`
   - 测试结果: `./result/bge/`

4. **分析结果**
   - 对比跨数据集 vs 同数据集性能
   - 评估泛化能力

---

## 相关文档

- [RUN_FUSIONRAG_STEERING_README.md](./RUN_FUSIONRAG_STEERING_README.md) - 脚本使用指南
- [CROSS_DATASET_EXPERIMENT_README.md](./CROSS_DATASET_EXPERIMENT_README.md) - 跨数据集实验详解
- [NEW_FEATURES.md](./NEW_FEATURES.md) - 新功能说明
