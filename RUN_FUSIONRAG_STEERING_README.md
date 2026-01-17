# run_fusionrag_steering.sh 使用说明

这个脚本集成了 **Steering Vector 计算** 和 **Rate Sweep 测试**，可以一键完成从统计到测试的完整流程。

---

## 功能特性

### 1. 自动计算 Steering Vectors（Step 0）
在开始测试前，自动计算 steering vectors（如果启用）：
- 支持**自动扫描所有样本**或**手动指定样本**
- 支持 **Manifold Projection**（PCA-based）
- 自动检测文件是否已存在，避免重复计算

### 2. Rate Sweep 测试
遍历多个 rate 值进行测试，使用计算好的 steering vectors

---

## 快速开始

### 默认配置运行

```bash
bash run_fusionrag_steering.sh
```

这将：
1. 在 2wikimqa 数据集上使用 20 个样本计算 steering vectors
2. 使用 Manifold Projection (PCA 阈值 0.7)
3. 测试 Rate = [0.0, 0.1, 0.15, 0.3, 0.5, 0.8, 0.9, 1.0]

---

## 配置选项

### 核心配置

```bash
# 数据和模型
DATASET_NAME="2wikimqa"           # 数据集名称
MODEL_NAME="Qwen2.5-7B-Instruct"  # 模型名称
RECALL_METHOD="no_preprocess_with_bias"  # 召回方法

# Rate 列表
RATE_LIST=(0.0 0.1 0.15 0.3 0.5 0.8 0.9 1.0)
```

### Steering Vector 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `COMPUTE_STEERING` | `true` | 是否计算 steering vectors |
| `USE_ALL_SAMPLES` | `false` | 是否自动扫描所有样本 |
| `STEERING_SAMPLE_IDS` | `0 1 2 ... 19` | 手动指定的样本 ID |
| `STEERING_OUTPUT_DIR` | `./kv_stats` | Steering vectors 输出目录 |
| `USE_MANIFOLD_PROJECTION` | `true` | 是否使用 Manifold Projection |
| `PCA_VARIANCE_THRESHOLD` | `0.7` | PCA 方差阈值 |
| `STEERING_ALPHA` | `1.0` | Steering 强度系数 |
| `STEERING_KEY_LAYERS` | `all` | 应用 Key steering 的层 |
| `STEERING_VALUE_LAYERS` | `all` | 应用 Value steering 的层 |

---

## 使用场景

### 场景 1: 使用所有样本 + 所有层

```bash
# 修改配置
vim run_fusionrag_steering.sh

# 设置以下参数:
USE_ALL_SAMPLES="true"
STEERING_KEY_LAYERS="all"
STEERING_VALUE_LAYERS="all"

# 运行
bash run_fusionrag_steering.sh
```

### 场景 2: 指定样本 + 只应用浅层

```bash
# 配置:
USE_ALL_SAMPLES="false"
STEERING_SAMPLE_IDS="0 1 2 3 4 5 6 7 8 9"
STEERING_KEY_LAYERS="0-9"
STEERING_VALUE_LAYERS="0-9"

# 运行
bash run_fusionrag_steering.sh
```

### 场景 3: 禁用 Manifold Projection

```bash
# 配置:
USE_MANIFOLD_PROJECTION="false"

# 运行
bash run_fusionrag_steering.sh
```

### 场景 4: 使用已有的 Steering Vectors

如果已经计算好了 steering vectors，可以直接使用：

```bash
# 配置:
COMPUTE_STEERING="false"
KV_STATS_PATH="./kv_stats/2wikimqa_steering_manifold.pt"

# 运行
bash run_fusionrag_steering.sh
```

### 场景 5: 切换回 BGE 方法

如果想使用传统的 BGE 方法而不是 steering vectors：

```bash
# 配置:
RECALL_METHOD="bge"

# 运行
bash run_fusionrag_steering.sh
```

---

## 执行流程

### Step 0: 计算 Steering Vectors（自动）

当 `RECALL_METHOD="no_preprocess_with_bias"` 且 `COMPUTE_STEERING="true"` 时：

1. 检查 steering vectors 文件是否已存在
2. 如果不存在，调用 `compute_kv_distribution_stats.py` 计算
3. 自动生成文件名：`{DATASET_NAME}_steering_{manifold|no_proj}.pt`
4. 保存到 `STEERING_OUTPUT_DIR` 目录

**输出示例**：
```
==========================================
Step 0: 计算 Steering Vectors
==========================================
数据集: 2wikimqa
模型: Qwen2.5-7B-Instruct
输出路径: ./kv_stats/2wikimqa_steering_manifold.pt

开始计算 steering vectors...
  使用指定样本: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  Manifold projection: 启用 (PCA 阈值: 0.7)

执行命令:
/home/shm/anaconda3/envs/fusionrag/bin/python compute_kv_distribution_stats.py ...

✓ Steering vectors 计算成功！
  保存位置: ./kv_stats/2wikimqa_steering_manifold.pt
==========================================
```

### Step 1-N: Rate Sweep 测试

遍历每个 rate 值，使用计算好的 steering vectors 进行测试：

```
==========================================
开始测试 Rate = 0.3
时间: 2025-01-17 14:30:00
==========================================

运行 test_fusionrag_reflect.py ...

✓ Rate 0.3 测试完成
==========================================
```

---

## 输出文件

### Steering Vectors

**位置**: `./kv_stats/{DATASET_NAME}_steering_{manifold|no_proj}.pt`

**示例**:
- `./kv_stats/2wikimqa_steering_manifold.pt`
- `./kv_stats/2wikimqa_steering_no_proj.pt`

### 测试结果

**位置**: `${RESULT_DIR}/` (默认: `./result/bge/`)

**文件**:
- CSV 文件：包含所有测试结果
- TXT 文件：详细日志
- JSON 文件：元数据

---

## 参数详解

### Steering Alpha (强度系数)

控制 steering vector 的应用强度：`h' = h + α * r_M`

| Alpha 值 | 效果 | 说明 |
|---------|------|------|
| `0.5` | 弱干预 | 保留更多原始信息 |
| `1.0` | 标准 | 论文推荐值 |
| `1.5` | 强干预 | 更激进的修正 |

### 层选择格式

| 格式 | 示例 | 说明 |
|------|------|------|
| `"all"` | `"all"` | 所有层 (0-27) |
| 范围 | `"0-10"` | 层 0 到 10 |
| 列举 | `"0,5,10"` | 特定层 |
| 混合 | `"0-5,20-27"` | 范围 + 列举 |

---

## 常见问题

### Q1: 如何重新计算 Steering Vectors？

**A**: 删除已有的 `.pt` 文件：
```bash
rm ./kv_stats/2wikimqa_steering_manifold.pt
bash run_fusionrag_steering.sh
```

### Q2: 如何查看正在使用哪个 Steering Vectors 文件？

**A**: 脚本开始时会显示：
```
Steering Vector 配置:
  计算 Steering: true
  ...
```

或检查 `KV_STATS_PATH` 变量。

### Q3: 脚本计算 Steering Vectors 时失败怎么办？

**A**:
1. 检查数据集路径是否正确
2. 确认样本 ID 是否存在
3. 查看错误日志

### Q4: 能否只运行计算部分或测试部分？

**A**: 可以：

**只计算 Steering Vectors**:
```bash
# 设置一个很小的 RATE_LIST
RATE_LIST=(0.3)
# 或直接运行 compute_kv_distribution_stats.py
```

**只运行测试（使用已有 Steering Vectors）**:
```bash
COMPUTE_STEERING="false"
KV_STATS_PATH="./kv_stats/existing_file.pt"
```

---

## 与其他脚本的对比

| 脚本 | 用途 | Steering 计算 | Rate Sweep |
|------|------|--------------|-----------|
| `run_fusionrag_steering.sh` | **一体化** | ✅ 自动 | ✅ |
| `run_cross_dataset_experiment.sh` | 跨数据集实验 | ✅ | ❌ (固定 rate) |
| `test_cross_dataset_quick.sh` | 快速测试 | ✅ | ❌ |
| `compute_kv_distribution_stats.py` | 单独计算 | ✅ | ❌ |
| `test_fusionrag_reflect.py` | 单次测试 | ❌ | ❌ |

---

## 最佳实践

### 推荐配置（质量优先）

```bash
USE_ALL_SAMPLES="true"              # 使用所有样本
USE_MANIFOLD_PROJECTION="true"      # 启用 Manifold Projection
PCA_VARIANCE_THRESHOLD="0.7"        # 70% 方差
STEERING_ALPHA="1.0"                # 标准强度
STEERING_KEY_LAYERS="all"           # 所有层
STEERING_VALUE_LAYERS="all"
```

### 推荐配置（速度优先）

```bash
USE_ALL_SAMPLES="false"
STEERING_SAMPLE_IDS="0 1 2 3 4"    # 仅 5 个样本
USE_MANIFOLD_PROJECTION="true"
STEERING_KEY_LAYERS="0-9"          # 仅浅层
STEERING_VALUE_LAYERS="0-9"
RATE_LIST=(0.3 0.5)                # 仅测试 2 个 rate
```

---

## 更新日志

### 2025-01-17
- ✅ 添加 Step 0: 自动计算 Steering Vectors
- ✅ 集成新功能：自动扫描所有样本
- ✅ 集成新功能：选择性层级应用
- ✅ 支持 Steering Alpha 参数
- ✅ 自动检测文件是否已存在
- ✅ 完善日志输出

---

## 相关文档

- [NEW_FEATURES.md](./NEW_FEATURES.md) - 新功能详细说明
- [STEERING_ALPHA_IMPLEMENTATION.md](./STEERING_ALPHA_IMPLEMENTATION.md) - Alpha 参数实现
- [CROSS_DATASET_EXPERIMENT_README.md](./CROSS_DATASET_EXPERIMENT_README.md) - 跨数据集实验
- [UPDATE_SUMMARY.md](./UPDATE_SUMMARY.md) - 最新更新总结
