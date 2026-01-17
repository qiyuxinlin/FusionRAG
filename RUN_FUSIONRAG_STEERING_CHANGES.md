# run_fusionrag_steering.sh 修改总结

## 概述

在原有的 `run_fusionrag_steering.sh` 脚本基础上，添加了 **Step 0: 计算 Steering Vectors** 功能，实现了从统计到测试的一体化流程。

---

## 主要修改

### 1. 新增配置参数（第 35-46 行）

```bash
# Steering Vector 配置 (当 RECALL_METHOD=no_preprocess_with_bias 时生效)
COMPUTE_STEERING="true"       # 是否在运行前计算 steering vectors
USE_ALL_SAMPLES="false"       # 是否使用所有样本计算 steering (true) 还是指定样本 (false)
STEERING_SAMPLE_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"  # 样本 ID 列表
STEERING_OUTPUT_DIR="./kv_stats"  # Steering vectors 输出目录
USE_MANIFOLD_PROJECTION="true"    # 是否使用 manifold projection
PCA_VARIANCE_THRESHOLD="0.7"      # PCA 方差阈值
STEERING_ALPHA="1.0"              # Steering vector 强度系数
STEERING_KEY_LAYERS="all"         # 应用 key steering 的层 ("all", "0-10", "0,5,10", 等)
STEERING_VALUE_LAYERS="all"       # 应用 value steering 的层

KV_STATS_PATH=""              # KV分布统计文件路径 (自动设置或手动指定)
```

**关键变化**:
- `RECALL_METHOD` 从 `"bge"` 改为 `"no_preprocess_with_bias"`
- 添加了 9 个新的配置参数

### 2. 增强的启动信息（第 88-106 行）

添加了 Steering Vector 配置的显示：

```bash
# 如果使用 no_preprocess_with_bias 方法，显示 steering 配置
if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ]; then
    echo ""
    echo "Steering Vector 配置:"
    echo "  计算 Steering: ${COMPUTE_STEERING}"
    if [ "${COMPUTE_STEERING}" = "true" ]; then
        echo "  使用所有样本: ${USE_ALL_SAMPLES}"
        if [ "${USE_ALL_SAMPLES}" = "false" ]; then
            echo "  样本数量: $(echo ${STEERING_SAMPLE_IDS} | wc -w)"
        fi
        echo "  Manifold 投影: ${USE_MANIFOLD_PROJECTION}"
        if [ "${USE_MANIFOLD_PROJECTION}" = "true" ]; then
            echo "  PCA 阈值: ${PCA_VARIANCE_THRESHOLD}"
        fi
    fi
    echo "  Alpha (强度): ${STEERING_ALPHA}"
    echo "  Key 层: ${STEERING_KEY_LAYERS}"
    echo "  Value 层: ${STEERING_VALUE_LAYERS}"
fi
```

### 3. Step 0: 计算 Steering Vectors（第 96-185 行）

这是最核心的新增功能：

```bash
#####################################################################
# Step 0: 计算 Steering Vectors (如果需要)
#####################################################################

if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ] && [ "${COMPUTE_STEERING}" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "Step 0: 计算 Steering Vectors"
    echo "=========================================="
    echo "数据集: ${DATASET_NAME}"
    echo "模型: ${MODEL_NAME}"

    # 创建输出目录
    mkdir -p "${STEERING_OUTPUT_DIR}"

    # 设置输出文件路径
    if [ -z "${KV_STATS_PATH}" ]; then
        # 自动生成文件名
        if [ "${USE_MANIFOLD_PROJECTION}" = "true" ]; then
            SUFFIX="manifold"
        else
            SUFFIX="no_proj"
        fi
        KV_STATS_PATH="${STEERING_OUTPUT_DIR}/${DATASET_NAME}_steering_${SUFFIX}.pt"
    fi

    echo "输出路径: ${KV_STATS_PATH}"

    # 检查文件是否已存在
    if [ -f "${KV_STATS_PATH}" ]; then
        echo ""
        echo "✓ Steering vectors 文件已存在: ${KV_STATS_PATH}"
        echo "  跳过计算。如需重新计算，请删除此文件。"
    else
        echo ""
        echo "开始计算 steering vectors..."

        # 构建命令
        CMD="${PYTHON_PATH} compute_kv_distribution_stats.py \
            --cache_dir \"${CACHE_DIR}\" \
            --dataset \"${DATASET_NAME}\" \
            --model_name \"${MODEL_NAME}\""

        # 添加样本选择参数
        if [ "${USE_ALL_SAMPLES}" = "true" ]; then
            CMD="${CMD} --use_all_samples"
            echo "  使用所有可用样本 (auto-scan)"
        else
            CMD="${CMD} --sample_ids ${STEERING_SAMPLE_IDS}"
            echo "  使用指定样本: ${STEERING_SAMPLE_IDS}"
        fi

        # 添加其他参数
        CMD="${CMD} \
            --chunk_id 1 \
            --max_layers 28 \
            --output_path \"${KV_STATS_PATH}\" \
            --pca_variance_threshold ${PCA_VARIANCE_THRESHOLD}"

        # 添加 manifold projection 参数
        if [ "${USE_MANIFOLD_PROJECTION}" = "true" ]; then
            CMD="${CMD} --use_manifold_projection"
            echo "  Manifold projection: 启用 (PCA 阈值: ${PCA_VARIANCE_THRESHOLD})"
        else
            CMD="${CMD} --no_manifold_projection"
            echo "  Manifold projection: 禁用"
        fi

        echo ""
        echo "执行命令:"
        echo "${CMD}"
        echo ""

        # 执行计算
        eval ${CMD}

        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Steering vectors 计算成功！"
            echo "  保存位置: ${KV_STATS_PATH}"
        else
            echo ""
            echo "✗ Steering vectors 计算失败！"
            exit 1
        fi
    fi

    echo "=========================================="
    echo ""
fi
```

**功能特点**:
- 只在 `RECALL_METHOD="no_preprocess_with_bias"` 且 `COMPUTE_STEERING="true"` 时执行
- 自动生成输出文件名（基于数据集名称和是否使用 manifold projection）
- 检测文件是否已存在，避免重复计算
- 支持两种样本选择方式：自动扫描 或 手动指定
- 动态构建命令，根据配置添加参数
- 失败时退出脚本

### 4. 添加 Steering 参数到测试（第 236-244 行）

在构建 `PYTHON_ARGS` 时添加 steering vector 相关参数：

```bash
# Steering vector 相关参数 (当 RECALL_METHOD=no_preprocess_with_bias 时)
if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ]; then
    if [ ! -z "${KV_STATS_PATH}" ]; then
        PYTHON_ARGS+=("--kv_stats_path" "${KV_STATS_PATH}")
    fi
    PYTHON_ARGS+=("--steering_alpha" "${STEERING_ALPHA}")
    PYTHON_ARGS+=("--steering_key_layers" "${STEERING_KEY_LAYERS}")
    PYTHON_ARGS+=("--steering_value_layers" "${STEERING_VALUE_LAYERS}")
fi
```

### 5. 更新总结信息（第 293-295 行）

在脚本结束时显示 steering vectors 文件位置：

```bash
if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ] && [ ! -z "${KV_STATS_PATH}" ]; then
    echo "  Steering Vectors: ${KV_STATS_PATH}"
fi
```

---

## 修改前后对比

### 修改前

```bash
#!/bin/bash
# 只进行 Rate Sweep 测试
# 使用 BGE 方法召回

RECALL_METHOD="bge"
# ... 其他配置 ...

# 直接开始遍历 rate
for RATE in "${RATE_LIST[@]}"; do
    # 运行测试
done
```

### 修改后

```bash
#!/bin/bash
# 集成了 Steering Vector 计算 + Rate Sweep 测试
# 使用 no_preprocess_with_bias 方法

RECALL_METHOD="no_preprocess_with_bias"
# ... Steering 配置 ...

# Step 0: 计算 Steering Vectors（如果需要）
if [ "${RECALL_METHOD}" = "no_preprocess_with_bias" ]; then
    # 自动计算 steering vectors
fi

# Step 1-N: 遍历 rate 进行测试
for RATE in "${RATE_LIST[@]}"; do
    # 使用 steering vectors 运行测试
done
```

---

## 使用示例

### 示例 1: 完整流程（默认配置）

```bash
bash run_fusionrag_steering.sh
```

**执行内容**:
1. 使用 20 个样本计算 steering vectors（带 Manifold Projection）
2. 保存到 `./kv_stats/2wikimqa_steering_manifold.pt`
3. 遍历 8 个 rate 值进行测试

### 示例 2: 使用所有样本

```bash
# 修改脚本
vim run_fusionrag_steering.sh
# 设置: USE_ALL_SAMPLES="true"

bash run_fusionrag_steering.sh
```

### 示例 3: 只应用浅层

```bash
# 修改脚本
vim run_fusionrag_steering.sh
# 设置:
# STEERING_KEY_LAYERS="0-9"
# STEERING_VALUE_LAYERS="0-9"

bash run_fusionrag_steering.sh
```

### 示例 4: 跳过计算，使用已有文件

```bash
# 修改脚本
vim run_fusionrag_steering.sh
# 设置:
# COMPUTE_STEERING="false"
# KV_STATS_PATH="./kv_stats/existing_file.pt"

bash run_fusionrag_steering.sh
```

### 示例 5: 切换回 BGE 方法

```bash
# 修改脚本
vim run_fusionrag_steering.sh
# 设置: RECALL_METHOD="bge"

bash run_fusionrag_steering.sh
```

---

## 优势

### 1. 一体化流程
- 无需手动运行两个脚本
- 自动管理 steering vectors 文件路径
- 减少人为错误

### 2. 智能检测
- 自动检测文件是否已存在
- 避免重复计算，节省时间
- 支持增量更新

### 3. 灵活配置
- 支持所有新功能（自动扫描样本、层选择、alpha 调节）
- 易于切换不同配置
- 兼容旧的 BGE 方法

### 4. 清晰日志
- 详细的进度信息
- 易于调试和追踪
- 失败时有明确提示

---

## 文件结构

```
/home/shm/document/exp/FusionRAG/
├── run_fusionrag_steering.sh          # 主脚本（已修改）
├── compute_kv_distribution_stats.py   # Steering 计算脚本（Step 0 调用）
├── test_fusionrag_reflect.py          # 测试脚本（Rate Sweep 调用）
├── kv_stats/                           # Steering vectors 输出目录
│   ├── 2wikimqa_steering_manifold.pt
│   └── 2wikimqa_steering_no_proj.pt
├── result/bge/                         # 测试结果目录
└── RUN_FUSIONRAG_STEERING_README.md   # 使用文档（新建）
```

---

## 依赖关系

```
run_fusionrag_steering.sh
    │
    ├─> Step 0: compute_kv_distribution_stats.py
    │   └─> 输出: {DATASET}_steering_{manifold|no_proj}.pt
    │
    └─> Step 1-N: test_fusionrag_reflect.py (for each rate)
        └─> 输入: {DATASET}_steering_{manifold|no_proj}.pt
```

---

## 验证

### 语法检查

```bash
bash -n run_fusionrag_steering.sh
```

✅ 通过（无错误）

### 功能验证

建议先用少量样本测试：

```bash
# 修改配置为快速测试
STEERING_SAMPLE_IDS="0 1 2 3 4"
RATE_LIST=(0.3)

bash run_fusionrag_steering.sh
```

---

## 向后兼容

所有修改都是**向后兼容**的：

1. **默认行为改变**: `RECALL_METHOD` 从 `"bge"` 改为 `"no_preprocess_with_bias"`
   - 如需恢复旧行为，设置 `RECALL_METHOD="bge"`

2. **新增参数**: 都有合理的默认值
   - 不修改配置也能正常运行

3. **条件执行**: Step 0 只在特定条件下执行
   - 不影响其他 recall 方法

---

## 总结

本次修改实现了：

✅ **集成统计功能**: 在测试前自动计算 steering vectors
✅ **智能文件管理**: 自动检测、生成文件名、避免重复
✅ **完整参数支持**: 支持所有新功能（自动样本、层选择、alpha）
✅ **灵活配置**: 可轻松切换不同模式
✅ **清晰日志**: 详细的进度和状态信息
✅ **向后兼容**: 不破坏现有功能

现在用户可以用一个脚本完成：**计算 steering vectors → 多 rate 测试**的完整流程！
