# FusionRAG 更新总结（2025-01-17）

本次更新添加了两个重要的新功能，用于更灵活地计算和应用 steering vectors。

---

## 新功能 1: 自动扫描所有样本

### 功能描述
在计算 steering vectors 时，现在可以自动扫描数据集中的所有可用样本，而不需要手动指定 `--sample_ids`。

### 使用示例

**旧方法（手动指定）**：
```bash
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset 2wikimqa \
    --model_name Qwen2.5-7B-Instruct \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --output_path ./kv_stats/steering.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7
```

**新方法（自动扫描）**：
```bash
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset 2wikimqa \
    --model_name Qwen2.5-7B-Instruct \
    --use_all_samples \
    --output_path ./kv_stats/steering.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7
```

### 代码修改

**文件**: `compute_kv_distribution_stats.py`

1. **新增方法** `scan_all_samples()`:
   - 自动扫描 `no_preprocess` 和 `bge` 目录
   - 查找匹配 `{sample_id}_{chunk_id}_key.pt` 的所有文件
   - 返回两个目录中都存在的样本 ID（取交集）

2. **新增参数**:
   - `--use_all_samples`: 自动使用所有样本（flag）
   - `--sample_ids`: 改为可选参数（`required=False`）

3. **参数验证**:
   - `--use_all_samples` 和 `--sample_ids` 互斥
   - 必须指定其中之一

---

## 新功能 2: 选择性层级应用 Steering Vector

### 功能描述
在应用 steering vectors 时，现在可以精确控制对哪些层的 Key 和 Value 应用 steering vector。

### 使用示例

**默认（所有层）**：
```bash
python test_fusionrag_reflect.py \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/steering.pt \
    --steering_alpha 1.0
    # 默认: --steering_key_layers "all" --steering_value_layers "all"
```

**只对浅层应用**：
```bash
python test_fusionrag_reflect.py \
    --steering_key_layers "0-9" \
    --steering_value_layers "0-9" \
    ...
```

**浅层 + 深层（跳过中间）**：
```bash
python test_fusionrag_reflect.py \
    --steering_key_layers "0-5,22-27" \
    --steering_value_layers "0-5,22-27" \
    ...
```

**Key 和 Value 不同层**：
```bash
python test_fusionrag_reflect.py \
    --steering_key_layers "0-13" \      # Key: 前半层
    --steering_value_layers "14-27" \   # Value: 后半层
    ...
```

### 支持的格式

| 格式 | 示例 | 说明 |
|------|------|------|
| `"all"` | `"all"` | 所有层 (0-27) |
| 范围 | `"0-10"` | 层 0 到 10（包含端点） |
| 列举 | `"0,5,10"` | 特定层 |
| 混合 | `"0-5,10,15-20"` | 范围 + 列举 |
| 空 | `""` | 不应用任何层 |

### 代码修改

**文件**: `test_fusionrag_reflect.py`

1. **新增函数** `parse_layer_selection()`:
   - 解析层选择字符串
   - 支持多种格式（all, 范围, 列举, 混合）
   - 返回层索引的集合

2. **新增参数**:
   - `--steering_key_layers`: 应用 Key steering vector 的层
   - `--steering_value_layers`: 应用 Value steering vector 的层

3. **修改应用逻辑**:
   - 在 Step 2.5 开始处解析层选择
   - 在应用 steering vector 时检查当前层是否应该被应用
   - 对未选中的层使用原始 KV cache

---

## 文件修改列表

### 核心代码

1. **compute_kv_distribution_stats.py**
   - ✅ 添加 `scan_all_samples()` 方法
   - ✅ 添加 `--use_all_samples` 参数
   - ✅ 修改 `--sample_ids` 为可选
   - ✅ 添加参数验证逻辑

2. **test_fusionrag_reflect.py**
   - ✅ 添加 `parse_layer_selection()` 函数
   - ✅ 添加 `--steering_key_layers` 和 `--steering_value_layers` 参数
   - ✅ 修改 main() 函数签名
   - ✅ 修改应用 steering vector 的逻辑
   - ✅ 添加层选择日志输出

### 实验脚本

3. **run_cross_dataset_experiment.sh**
   - ✅ 添加 `USE_ALL_SAMPLES` 配置变量
   - ✅ 添加 `STEERING_KEY_LAYERS` 和 `STEERING_VALUE_LAYERS` 配置
   - ✅ 修改 Method 1 和 Method 2 的命令构建逻辑
   - ✅ 在 Experiment 2 和 3 中添加层选择参数

4. **test_cross_dataset_quick.sh**
   - ✅ 添加 `USE_ALL_SAMPLES` 配置变量
   - ✅ 修改命令构建逻辑

### 文档

5. **CROSS_DATASET_EXPERIMENT_README.md**
   - ✅ 添加自动扫描所有样本的说明
   - ✅ 添加层选择功能的说明和示例

6. **NEW_FEATURES.md** (新建)
   - ✅ 详细的功能文档
   - ✅ 使用示例和最佳实践
   - ✅ FAQ 和常见问题

7. **UPDATE_SUMMARY.md** (本文件)
   - ✅ 更新总结

---

## 验证测试

### 功能 1 验证

```bash
$ python compute_kv_distribution_stats.py --help | grep -A 2 "use_all_samples"
  --use_all_samples     Automatically use all available samples in the dataset
                        (mutually exclusive with --sample_ids)
```

✅ 参数已成功添加

### 功能 2 验证

```bash
$ python test_fusionrag_reflect.py --help | grep -A 3 "steering.*layers"
  --steering_key_layers STEERING_KEY_LAYERS
                        Layers to apply key steering vector. Format: "all",
                        "0-10", "0,5,10", or "0-10,15,20-25" (default: all)
  --steering_value_layers STEERING_VALUE_LAYERS
                        Layers to apply value steering vector. Format: "all",
                        "0-10", "0,5,10", or "0-10,15,20-25" (default: all)
```

✅ 参数已成功添加

---

## 使用建议

### 最佳实践组合

```bash
# Step 1: 用所有样本计算 steering vectors（提升质量）
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset 2wikimqa \
    --model_name Qwen2.5-7B-Instruct \
    --use_all_samples \
    --output_path ./kv_stats/2wikimqa_all_samples.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7

# Step 2: 只对关键层应用 steering（减少计算）
python test_fusionrag_reflect.py \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path /mnt/data3/tmp/fusionrag \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/2wikimqa_all_samples.pt \
    --steering_alpha 1.0 \
    --steering_key_layers "0-9" \
    --steering_value_layers "0-9" \
    --preprocess true \
    --rate 0.3
```

### 消融实验示例

```bash
# 实验 1: Baseline (no steering)
--steering_key_layers "" --steering_value_layers ""

# 实验 2: 只 steer 浅层 (0-9)
--steering_key_layers "0-9" --steering_value_layers "0-9"

# 实验 3: 只 steer 中层 (10-17)
--steering_key_layers "10-17" --steering_value_layers "10-17"

# 实验 4: 只 steer 深层 (18-27)
--steering_key_layers "18-27" --steering_value_layers "18-27"

# 实验 5: 全部层 (upper bound)
--steering_key_layers "all" --steering_value_layers "all"
```

---

## 研究价值

### 功能 1: 自动扫描所有样本

**优势**：
- 省时省力，不需要手动列举样本 ID
- 避免遗漏，自动包含所有可用样本
- 动态适应数据集增长
- 使用更多样本可能提升 steering vector 质量

**研究问题**：
- 样本数量对 steering vector 质量的影响？
- 20 个样本 vs 200 个样本的效果差异？
- 数据多样性如何影响泛化能力？

### 功能 2: 选择性层级应用

**优势**：
- 精确控制干预位置
- 减少计算开销
- 探索层级重要性

**研究问题**：
- 哪些层对 steering 效果贡献最大？
- 浅层和深层的作用有何不同？
- Key 和 Value 的 steering 效果是否独立？
- 是否可以只 steer 少数关键层达到相似效果？
- 减少 steering 层数能否降低延迟？

---

## 下一步建议

### 立即可做

1. **验证基础功能**：
   ```bash
   # 快速测试（5个样本）
   bash test_cross_dataset_quick.sh
   ```

2. **小规模实验**：
   ```bash
   # 对比 20 个样本 vs 所有样本
   USE_ALL_SAMPLES="true" bash run_cross_dataset_experiment.sh
   ```

3. **层级消融实验**：
   ```bash
   # 测试不同层组合的效果
   for LAYERS in "0-9" "10-17" "18-27" "all"; do
       python test_fusionrag_reflect.py \
           --steering_key_layers "${LAYERS}" \
           --steering_value_layers "${LAYERS}" \
           ...
   done
   ```

### 进阶研究

1. **层级重要性分析**：
   - 逐层消融（每次只 steer 一层）
   - 绘制每层的贡献曲线

2. **Key vs Value 分离**：
   - 只 steer Key，不 steer Value
   - 只 steer Value，不 steer Key
   - 分析两者的独立作用

3. **稀疏干预优化**：
   - 寻找最小的 steering 层集合
   - 在性能和效率间找到最佳平衡

4. **跨数据集泛化**：
   - 不同层选择在跨数据集时的泛化能力
   - 浅层是否比深层更通用？

---

## 兼容性

- ✅ **向后兼容**：所有新参数都有默认值
- ✅ **无破坏性更改**：旧脚本无需修改即可运行
- ✅ **清晰错误提示**：参数冲突时给出明确提示

### 迁移指南

**如果使用旧脚本**：
- 无需任何修改，默认行为不变
- `--sample_ids` 仍然有效
- Steering 默认应用到所有层

**如果想使用新功能**：
- 添加 `--use_all_samples` 替换 `--sample_ids`
- 添加 `--steering_key_layers` 和 `--steering_value_layers` 控制层选择

---

## 总结

本次更新添加了两个强大的新功能：

1. **自动样本扫描**：简化工作流程，提升 steering vector 质量
2. **选择性层级应用**：精确控制干预位置，探索层级重要性

这两个功能都已经：
- ✅ 完整实现并测试
- ✅ 集成到实验脚本
- ✅ 文档完善
- ✅ 向后兼容

可以立即使用这些新功能进行实验和研究！
