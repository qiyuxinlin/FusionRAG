# 新功能文档

本文档介绍了两个新增的高级功能，用于更灵活地计算和应用 steering vectors。

---

## 功能 1: 自动扫描所有样本

### 功能描述

在计算 steering vectors 时，现在可以自动扫描数据集中的**所有可用样本**，而不需要手动指定 `--sample_ids`。

### 使用方法

#### 旧方法（手动指定样本）

```bash
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset 2wikimqa \
    --model_name Qwen2.5-7B-Instruct \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --output_path ./kv_stats/2wikimqa_steering.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7
```

#### 新方法（自动扫描所有样本）

```bash
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset 2wikimqa \
    --model_name Qwen2.5-7B-Instruct \
    --use_all_samples \
    --output_path ./kv_stats/2wikimqa_steering_all.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `--sample_ids` | int list | 手动指定样本 ID 列表（与 `--use_all_samples` 互斥） |
| `--use_all_samples` | flag | 自动扫描并使用所有可用样本（与 `--sample_ids` 互斥） |

### 工作原理

1. **扫描目录**：自动扫描 `no_preprocess` 和 `bge` 两个目录
2. **查找样本**：找到匹配 `{sample_id}_{chunk_id}_key.pt` 格式的所有文件
3. **取交集**：只使用在两个目录中都存在的样本（确保配对）
4. **自动排序**：按样本 ID 升序排列

### 优势

- **省时省力**：不需要手动列举样本 ID
- **避免遗漏**：自动包含所有可用样本
- **动态适应**：数据集增加样本时无需修改命令
- **更多数据**：使用全部样本可能提升 steering vector 质量

### 输出示例

```
============================================================
Auto-scanning for all available samples...
============================================================
Method 'no_preprocess': /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/2wikimqa/kv_cache
Method 'bge': /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/2wikimqa/preprocess_kv_cache_global_topk10_bge
Found 200 samples with chunk_id=1
Sample IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]

Using 200 samples to compute steering vectors...
```

---

## 功能 2: 选择性层级应用 Steering Vector

### 功能描述

在应用 steering vectors 时，现在可以**精确控制**对哪些层的 Key 和 Value 应用 steering vector，而不是对所有层一刀切。

### 使用方法

#### 默认（所有层应用）

```bash
python test_fusionrag_reflect.py \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/steering.pt \
    --steering_alpha 1.0 \
    # 默认: --steering_key_layers "all" --steering_value_layers "all"
    ...
```

#### 示例 1: 只对浅层应用

```bash
python test_fusionrag_reflect.py \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/steering.pt \
    --steering_alpha 1.0 \
    --steering_key_layers "0-9" \
    --steering_value_layers "0-9" \
    ...
```

#### 示例 2: 只对深层应用

```bash
python test_fusionrag_reflect.py \
    --steering_key_layers "18-27" \
    --steering_value_layers "18-27" \
    ...
```

#### 示例 3: 浅层 + 深层（跳过中间层）

```bash
python test_fusionrag_reflect.py \
    --steering_key_layers "0-5,22-27" \
    --steering_value_layers "0-5,22-27" \
    ...
```

#### 示例 4: 只对 Key 应用，Value 保持原样

```bash
python test_fusionrag_reflect.py \
    --steering_key_layers "all" \
    --steering_value_layers "" \
    ...
```

#### 示例 5: Key 和 Value 不同层

```bash
python test_fusionrag_reflect.py \
    --steering_key_layers "0-13" \      # Key: 前半层
    --steering_value_layers "14-27" \   # Value: 后半层
    ...
```

#### 示例 6: 特定层（如每隔 5 层）

```bash
python test_fusionrag_reflect.py \
    --steering_key_layers "0,5,10,15,20,25" \
    --steering_value_layers "0,5,10,15,20,25" \
    ...
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--steering_key_layers` | str | `"all"` | 应用 Key steering vector 的层 |
| `--steering_value_layers` | str | `"all"` | 应用 Value steering vector 的层 |

### 格式规范

| 格式 | 示例 | 说明 |
|------|------|------|
| `"all"` | `"all"` | 所有层 (0-27，共 28 层) |
| 范围 | `"0-10"` | 层 0 到 10（包含端点，共 11 层） |
| 列举 | `"0,5,10"` | 特定的层 0, 5, 10 |
| 混合 | `"0-5,10,15-20"` | 层 0-5, 层 10, 层 15-20 |
| 空 | `""` 或 `"none"` | 不应用任何层 |

### 输出示例

```
================================================================================
Applying steering vectors to no_preprocess KV caches for example 1
Method: Manifold Steering (from overthinking paper)
================================================================================

Layer selection:
  Key steering layers: 0-9 -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Value steering layers: 14-27 -> [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

  Using manifold steering vectors (α=1.0)
  Processing chunk 1/5...
```

### 研究价值

通过选择性应用 steering vector，可以探索：

1. **层级重要性**：哪些层对 steering 效果贡献最大？
2. **浅层 vs 深层**：浅层和深层的作用有何不同？
3. **Key vs Value**：Key 和 Value 的 steering 效果是否独立？
4. **稀疏干预**：是否可以只 steer 少数关键层达到相似效果？
5. **计算效率**：减少 steering 层数能否降低延迟？

### 消融实验示例

```bash
# 实验 1: Baseline (no steering)
python test_fusionrag_reflect.py \
    --steering_key_layers "" \
    --steering_value_layers "" \
    ...

# 实验 2: 只 steer 浅层 (0-9)
python test_fusionrag_reflect.py \
    --steering_key_layers "0-9" \
    --steering_value_layers "0-9" \
    ...

# 实验 3: 只 steer 中层 (10-17)
python test_fusionrag_reflect.py \
    --steering_key_layers "10-17" \
    --steering_value_layers "10-17" \
    ...

# 实验 4: 只 steer 深层 (18-27)
python test_fusionrag_reflect.py \
    --steering_key_layers "18-27" \
    --steering_value_layers "18-27" \
    ...

# 实验 5: 全部层 (upper bound)
python test_fusionrag_reflect.py \
    --steering_key_layers "all" \
    --steering_value_layers "all" \
    ...
```

---

## 组合使用示例

### 最佳实践：使用所有样本 + 选择性层级应用

```bash
# Step 1: 用所有样本计算 steering vectors
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset 2wikimqa \
    --model_name Qwen2.5-7B-Instruct \
    --use_all_samples \
    --output_path ./kv_stats/2wikimqa_all_samples.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7

# Step 2: 只对浅层应用 steering（测试假设：浅层更重要）
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

---

## 技术细节

### 功能 1 实现

**核心函数** (`compute_kv_distribution_stats.py`):

```python
def scan_all_samples(self, chunk_id: int = 1) -> List[int]:
    """自动扫描所有可用样本"""
    no_prep_path = self.method_paths['no_preprocess']
    bge_path = self.method_paths['bge']

    no_prep_samples = set()
    bge_samples = set()

    # 扫描 no_preprocess 目录
    for key_file in no_prep_path.glob(f"*_{chunk_id}_key.pt"):
        sample_id = int(key_file.stem.split('_')[0])
        value_file = no_prep_path / f"{sample_id}_{chunk_id}_value.pt"
        if value_file.exists():
            no_prep_samples.add(sample_id)

    # 扫描 bge 目录
    for key_file in bge_path.glob(f"*_{chunk_id}_key.pt"):
        sample_id = int(key_file.stem.split('_')[0])
        value_file = bge_path / f"{sample_id}_{chunk_id}_value.pt"
        if value_file.exists():
            bge_samples.add(sample_id)

    # 返回交集（都存在的样本）
    return sorted(no_prep_samples & bge_samples)
```

### 功能 2 实现

**核心函数** (`test_fusionrag_reflect.py`):

```python
def parse_layer_selection(layer_spec: str, max_layers: int = 28) -> set:
    """解析层选择规范"""
    if layer_spec.lower() == "all":
        return set(range(max_layers))

    layers = set()
    parts = layer_spec.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # 范围格式: "0-10"
            start, end = part.split('-')
            layers.update(range(int(start), int(end) + 1))
        else:
            # 单层: "5"
            layers.add(int(part))

    return layers
```

**应用逻辑**:

```python
# 解析层选择
key_layers_to_apply = parse_layer_selection(steering_key_layers)
value_layers_to_apply = parse_layer_selection(steering_value_layers)

# 应用时检查
for layer_idx in range(num_layers):
    if layer_idx in key_layers_to_apply:
        # 应用 key steering
        steered_key = apply_steering_vector(kv_cache, steering_vec, ...)
    else:
        # 使用原始 key
        steered_key = kv_cache
```

---

## 常见问题 (FAQ)

### Q1: `--use_all_samples` 和 `--sample_ids` 能同时使用吗？

**A**: 不能。两个参数互斥。如果同时指定会报错：
```
error: --use_all_samples and --sample_ids are mutually exclusive. Please use only one.
```

### Q2: 如果 no_preprocess 和 bge 目录中样本不一致怎么办？

**A**: 程序会自动取交集，只使用两个目录中都存在的样本。例如：
- no_preprocess: [0, 1, 2, 3, 4]
- bge: [0, 2, 4, 6, 8]
- **实际使用**: [0, 2, 4]

### Q3: 层选择支持负数索引吗（如 -1 表示最后一层）？

**A**: 目前不支持。请使用显式索引，如 `27` 表示最后一层（Qwen2.5-7B 共 28 层）。

### Q4: 能否只应用 steering 到一半的 Key 和一半的 Value？

**A**: 可以！Key 和 Value 的层选择是独立的：
```bash
--steering_key_layers "0-13" \     # Key: 前半
--steering_value_layers "14-27"    # Value: 后半
```

### Q5: 空字符串 `""` 和 `"none"` 有什么区别？

**A**: 效果相同，都表示不应用任何层。推荐使用 `""` 以避免歧义。

### Q6: 使用所有样本会不会太慢？

**A**: 取决于数据集大小。建议：
- **小数据集** (<100 样本): 直接用 `--use_all_samples`
- **大数据集** (>500 样本): 先用少量样本测试，确认效果后再用全部
- **超大数据集**: 考虑分批计算或采样

---

## 更新日志

### 2025-01-17

**功能 1: 自动扫描所有样本**
- ✅ 添加 `--use_all_samples` 参数
- ✅ 实现 `scan_all_samples()` 方法
- ✅ 添加参数互斥验证
- ✅ 更新文档和示例

**功能 2: 选择性层级应用**
- ✅ 添加 `--steering_key_layers` 和 `--steering_value_layers` 参数
- ✅ 实现 `parse_layer_selection()` 解析函数
- ✅ 支持多种格式 (all, 范围, 列举, 混合)
- ✅ 更新应用逻辑支持层选择
- ✅ 添加详细日志输出
- ✅ 更新文档和消融实验示例

---

## 相关文档

- [CROSS_DATASET_EXPERIMENT_README.md](./CROSS_DATASET_EXPERIMENT_README.md) - 跨数据集实验指南
- [STEERING_ALPHA_IMPLEMENTATION.md](./STEERING_ALPHA_IMPLEMENTATION.md) - Alpha 参数实现
- [MANIFOLD_STEERING_METHOD.md](./MANIFOLD_STEERING_METHOD.md) - Manifold Steering 方法详解
