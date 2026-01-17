# No Preprocess with Distribution Bias - 设计说明

## 设计原则

**核心思想**：避免 preprocess 阶段的文档召回和融合开销，通过动态应用分布偏置来对齐 KV cache 分布。

## 与其他方法的对比

### BGE 召回方法（标准 FusionRAG）

```
PREPROCESS="true"
RECALL_METHOD="bge"

Workflow:
1. Step 1: 生成 no_preprocess KV cache
2. Step 2: Preprocess
   - 使用 BGE 计算文档相似度
   - 召回 top-k 相似文档
   - 融合相似文档的 KV cache
   - 保存 preprocessed KV cache
3. Step 3: 使用 preprocessed KV cache 生成答案
```

**开销**：
- BGE 相似度计算（CPU/GPU）
- 文档召回和排序
- KV cache 融合（内存和计算）
- 额外的存储空间

### No Preprocess 方法（Baseline）

```
PREPROCESS="false"

Workflow:
1. Step 1: 生成 no_preprocess KV cache
2. Step 2: 跳过（无 preprocess）
3. Step 3: 使用 no_preprocess KV cache 生成答案
```

**问题**：
- 性能较差（与 BGE 方法相比）
- KV cache 分布与 BGE 不同

### No Preprocess with Bias 方法（新方法）

```
PREPROCESS="true"
RECALL_METHOD="no_preprocess_with_bias"
KV_STATS_PATH="./kv_stats/stats.pt"

Workflow:
1. Step 1: 生成 no_preprocess KV cache（与 baseline 相同）
2. Step 2: 跳过召回和融合（关键优化）
3. Step 2.5: 动态应用分布偏置（lazy evaluation）
   - 只在需要回答问题时应用
   - 加载 no_preprocess KV cache
   - 应用预计算的 scale 和 bias
   - 临时保存 biased KV cache
4. Step 3: 使用 biased KV cache 生成答案
```

**优势**：
- ✅ 无需 BGE 相似度计算
- ✅ 无需文档召回和排序
- ✅ 无需 KV cache 融合
- ✅ 只需简单的线性变换
- ✅ Lazy evaluation，节省不必要的计算
- ✅ 临时存储，使用后可清理

## 实现细节

### 代码修改位置

#### 1. test_fusionrag_reflect.py

**添加 Enum**：
```python
class RecallMethod(Enum):
    ...
    NO_PREPROCESS_WITH_BIAS = "no_preprocess_with_bias"
```

**跳过 Step 2**（第 1750 行）：
```python
# Step 2: FusionRAG preprocess (if enabled, but skip for NO_PREPROCESS_WITH_BIAS)
if preprocess and rate != 1 and recall_method_enum != RecallMethod.NO_PREPROCESS_WITH_BIAS:
    # 文档召回和融合逻辑（对 NO_PREPROCESS_WITH_BIAS 跳过）
    ...
```

**添加 Step 2.5**（第 2028 行）：
```python
# Step 2.5: Apply distribution bias for NO_PREPROCESS_WITH_BIAS method
if preprocess and recall_method_enum == RecallMethod.NO_PREPROCESS_WITH_BIAS:
    # 1. 复制 system cache
    # 2. 对每个文档 chunk：
    #    - 加载 no_preprocess KV cache
    #    - 应用 scale 和 bias
    #    - 临时保存到 preprocess_save_path
    ...
```

**添加参数支持**：
- `--kv_stats_path`: 统计文件路径
- 函数参数 `kv_stats_path`

#### 2. run_fusionrag_sweep.sh

**添加配置变量**：
```bash
RECALL_METHOD="no_preprocess_with_bias"
KV_STATS_PATH="./kv_stats/stats.pt"
```

**传递参数**：
```bash
if [ ! -z "${KV_STATS_PATH}" ]; then
    PYTHON_ARGS+=("--kv_stats_path" "${KV_STATS_PATH}")
fi
```

### 执行流程

#### Offline 阶段（一次性）

```bash
# 计算分布统计量
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --model_name Qwen2.5-7B-Instruct \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 \
    --output_path ./kv_stats/musique_stats.pt
```

**输出**：
- `musique_stats.pt`: 包含每层的 scale 和 bias
- `musique_stats.json`: 元数据

#### Online 阶段（每次运行）

```bash
# 运行测试
python test_fusionrag_reflect.py \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/musique_stats.pt \
    --preprocess true \
    --rate 0.3
```

**执行过程**：
1. **首次调用（Example 0）**：
   - Step 1: 生成 no_preprocess KV cache
   - Step 2: 跳过（因为是 NO_PREPROCESS_WITH_BIAS）
   - Step 2.5: 应用偏置到 Example 0 的所有 chunks
   - Step 3: 回答 Example 0 的所有子问题

2. **后续调用（Example 1, 2, ...）**：
   - 重复上述流程
   - 每个 example 独立处理

### 临时存储管理

**存储位置**：
```
/mnt/data3/tmp/fusionrag/
├── Qwen2.5-7B-Instruct/
│   └── musique/
│       ├── kv_cache/                                          # no_preprocess KV cache（持久）
│       │   ├── 0_0_key.pt, 0_0_value.pt                      # example 0, chunk 0 (system)
│       │   ├── 0_1_key.pt, 0_1_value.pt                      # example 0, chunk 1 (doc 1)
│       │   └── ...
│       └── preprocess_kv_cache_global_topk10_no_preprocess_with_bias/  # biased KV cache（临时）
│           └── (空目录，文件用完后自动删除)
```

**自动清理机制**（test_fusionrag_reflect.py:2398-2424）：
```python
# 在回答完一个 example 的所有子问题后立即清理
if preprocess and recall_method_enum == RecallMethod.NO_PREPROCESS_WITH_BIAS:
    # 1. 删除 system cache 副本
    os.remove(f"{preprocess_save_path}/{example_id}_0_key.pt")
    os.remove(f"{preprocess_save_path}/{example_id}_0_value.pt")

    # 2. 删除所有文档的 biased KV cache
    for chunk_id in doc_chunks:
        os.remove(f"{preprocess_save_path}/{example_id}_{chunk_id}_key.pt")
        os.remove(f"{preprocess_save_path}/{example_id}_{chunk_id}_value.pt")
```

**清理时机**：
- ✅ 每个 example 回答完所有子问题后立即清理
- ✅ 不占用持久存储空间
- ✅ 不会误用旧的临时文件
- ✅ 内存友好：只保留当前正在使用的 example 的 biased KV cache

## 性能预期

### 时间开销

| 方法 | Step 2 开销 | Step 2.5 开销 | 总开销 |
|------|------------|--------------|--------|
| BGE 召回 | BGE 计算 + 召回 + 融合 | - | 高 |
| No Preprocess | - | - | 低 |
| **No Preprocess + Bias** | **跳过** | **线性变换（很快）** | **低-中** |

### 空间开销

| 方法 | 额外存储 |
|------|----------|
| BGE 召回 | 持久保存 preprocessed KV cache |
| No Preprocess | 无 |
| **No Preprocess + Bias** | **临时保存 biased KV cache** |

### 准确率预期

- No Preprocess（baseline）：最低
- **No Preprocess + Bias**：中等（取决于统计质量）
- BGE 召回（upper bound）：最高

## 调试和验证

### 验证偏置应用

```python
# 检查统计文件
import torch
stats = torch.load('./kv_stats/musique_stats.pt')
print(f"Layers: {len(stats['key_stats'])}")
print(f"Samples: {stats['metadata']['num_samples']}")

# 检查某一层的 scale 和 bias
layer_0_key_bias = stats['key_stats']['layer_0']['bias']
layer_0_key_scale = stats['key_stats']['layer_0']['scale']
print(f"Key bias shape: {layer_0_key_bias.shape}")
print(f"Key scale shape: {layer_0_key_scale.shape}")
```

### 验证 KV cache 分布

```python
# 比较应用偏置前后的分布
no_prep_key = torch.load('kv_cache/0_1_key.pt')
biased_key = torch.load('preprocess_kv_cache_global_topk10_no_preprocess_with_bias/0_1_key.pt')

print(f"No preprocess mean: {no_prep_key[0].mean()}")
print(f"Biased mean: {biased_key[0].mean()}")
print(f"Expected BGE mean: {stats['key_stats']['layer_0']['bge_mean'].mean()}")
```

## 扩展和优化

### 可能的改进

1. **完全避免临时保存**：
   - 修改 `load_kv_and_generate` 函数，支持直接传入 KV cache
   - 在内存中应用偏置，不保存到磁盘

2. **自适应偏置**：
   - 为不同类型的文档学习不同的偏置
   - 基于文档长度或内容动态调整偏置

3. **在线更新统计**：
   - 随着处理更多样本，逐步更新统计量
   - 使用移动平均来平滑统计

4. **层级偏置**：
   - 不同层使用不同的偏置强度
   - 可能只对关键层应用偏置

## 总结

`no_preprocess_with_bias` 方法通过以下设计实现了性能和效率的平衡：

1. ✅ **跳过召回和融合**：避免最耗时的操作
2. ✅ **动态偏置应用**：lazy evaluation，只在需要时计算
3. ✅ **临时存储**：不占用持久存储空间
4. ✅ **简单高效**：只需线性变换
5. ✅ **一次统计，多次使用**：统计文件可重用

这种设计使得方法既能享受到分布对齐的好处，又能避免文档召回和融合的开销。
