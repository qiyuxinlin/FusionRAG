# 跨数据集实验：2wikimqa → musique

## 实验设计

测试 **Steering Vectors 的泛化能力**：

- **训练（Offline）**：在 2wikimqa 数据集上计算 steering vectors
- **测试（Online）**：在 musique 数据集上应用这些 steering vectors

## 快速开始

### 1. 快速测试（5个样本）

```bash
bash test_cross_dataset_quick.sh
```

输出示例：
```
✓ Test successful!
Steering vectors saved to: ./kv_stats/test_2wikimqa_steering.pt

  - Samples analyzed: 5
  - Layers processed: 28
  - Manifold projection: Enabled (variance threshold: 0.7)
  - Dimension reduction: 512 -> 20-70 (per layer)
```

### 2. 完整实验（20个样本）

```bash
bash run_cross_dataset_experiment.sh
```

这将运行 **4种方法**的对比实验：
1. No Preprocess (baseline on musique)
2. Steering w/o projection (2wikimqa → musique)
3. **Steering + Manifold** (2wikimqa → musique, 推荐)
4. BGE (upper bound on musique)

## 数据集路径

### 2wikimqa（训练）
```
/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/2wikimqa/
├── kv_cache/                                      # no_preprocess
│   ├── 0_1_key.pt
│   ├── 0_1_value.pt
│   └── ...
└── preprocess_kv_cache_global_topk10_bge/        # BGE
    ├── 0_1_key.pt
    ├── 0_1_value.pt
    └── ...
```

### musique（测试）
```
/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
├── kv_cache/                                      # no_preprocess
└── preprocess_kv_cache_global_topk10_bge/        # BGE
```

## KV Cache 格式

两个数据集都使用 **Grouped Attention** 格式：

```python
# Shape: [num_layers, num_groups, num_heads_per_group, seq_len, head_dim]
2wikimqa: [28, 1, 4, 64, 128]   # seq_len = 64
musique:  [28, 1, 4, 50, 128]   # seq_len = 50
```

**关键**：`seq_len` 不同不影响 steering vectors，因为计算时在 `seq_len` 维度求平均。

## 实验参数

| 参数 | 值 |
|-----|-----|
| 训练样本数 | 20 (2wikimqa) |
| 测试数据集 | musique |
| Manifold projection | 启用 |
| PCA 方差阈值 | 0.7 (70%) |
| Steering strength (α) | 1.0 (可调) |
| Rate | [0.0, 0.3, 0.5, 1.0] |

### Steering Alpha 参数说明

**公式**: `h' = h + α * r_M`

**Alpha (α) 作用**：
- **α = 1.0** (默认): 完全应用 steering vector（论文标准方法）
- **α < 1.0** (如 0.5, 0.8): 更弱的干预，更保守的修正
  - 适用场景：担心 steering vector 过度修改原始表示
  - 效果：保留更多原始 KV cache 的信息
- **α > 1.0** (如 1.2, 1.5): 更强的干预，更激进的修正
  - 适用场景：原始分布偏离理想状态较远
  - 效果：更大幅度地向目标分布靠拢

**调参建议**：
```bash
# 在脚本中修改 STEERING_ALPHA 参数
STEERING_ALPHA="1.0"  # 默认值

# 或在命令行中指定
python test_fusionrag_reflect.py \
    --steering_alpha 0.8 \
    --kv_stats_path ./kv_stats/steering.pt \
    ...
```

**超参数搜索示例**：
```bash
for ALPHA in 0.5 0.8 1.0 1.2 1.5; do
    python test_fusionrag_reflect.py \
        --steering_alpha ${ALPHA} \
        --result_path ./result/alpha_${ALPHA}/ \
        ...
done
```

## 预期结果

### 维度降低（PCA）

原始维度：`num_heads × head_dim = 4 × 128 = 512`

PCA 后（70% 方差）：
```
Layer 0 Key:   36 components (93% reduction)
Layer 0 Value: 75 components (85% reduction)
...
Layer 27 Key:  29 components (94% reduction)
Layer 27 Value: 45 components (91% reduction)
```

### 性能预期（跨数据集）

| 方法 | 准确率（musique） | 说明 |
|------|------------------|------|
| No Preprocess | 基准 | Baseline |
| Steering (no proj) | +3-5% | 可能有分布偏移 |
| **Steering + Manifold** | **+5-8%** | **流形投影帮助泛化** |
| BGE (upper bound) | +15% | 数据集内上限 |

**关键问题**：跨数据集的泛化能力如何？

- ✅ 如果效果接近：说明 steering vectors 学到了通用模式
- ❌ 如果效果下降：说明有数据集特定的分布差异

## 理论依据

### 为什么跨数据集可能有效？

1. **共享的模型架构**：同一个 Qwen2.5-7B 模型
2. **相似的任务**：都是多跳问答任务
3. **流形投影的帮助**：
   - 投影到低维流形可能捕获更通用的特征
   - 消除数据集特定的噪声

### Theorem 4.1（论文）

```
E[||r_noise||²] = tr((I - P_M)Σ_noise)
```

**含义**：流形投影消除正交补空间的噪声，保留主要信号。

**跨数据集**：主要信号（任务通用）在流形上，数据集特定噪声在 M⊥。

## 文件输出

### Steering Vectors
```
./kv_stats/cross_dataset/
├── 2wikimqa_to_musique_no_proj.pt      # 无投影
├── 2wikimqa_to_musique_no_proj.json
├── 2wikimqa_to_musique_manifold.pt     # 有投影（推荐）
└── 2wikimqa_to_musique_manifold.json
```

### 测试结果
```
./result/cross_dataset/
├── musique_baseline/                    # No preprocess
├── musique_steering_no_proj/            # Steering w/o projection
├── musique_steering_manifold/           # Steering + Manifold
└── musique_bge/                         # BGE (upper bound)
```

## 代码修改（已完成）

### 1. 支持 5D KV Cache

```python
# compute_kv_distribution_stats.py

def compute_layer_mean(cache_tensor):
    if cache_tensor.dim() == 4:
        # Grouped: [num_groups, num_heads_per_group, seq_len, head_dim]
        cache_tensor = cache_tensor.reshape(num_heads, seq_len, head_dim)
    # Mean over seq_len
    return cache_tensor.mean(dim=1)
```

### 2. 处理 bfloat16

```python
def compute_manifold_projection(activations):
    # numpy 不支持 bfloat16，转换为 float32
    if activations.dtype == torch.bfloat16:
        activations = activations.float()
    # ...
```

### 3. 路径配置

```bash
# 基础路径（不包含 model_name/dataset）
CACHE_BASE="/mnt/data3/tmp/fusionrag"

# 代码自动添加 model_name/dataset
# -> /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/2wikimqa/
```

## 对比实验

完整的 4×4 对比矩阵：

| 训练 → 测试 | musique → musique | 2wikimqa → musique |
|------------|-------------------|-------------------|
| No Preprocess | 基准 | 基准（musique） |
| Steering (no proj) | 同数据集 | **跨数据集** |
| Steering + Manifold | 同数据集（最佳） | **跨数据集（测试泛化）** |
| BGE | 上限 | 上限（musique） |

## 使用示例

### 计算 Steering Vectors（2wikimqa）

**方法 1: 指定样本 ID**
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

**方法 2: 自动使用所有样本（新功能！）**
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

**说明**：
- `--sample_ids`: 手动指定样本 ID 列表
- `--use_all_samples`: 自动扫描并使用数据集中的所有样本（两者互斥）
- 使用 `--use_all_samples` 时会自动找到 no_preprocess 和 bge 目录中共同存在的所有样本

### 在 musique 上测试

**基础用法（所有层应用 steering）**
```bash
python test_fusionrag_reflect.py \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path /mnt/data3/tmp/fusionrag \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/2wikimqa_steering.pt \
    --steering_alpha 1.0 \
    --preprocess true \
    --rate 0.3
```

**选择性层级应用（新功能！）**
```bash
# 只对前 10 层的 Key 和 Value 应用 steering
python test_fusionrag_reflect.py \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path /mnt/data3/tmp/fusionrag \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/2wikimqa_steering.pt \
    --steering_alpha 1.0 \
    --steering_key_layers "0-9" \
    --steering_value_layers "0-9" \
    --preprocess true \
    --rate 0.3

# 只对 Key 的后半层应用，Value 保持不变
python test_fusionrag_reflect.py \
    --steering_key_layers "14-27" \
    --steering_value_layers "none" \
    ...

# 对特定层应用（如浅层 + 深层）
python test_fusionrag_reflect.py \
    --steering_key_layers "0-5,20-27" \
    --steering_value_layers "0-5,20-27" \
    ...
```

**层选择参数说明**：
- `--steering_key_layers`: 应用 Key steering vector 的层
- `--steering_value_layers`: 应用 Value steering vector 的层
- 格式支持:
  - `"all"` (默认): 所有层 (0-27)
  - `"0-10"`: 范围 (层 0 到 10)
  - `"0,5,10"`: 特定层
  - `"0-5,10,15-20"`: 混合格式
  - `"none"` 或 `""`: 不应用任何层

**调整 alpha 参数**：
```bash
# 测试不同的 steering strength
for ALPHA in 0.5 0.8 1.0 1.2 1.5; do
    python test_fusionrag_reflect.py \
        --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
        --data_path ./data/result_reflect.json \
        --dataset_name musique \
        --cache_path /mnt/data3/tmp/fusionrag \
        --recall_method no_preprocess_with_bias \
        --kv_stats_path ./kv_stats/2wikimqa_steering.pt \
        --steering_alpha ${ALPHA} \
        --result_path ./result/musique_alpha_${ALPHA}/ \
        --preprocess true \
        --rate 0.3
done
```

## 总结

✅ **实现完成**：
- 支持 5D grouped attention KV cache
- 支持 bfloat16 dtype
- 跨数据集路径配置
- 完整的测试脚本

✅ **实验设计**：
- 2wikimqa（200 samples） → musique
- 测试 steering vectors 泛化能力
- 4 种方法对比

🎯 **关键问题**：
- 跨数据集的泛化能力如何？
- 流形投影是否帮助泛化？
- 与同数据集性能差距多大？

📊 **下一步**：运行完整实验，分析结果
