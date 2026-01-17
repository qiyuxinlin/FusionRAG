# No Preprocess with Distribution Bias - 使用说明

## 概述

`no_preprocess_with_bias` 是一种新的 KV cache 处理方法，灵感来源于 BatchNorm。该方法通过统计分析 BGE 召回方法和 no_preprocess 方法之间的分布差异，学习一个偏置向量，在推理时直接应用到 no_preprocess 的 KV cache 上，从而避免实际的文档召回和融合操作。

## 方法原理

### 类比 BatchNorm + Manifold Steering

BatchNorm 通过以下转换来标准化数据分布：
```
x_normalized = (x - mean) / std
x_transformed = x_normalized * gamma + beta
```

类似地，`no_preprocess_with_bias` 方法：

1. **Offline 阶段（统计 + 流形投影）**：
   - 对多个样本的 BGE 和 no_preprocess KV cache 进行分析
   - 计算每个 layer 上的均值和标准差
   - **使用 PCA 进行流形投影**：识别低维流形，消除高维噪声干扰
   - 将 bias 和 scale 向量投影到主流形上
   - 计算从 no_preprocess 到 BGE 分布的转换参数（投影后的 scale 和 bias）

2. **Online 阶段（应用）**：
   - **跳过文档召回和融合**：不进行 BGE 相似度计算和文档 KV 融合
   - **动态应用偏置**：在回答问题前，对 no_preprocess KV cache 应用预计算的 scale 和 bias
   - **临时保存**：将 biased KV cache 临时保存，用于生成答案
   - 将 no_preprocess 的分布对齐到 BGE 的分布

### 流形投影原理（来自论文）

根据 "Mitigating Overthinking in Large Reasoning Models via Manifold Steering" 论文：

- **问题**：在高维空间中直接计算偏置向量会引入干扰噪声（interference noise）
- **核心洞察**：激活值实际上存在于低维流形 M ⊂ R^d，其中 d_eff << d
- **解决方案**：使用 PCA 识别主流形，将偏置向量投影到该流形上
  ```
  P_M = U_eff @ U_eff^T  # 投影矩阵，U_eff 是 top-k 主成分
  bias_projected = P_M @ bias  # 投影消除正交补空间的噪声
  ```
- **理论保证**：Theorem 4.1 证明正交补空间 M⊥ 中的噪声会累积，投影可以消除这些噪声

### 转换公式

对于每个 layer 的 KV cache：
```
# 基础公式（与 BatchNorm 类似）
KV_aligned = KV_no_preprocess * scale + bias

# 其中 scale 和 bias 是投影到流形后的向量
scale = P_M @ (std_bge / std_no_preprocess)
bias = P_M @ (mean_bge - mean_no_preprocess * scale)
```

其中：
- `P_M`：PCA 投影矩阵，将向量投影到低维流形
- `scale_projected`：投影后的缩放因子（消除了噪声）
- `bias_projected`：投影后的偏置向量（消除了噪声）

## 使用流程

### 步骤 1: 计算分布统计量（Offline）

首先需要使用现有的 KV cache 计算分布统计量。这一步只需要运行一次。

```bash
python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --model_name Qwen2.5-7B-Instruct \
    --sample_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --chunk_id 1 \
    --max_layers 28 \
    --output_path ./kv_stats/musique_qwen2.5-7b_stats.pt \
    --use_manifold_projection \
    --pca_variance_threshold 0.7
```

**参数说明**：
- `--cache_dir`: KV cache 根目录
- `--dataset`: 数据集名称
- `--model_name`: 模型名称
- `--sample_ids`: 用于统计的样本 ID 列表（建议使用 10-20 个样本）
- `--chunk_id`: 要分析的 chunk ID（默认为 1）
- `--max_layers`: 模型的层数
- `--output_path`: 输出统计文件的路径
- `--use_manifold_projection`: 启用 PCA 流形投影（默认启用，推荐）
- `--no_manifold_projection`: 禁用流形投影（使用原始 BatchNorm 方法）
- `--pca_variance_threshold`: PCA 累积方差阈值（默认 0.7，即保留 70% 方差）

**输出文件**：
- `*.pt`: PyTorch 格式的统计文件，包含每层的 scale 和 bias（已投影到流形）
- `*.json`: 元数据文件，便于查看统计信息（包含流形投影设置）

### 步骤 2: 使用统计量运行测试（Online）

#### 方法 A: 使用 run_fusionrag_sweep.sh

修改 `run_fusionrag_sweep.sh` 中的配置：

```bash
# FusionRAG 方法配置
RECALL_METHOD="no_preprocess_with_bias"
KV_STATS_PATH="./kv_stats/musique_qwen2.5-7b_stats.pt"

# 其他配置保持不变
REPROCESS_METHOD="FusionRAG"
TOPK="10"
PREPROCESS="true"
# ...
```

然后运行：
```bash
bash run_fusionrag_sweep.sh
```

#### 方法 B: 直接使用 Python 脚本

```bash
python test_fusionrag_reflect.py \
    --model_type qwen \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path /mnt/data3/tmp/fusionrag \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/musique_qwen2.5-7b_stats.pt \
    --rate 0.3 \
    --topk 10 \
    --preprocess true
```

#### 方法 C: 使用示例脚本

我们提供了完整的示例脚本 `run_no_preprocess_with_bias_example.sh`，包含了统计计算和测试的完整流程：

```bash
bash run_no_preprocess_with_bias_example.sh
```

## 文件说明

### 新增文件

1. **compute_kv_distribution_stats.py**
   - 计算 BGE 和 no_preprocess 之间的分布差异
   - 生成 scale 和 bias 统计文件

2. **run_no_preprocess_with_bias_example.sh**
   - 完整示例脚本，演示如何使用新方法
   - 包含 offline 统计和 online 测试两个阶段

3. **NO_PREPROCESS_WITH_BIAS_README.md**
   - 本说明文档

### 修改的文件

1. **test_fusionrag_reflect.py**
   - 添加 `RecallMethod.NO_PREPROCESS_WITH_BIAS` enum
   - 添加 `load_kv_distribution_stats()` 函数
   - 添加 `apply_distribution_bias()` 函数
   - 在预处理阶段应用分布偏置
   - 添加 `--kv_stats_path` 命令行参数

2. **run_fusionrag_sweep.sh**
   - 添加 `KV_STATS_PATH` 配置变量
   - 在参数列表中添加 `--kv_stats_path` 支持

## 工作原理详解

### Offline 阶段

1. **加载 KV Cache**：
   - 从 `kv_cache/` 目录加载 no_preprocess 的 KV cache
   - 从 `preprocess_kv_cache_global_topk10_bge/` 加载 BGE 的 KV cache

2. **计算统计量**：
   ```python
   # 对每个 layer
   for layer in range(num_layers):
       # 计算均值和标准差（在 seq_len 维度上）
       bge_mean = bge_kv[layer].mean(dim=seq_len)
       bge_std = bge_kv[layer].std(dim=seq_len)
       no_prep_mean = no_prep_kv[layer].mean(dim=seq_len)
       no_prep_std = no_prep_kv[layer].std(dim=seq_len)

       # 计算转换参数
       scale = bge_std / (no_prep_std + eps)
       bias = bge_mean - no_prep_mean * scale
   ```

3. **聚合跨样本统计**：
   - 对多个样本的统计量取平均
   - 保存每层的 scale 和 bias

### Online 阶段（动态应用）

**关键设计**：`PREPROCESS="true"` 但跳过召回和融合

1. **Step 1: 生成 no_preprocess KV cache**（与普通流程相同）
   ```python
   # 为每个文档生成独立的 KV cache
   prefill_and_save_kv_cache(doc_tensor, save_path)
   ```

2. **Step 2: 跳过文档召回和融合**
   - 对于 NO_PREPROCESS_WITH_BIAS 方法，跳过整个 preprocess 阶段
   - 不进行 BGE 相似度计算
   - 不进行文档 KV cache 融合

3. **Step 2.5: 回答问题前动态应用偏置**（lazy evaluation）
   ```python
   # 只在需要回答这个 example 的问题时才应用偏置
   for chunk_id in doc_chunks:
       # 加载 no_preprocess KV Cache
       no_prep_key = torch.load(f"{save_path}/{example_id}_{chunk_id}_key.pt")
       no_prep_value = torch.load(f"{save_path}/{example_id}_{chunk_id}_value.pt")

       # 应用分布偏置
       for layer in range(num_layers):
           biased_key[layer] = no_prep_key[layer] * scale + bias
           biased_value[layer] = no_prep_value[layer] * scale + bias

       # 临时保存（用于 load_kv_and_generate）
       torch.save(biased_key, f"{preprocess_save_path}/{example_id}_{chunk_id}_key.pt")
       torch.save(biased_value, f"{preprocess_save_path}/{example_id}_{chunk_id}_value.pt")
   ```

4. **Step 3: 使用 biased KV cache 生成答案**
   - `load_kv_and_generate` 从 `preprocess_save_path` 加载 biased KV cache
   - 正常解码生成答案

5. **Step 4: 自动清理临时文件**
   - 回答完该 example 的所有子问题后
   - 自动删除所有临时 biased KV cache 文件
   - 释放存储空间

## 优势

1. **跳过文档召回阶段**：
   - 不需要进行 BGE 相似度计算
   - 不需要召回和排序相似文档
   - 不需要融合多个文档的 KV cache

2. **动态偏置应用 + 自动清理**：
   - Lazy evaluation：只在需要回答问题时才应用偏置
   - 用完即删：回答完问题后自动清理临时文件
   - 零持久存储开销：不会留下任何额外的 KV cache 文件
   - 内存友好：同一时间只保留一个 example 的 biased KV cache

3. **计算高效**：
   - 偏置应用只需简单的线性变换（乘法和加法）
   - 比文档召回和融合快得多

4. **流形投影消除噪声**（新增）：
   - 基于 PCA 识别低维流形，消除高维空间的干扰噪声
   - 理论保证（Theorem 4.1）：正交补空间的噪声会被投影消除
   - 实验显示 k=10 维主成分即可捕获 >70% 方差
   - 投影后的偏置向量更准确，减少无关维度的干扰

5. **一次统计，多次使用**：
   - 统计文件只需计算一次
   - 可以在多个实验中重复使用
   - 不同的 rate 可以共享同一个统计文件

## 注意事项

1. **统计样本数量**：
   - 建议使用 10-20 个样本计算统计量
   - 样本太少可能导致统计不准确
   - 样本太多会增加计算时间

2. **chunk_id 选择**：
   - 默认使用 chunk_id=1（第一个文档块）
   - 如果需要分析其他 chunk，需要修改 `--chunk_id` 参数

3. **统计文件管理**：
   - 不同数据集需要单独计算统计量
   - 不同模型需要单独计算统计量
   - 建议使用清晰的命名规则：`{dataset}_{model}_stats.pt`

4. **依赖关系**：
   - 必须先运行正常的 no_preprocess 和 BGE 方法生成 KV cache
   - 统计文件必须在运行 online 测试前生成

## 实验建议

### 对比实验

建议进行以下对比实验来评估方法效果：

1. **Baseline: no_preprocess**
   ```bash
   RECALL_METHOD="bge"
   PREPROCESS="false"
   ```

2. **BGE 召回（标准方法）**
   ```bash
   RECALL_METHOD="bge"
   PREPROCESS="true"
   ```

3. **No Preprocess + Distribution Bias（新方法）**
   ```bash
   RECALL_METHOD="no_preprocess_with_bias"
   KV_STATS_PATH="./kv_stats/musique_qwen2.5-7b_stats.pt"
   PREPROCESS="true"
   ```

### 评估指标

- **准确率**：答案正确率
- **速度**：每个问题的平均处理时间
- **内存**：峰值内存使用
- **与 BGE 的差距**：新方法相比 BGE 的性能差距

## 故障排除

### 问题 1: "Statistics file not found"

**原因**：未正确指定统计文件路径或文件不存在

**解决**：
```bash
# 检查文件是否存在
ls -lh ./kv_stats/musique_qwen2.5-7b_stats.pt

# 如果不存在，先运行步骤 1 计算统计量
python compute_kv_distribution_stats.py ...
```

### 问题 2: "No_preprocess KV cache not found"

**原因**：no_preprocess 的 KV cache 尚未生成

**解决**：
```bash
# 先运行 no_preprocess 方法生成 KV cache
RECALL_METHOD="bge"
PREPROCESS="false"
bash run_fusionrag_sweep.sh
```

### 问题 3: 统计计算时内存不足

**原因**：同时加载太多样本的 KV cache

**解决**：
- 减少 `--sample_ids` 的数量
- 或分批计算统计量后手动合并

## 技术细节

### KV Cache 形状

- Key/Value cache 形状：`[num_layers, num_heads, seq_len, head_dim]`
- 统计量计算在 `seq_len` 维度上进行
- Bias/Scale 形状：`[num_heads, head_dim]`（每层）

### 统计文件格式

```python
{
    'key_stats': {
        'layer_0': {
            'bge_mean': Tensor[num_heads, head_dim],
            'bge_std': Tensor[num_heads, head_dim],
            'no_prep_mean': Tensor[num_heads, head_dim],
            'no_prep_std': Tensor[num_heads, head_dim],
            'bias': Tensor[num_heads, head_dim],
            'scale': Tensor[num_heads, head_dim]
        },
        'layer_1': {...},
        ...
    },
    'value_stats': {
        'layer_0': {...},
        ...
    },
    'metadata': {
        'num_samples': int,
        'num_layers': int,
        'sample_ids': list,
        'chunk_id': int
    }
}
```

## 参考

- 相关脚本：`compute_kv_distribution_stats.py`
- 主测试脚本：`test_fusionrag_reflect.py`
- 示例脚本：`run_no_preprocess_with_bias_example.sh`
- Sweep 脚本：`run_fusionrag_sweep.sh`

## 更新日志

- **2025-01-16**: 初始版本，添加 no_preprocess_with_bias 方法支持
