# BGE_SHUFFLED召回模式使用说明

## 功能概述

**BGE_SHUFFLED模式**是一个新的召回方法，用于验证**召回文档内部顺序的重要性**。

### 实验设计思路

- **召回阶段**：使用BGE相似度召回topk个文档（与BGE模式完全相同）
- **融合阶段**：对每个召回文档的KV cache进行**随机打乱**（shuffle），破坏文档内部的token顺序
- **当前文档**：不打乱当前文档的KV顺序，只打乱召回文档
- **目的**：如果打乱后性能显著下降，说明召回文档内部的语义顺序很重要；如果性能不变，说明可能只需要这些token的存在，而不关心它们的顺序

### 与其他召回方法的对比

| 召回方法 | 召回逻辑 | 融合内容 | 验证目的 |
|---------|---------|---------|---------|
| **BGE** | BGE相似度 | 原始顺序的KV | Baseline |
| **BGE_SHUFFLED** | **BGE相似度** | **打乱顺序的KV** | **顺序重要性** |
| **Random** | 随机召回 | 原始顺序的KV | 排除语义相关性 |
| **RANDOM_TEXT** | BGE长度匹配 | 无关文本KV | 位置适应假设 |

## 实现细节

### 1. Shuffle机制
```python
# 对每个召回文档的KV cache进行shuffle
if recall_method_enum == RecallMethod.BGE_SHUFFLED:
    # 生成随机排列索引
    seq_len = chunk_key_cache[0].shape[1]  # 文档长度
    shuffle_indices = torch.randperm(seq_len)  # 随机排列

    # 对所有层使用相同的shuffle索引（保持K和V对应）
    for layer_idx in range(len(chunk_key_cache)):
        chunk_key_cache[layer_idx] = chunk_key_cache[layer_idx][:, shuffle_indices, :]
        chunk_value_cache[layer_idx] = chunk_value_cache[layer_idx][:, shuffle_indices, :]
```

### 2. 关键特性
- ✅ **所有层使用相同的shuffle索引**：保证同一位置的K和V仍然对应
- ✅ **只打乱召回文档**：当前文档保持原始顺序
- ✅ **每次运行shuffle不同**：每次加载KV时重新生成随机索引
- ✅ **打乱token级别**：在sequence维度上打乱，完全破坏文档内部的语义顺序

### 3. Shuffle范围
```
System KV:     [不打乱]
召回文档1 KV:  [打乱] token_3, token_7, token_1, token_5, ...
召回文档2 KV:  [打乱] token_9, token_2, token_6, token_4, ...
...
召回文档K KV:  [打乱] token_8, token_1, token_3, token_2, ...
当前文档 KV:   [不打乱] token_1, token_2, token_3, ...
```

## 使用方法

### 方法1: 修改启动脚本运行

编辑 `run_fusionrag.sh`:
```bash
RECALL_METHOD="bge_shuffled"  # 修改这一行
```

运行:
```bash
bash run_fusionrag.sh
```

### 方法2: 直接运行Python脚本

```bash
/home/shm/anaconda3/envs/fusionrag/bin/python test_fusionrag_reflect.py \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --data_path ./data/result_reflect.json \
    --cache_path /mnt/data3/tmp/fusionrag \
    --result_path ./result \
    --rate 0.15 \
    --topk 10 \
    --preprocess \
    --recall_method bge_shuffled \
    --preprocess_scope global \
    --revert_rope \
    --device cuda:0
```

### 方法3: TopK扫描实验

编辑 `run_fusionrag_topk_sweep.sh`:
```bash
RECALL_METHOD="bge_shuffled"  # 修改这一行
```

运行:
```bash
bash run_fusionrag_topk_sweep.sh
```

## 文件说明

### KV Cache存储
- **原始文档KV**: `/mnt/data3/tmp/fusionrag/{model}/{dataset}/kv_cache/`
  - 格式: `{example_id}_{chunk_id}_key.pt` / `{example_id}_{chunk_id}_value.pt`
  - 这些KV会在加载时被shuffle
- **预处理KV**: `/mnt/data3/tmp/fusionrag/{model}/{dataset}/preprocess_kv_cache_global_topk{K}_bge_shuffled/`
  - 包含shuffle后的融合KV

### 结果目录
- **路径**: `result/{model}/{dataset}/FusionRAG_global_topk{K}_bge_shuffled/`

## 预期实验结果

### 假设1: 顺序非常重要（语义完整性）
如果**BGE_SHUFFLED性能 << BGE性能**，说明:
- 召回文档内部的token顺序承载了重要的语义信息
- 打乱顺序破坏了语义完整性，导致性能下降
- FusionRAG需要召回文档的**完整语义结构**，而不仅仅是token的集合

### 假设2: 顺序不重要（Bag-of-Tokens）
如果**BGE_SHUFFLED性能 ≈ BGE性能**，说明:
- 召回文档可能只提供了一个"token池"
- 模型可能通过attention机制自己找到相关信息，不依赖原始顺序
- 支持"Bag-of-Tokens"假设

### 假设3: 部分重要（介于两者之间）
如果**BGE性能 > BGE_SHUFFLED性能 > Random性能**，说明:
- 顺序有一定作用，但不是决定性因素
- 语义相关性比顺序更重要

### 预期性能排序（从高到低）
1. **BGE** (相关文档 + 正确顺序) ← Baseline
2. **BGE_SHUFFLED** (相关文档 + 随机顺序) ← 本实验
3. **Random** (随机文档 + 正确顺序)
4. **No_Preprocess** (无融合)

## 论文价值

### 实验意义

1. **验证顺序重要性**
   - 如果shuffle导致性能下降 → 证明顺序很重要
   - 如果shuffle不影响性能 → 证明只需要token集合

2. **区分不同机制**
   - BGE vs BGE_SHUFFLED: 隔离"顺序"因素
   - BGE_SHUFFLED vs Random: 隔离"相关性"因素
   - RANDOM_TEXT vs BGE_SHUFFLED: 隔离"内容"因素

3. **理论贡献**
   - 深入理解FusionRAG的工作机制
   - 区分"语义融合"和"token池"两种范式
   - 为优化方向提供指导

### 可能的论文观点

**场景A**: 如果shuffle严重影响性能
> "我们的实验表明，FusionRAG不仅需要相关文档的token，更需要这些token保持原始的语义顺序。BGE_SHUFFLED模式性能下降X%，证明文档的语义结构对于知识融合至关重要。"

**场景B**: 如果shuffle不影响性能
> "有趣的是，打乱召回文档的内部顺序并不影响性能（BGE vs BGE_SHUFFLED: X% vs Y%），这表明FusionRAG可能通过attention机制从'token池'中提取相关信息，而不依赖于原始文档的语义结构。"

## 注意事项

1. **每次运行shuffle不同**
   - 由于使用 `torch.randperm()`，每次运行的shuffle顺序都不同
   - 如需可重复实验，可以设置随机种子
   - 建议多次运行取平均值

2. **只打乱召回文档**
   - 当前文档（正在处理的文档）不会被打乱
   - System prompt KV也不会被打乱
   - 只有topk个召回的相似文档会被打乱

3. **KV Cache复用**
   - 原始KV cache保持不变（未打乱）
   - Shuffle在每次加载时动态进行
   - 预处理后的融合KV会保存（包含shuffle效果）

4. **与其他模式对比**
   ```bash
   # 对比不同召回方法的结果
   cat result/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.txt
   cat result/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge_shuffled/rate_0.15_revert_rope.txt
   cat result/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_random/rate_0.15_revert_rope.txt
   ```

## 实现位置

### 代码修改记录

1. `test_fusionrag_reflect.py`:
   - **Line 75**: 添加 `RecallMethod.BGE_SHUFFLED` 枚举
   - **Line 650-657**: BGE和BGE_SHUFFLED共用召回逻辑
   - **Line 1271**: 添加到 `recall_method_map`
   - **Line 1318**: 添加到 `recall_method_display`
   - **Line 1835-1846**: 实现KV shuffle逻辑（核心代码）

2. 启动脚本:
   - `run_fusionrag.sh`: Line 43 添加bge_shuffled选项
   - `run_fusionrag_topk_sweep.sh`: Line 32 添加bge_shuffled选项

### 核心代码位置

```python
# test_fusionrag_reflect.py, Line 1835-1846
if recall_method_enum == RecallMethod.BGE_SHUFFLED:
    # Generate shuffle indices for the sequence dimension
    seq_len = chunk_key_cache[0].shape[1]
    shuffle_indices = torch.randperm(seq_len)

    # Apply same shuffle to all layers
    for layer_idx in range(len(chunk_key_cache)):
        # Shuffle along sequence dimension (dim=1)
        chunk_key_cache[layer_idx] = chunk_key_cache[layer_idx][:, shuffle_indices, :]
        chunk_value_cache[layer_idx] = chunk_value_cache[layer_idx][:, shuffle_indices, :]
```

## 扩展实验建议

1. **固定Shuffle顺序**
   - 修改代码使用固定随机种子
   - 验证结果可重复性

2. **部分Shuffle**
   - 只打乱前50%或后50%的tokens
   - 研究不同位置token的重要性

3. **窗口Shuffle**
   - 在小窗口内打乱（保持局部顺序）
   - 研究局部vs全局顺序的影响

4. **层级Shuffle**
   - 不同层使用不同shuffle索引
   - 研究跨层一致性的重要性
