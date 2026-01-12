# Attention Distillation 方案：让 3B Draft Model 达到 7B 效果

## 1. 问题背景

### 1.1 当前状况
- **7B 作为 Draft Model**: 准确率 86.40%
- **3B 作为 Draft Model**: 准确率 84.00%
- **差距**: 2.4% (18 个 critical cases)

### 1.2 问题分析
通过分析 3B 和 7B 的 attention 分布，发现：

| 指标 | 数值 |
|------|------|
| Attention Correlation | 0.65 |
| Token Selection IoU | 0.57 |
| Layer Selection Overlap | 0.8/4 |

**核心问题**：3B 和 7B 的 attention 模式本质上不同，即使层映射、分布校准等方法也无法改善。

### 1.3 尝试过的方法

| 方法 | IoU with 7B | 结论 |
|------|-------------|------|
| Baseline (3B 熵选层) | 0.568 | 基准 |
| 层映射 | 0.568 | 无改善 |
| 分布校准 | 0.457 | 更差 |
| 融合 Embedding | 0.258 | 更差 |
| 蒸馏模型 (DistilQwen2.5-3B) | 0.509 | 无改善 |

---

## 2. 推荐方案：Query-Document Attention Distillation

### 2.1 核心思想

不需要对齐完整的 attention 矩阵，只需要对齐 **query→document** 的部分：

```
完整 attention: [seq_len, seq_len] = [2000, 2000] → 4M 元素/层
Query→Doc attention: [query_len, doc_len] = [30, 1200] → 36K 元素/层
```

**优势**：
- 内存减少 100x+
- 更聚焦于 RAG 场景关心的 attention 模式
- 可以用更大的 batch size

### 2.2 训练流程

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: 离线预计算 7B 的 query→doc attention          │
├─────────────────────────────────────────────────────────┤
│  for each sample in dataset:                            │
│      input_ids = tokenize(system + docs + query)        │
│      attn_7b = model_7b.forward(input_ids)              │
│      q2d_attn = attn_7b[:, query_start:, doc_start:end] │
│      # Shape: [num_layers, num_heads, query_len, doc_len]│
│      save_compressed(q2d_attn)                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 2: 训练 3B 对齐 7B 的 attention                   │
├─────────────────────────────────────────────────────────┤
│  for each batch:                                        │
│      # Forward 3B                                       │
│      attn_3b = model_3b.forward(input_ids)              │
│      q2d_3b = attn_3b[:, query_start:, doc_start:end]   │
│                                                         │
│      # Load pre-computed 7B attention                   │
│      q2d_7b = load(sample_id)                           │
│                                                         │
│      # Compute loss                                     │
│      L_attn = KL(q2d_3b || q2d_7b)                      │
│      L_total = L_LM + α * L_attn                        │
│                                                         │
│      # Backward & update                                │
│      L_total.backward()                                 │
│      optimizer.step()                                   │
└─────────────────────────────────────────────────────────┘
```

### 2.3 层映射策略

3B (36 layers) 和 7B (28 layers) 按相对深度映射：

```
3B Layer 18 (51%) -> 7B Layer 13
3B Layer 21 (60%) -> 7B Layer 16
3B Layer 25 (71%) -> 7B Layer 19
3B Layer 29 (83%) -> 7B Layer 22
3B Layer 33 (94%) -> 7B Layer 25
3B Layer 35 (100%) -> 7B Layer 27
```

### 2.4 Head 数量处理

- 3B: 16 heads
- 7B: 28 heads

策略：将 7B 的 28 个 heads 分组平均为 16 个，以匹配 3B。

```python
# 7B heads 分组平均
teacher_attn = teacher_attn.view(batch, 16, 28//16, query_len, doc_len)
teacher_attn = teacher_attn.mean(dim=2)  # [batch, 16, query_len, doc_len]
```

---

## 3. 实现细节

### 3.1 Loss 设计

```python
def attention_distillation_loss(student_attn, teacher_attn, temperature=1.0):
    """
    Args:
        student_attn: [batch, heads, query_len, doc_len]
        teacher_attn: [batch, heads, query_len, doc_len]
    """
    # Softmax with temperature
    student_soft = F.softmax(student_attn / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_attn / temperature, dim=-1)

    # KL divergence
    loss = F.kl_div(
        student_soft.log(),
        teacher_soft,
        reduction='batchmean'
    ) * (temperature ** 2)

    return loss
```

### 3.2 训练超参数

| 参数 | 推荐值 |
|------|--------|
| Learning Rate | 1e-5 |
| Batch Size | 4-8 |
| Epochs | 3-5 |
| α (attention loss weight) | 1.0-10.0 |
| Temperature | 1.0-2.0 |
| Layers to align | 后 50% 的层 |

### 3.3 数据准备

使用现有的 RAG 数据集：
- `result_reflect.json` 中的 250 个 testable 样本
- 可扩展到其他 RAG 数据集

---

## 4. 资源需求

### 4.1 Phase 1: 预计算 7B Attention

| 项目 | 需求 |
|------|------|
| GPU | 1x 24GB (加载 7B) |
| 时间 | ~0.5s/sample → 1000 samples ≈ 8 分钟 |
| 存储 | ~50KB/sample (压缩后) → 1000 samples ≈ 50MB |

### 4.2 Phase 2: 训练 3B

| 项目 | 需求 |
|------|------|
| GPU | 1x 16GB (加载 3B + gradients) |
| 时间 | ~2-4 小时 (3 epochs, 1000 samples) |
| 存储 | ~6GB (model checkpoint) |

---

## 5. 预期效果

| 指标 | 训练前 | 训练后 (预期) |
|------|--------|---------------|
| Attention Correlation | 0.65 | 0.80+ |
| Token Selection IoU | 0.57 | 0.70-0.80 |
| Accuracy Gap (vs 7B) | 2.4% | <1% |

---

## 6. 风险与备选方案

### 6.1 风险

1. **过拟合**：可能在训练集上过拟合 7B 的 attention 模式
   - 缓解：使用更多数据、early stopping、正则化

2. **LM 能力下降**：attention distillation 可能损害原始 LM 能力
   - 缓解：调整 α 权重、使用较小的学习率

3. **泛化性**：在其他 RAG 数据集上可能效果下降
   - 缓解：使用多样化的训练数据

### 6.2 备选方案

1. **更激进的选择策略**：增加 3B 的选择比例 (rate: 0.2 → 0.3)
2. **多模型投票**：3B + 1.5B 选择取并集
3. **混合策略**：attention + BM25/TF-IDF 关键词匹配

---

## 7. 下一步

1. [ ] 验证 3B 模型本身（非 draft）能否回答 critical cases
2. [ ] 实现 Phase 1: 预计算 7B attention
3. [ ] 实现 Phase 2: Attention distillation 训练
4. [ ] 评估训练后的 3B 在 token selection 上的 IoU
5. [ ] 端到端测试训练后的 3B 作为 draft model 的准确率
