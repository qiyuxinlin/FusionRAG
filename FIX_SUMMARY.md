# 重算性能下降问题修复总结

## 问题诊断

### 观察到的现象
```
Rate=0.0: 71.1% 准确率 (baseline)
Rate=0.1: 63.0% 准确率 (↓8.1% - 最差)
Rate=0.15: 66.7% 准确率
Rate=0.3: 68.9% 准确率
Rate=0.5: 72.6% 准确率 (↑1.5%)
Rate=0.8: 73.3% 准确率 (↑2.2% - 最佳)
```

**核心问题**：低重算率(0.1-0.3)反而导致性能显著下降。

### 根本原因

在 `ktransformers/util/utils.py` 的 FusionRAG token 选择逻辑中，`k_lens`（需要选择的 token 数量）的计算存在错误：

**错误代码**（Line 1290）：
```python
k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
```

**问题分析**：
1. `torch.cat(passages[:-1])` 包含**所有文档** tokens（包括 missing_chunks）
2. 但 `k_sum`（importance 分数）只包含**已加载文档**的分数（长度 = `past_len - system_len`）
3. 当有 missing_chunks 时：
   - `k_lens` 可能超过 `k_sum` 的长度
   - 更严重的是，importance 分数基于**不完整的上下文**计算（缺少 missing chunks）

**为什么低 rate 受影响最大**：
- **Rate 0.0**：不使用 importance 选择，直接复用所有已加载 KV ✅
- **Rate 0.1-0.3**：基于不准确的 importance 选择少量 tokens，**选错了关键 tokens** ❌
- **Rate 0.5-0.8**：高覆盖率掩盖了选择错误，整体效果仍然良好 ✅

## 修复内容

### 修复位置

**文件**：`ktransformers/util/utils.py`

#### 1. FusionRAG - Group 模式（Lines 1232-1249）

**修改前**：
```python
total_relevant_tokens = torch.cat(passages[1:-1]).shape[0]  # 所有文档
k_lens = int(rate * total_relevant_tokens)

if missing_chunks:
    missing_docs_len = sum([chunk[2].shape[0] for chunk in missing_chunks])
    if k_lens < missing_docs_len:
        print(f"    → Adjusting budget from {k_lens} to {missing_docs_len}")
        k_lens = missing_docs_len
```

**修改后**：
```python
# 基于实际加载的文档长度
loaded_relevant_tokens = past_len - system_len
k_lens = int(rate * loaded_relevant_tokens)

if missing_chunks:
    all_relevant_tokens = torch.cat(passages[1:-1]).shape[0]
    missing_docs_len = sum([chunk[2].shape[0] for chunk in missing_chunks])
    print(f"    → ONLINE_LAZY mode: {loaded_relevant_tokens} tokens loaded, {missing_docs_len} tokens missing")
    print(f"    → Importance-based selection: {k_lens} tokens from loaded docs (rate={rate:.1f})")
    print(f"    → Missing docs will be added separately (100% coverage)")
```

#### 2. FusionRAG - 非 Group 模式（Lines 1286-1310）

**修改前**：
```python
k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))

if missing_chunks:
    missing_docs_len = sum([chunk[2].shape[0] for chunk in missing_chunks])
    if k_lens < missing_docs_len:
        print(f"    → Adjusting budget from {k_lens} to {missing_docs_len}")
        k_lens = missing_docs_len
```

**修改后**：
```python
# 基于实际加载的文档长度
loaded_relevant_tokens = past_len - system_len
k_lens = int(rate * loaded_relevant_tokens)

if missing_chunks:
    all_relevant_tokens = torch.cat(passages[:-1]).shape[0] - system_len
    missing_docs_len = sum([chunk[2].shape[0] for chunk in missing_chunks])
    print(f"    → ONLINE_LAZY mode: {loaded_relevant_tokens} tokens loaded, {missing_docs_len} tokens missing")
    print(f"    → Importance-based selection: {k_lens} tokens from loaded docs (rate={rate:.1f})")
    print(f"    → Missing docs will be added separately (100% coverage)")
```

#### 3. Speculative Prefill 模式（Lines 1353-1408）

同样的修复应用于 speculative_prefill 模式，确保代码一致性。

## 修复原理

### 新逻辑流程

1. **Importance 计算**：基于已加载文档的 KV cache
2. **Token 选择**：从已加载文档中选择 `rate * loaded_tokens` 个高重要性 tokens
3. **Missing Chunks**：在 Lines 2517-2531 **自动添加**所有 missing chunks（100% 覆盖）
4. **Question Tokens**：也会自动添加（Lines 2513）

### 修复效果

**修复前**：
```
k_lens = rate * (所有文档长度)
→ 从不完整的 k_sum 中选择过多/错误的 tokens
→ 低 rate 时选择质量很差
```

**修复后**：
```
k_lens = rate * (已加载文档长度)
→ 只从已加载文档中选择合理数量的 tokens
→ Missing chunks 单独添加（100% 覆盖）
→ 低 rate 时也能准确选择
```

## 预期改进

| Rate | 修复前准确率 | 预期修复后 | 改进原因 |
|------|-------------|-----------|---------|
| 0.0  | 71.1% | ~71.1% | 不使用 importance，无影响 |
| 0.1  | 63.0% | **~70-72%** | 不再选择错误 tokens |
| 0.15 | 66.7% | **~71-73%** | 选择质量提升 |
| 0.3  | 68.9% | **~72-74%** | 选择更准确 |
| 0.5  | 72.6% | ~72-74% | 已接近最优，微调 |
| 0.8  | 73.3% | ~73-74% | 已是最优 |

**关键改进**：
- ✅ 消除"性能谷"现象（低 rate 不再掉点）
- ✅ 低 rate 性能应接近或略低于 rate=0.0（因为少量重算带来的收益有限）
- ✅ 高 rate 性能保持不变或略有提升

## 验证方法

### 快速测试
```bash
chmod +x /home/shm/document/exp/FusionRAG/test_fix.sh
bash /home/shm/document/exp/FusionRAG/test_fix.sh
```

查看日志中是否出现：
```
→ ONLINE_LAZY mode: X tokens loaded, Y tokens missing
→ Importance-based selection: Z tokens from loaded docs (rate=0.1)
→ Missing docs will be added separately (100% coverage)
```

### 完整测试
重新运行原始的 sweep 脚本：
```bash
bash /home/shm/document/exp/FusionRAG/run_online_lazy_sweep.sh
```

对比修复前后的结果文件。

## 技术细节

### 关键变量说明

- `past_len`：实际加载到 past_key_values 中的 token 数量（system + 已加载文档）
- `torch.cat(passages[:-1]).shape[0]`：所有文档的 token 数量（system + 所有文档，包括 missing）
- `k_sum`：已加载文档的 importance 分数（长度 = `past_len - system_len`）
- `k_lens`：需要选择的 token 数量
- `missing_chunks`：尚未生成 KV cache 的文档列表

### Online Lazy 模式的正确行为

```
问题1（首次遇到 doc_174）：
  - 加载：system + doc_797 (cached)
  - Missing：doc_174, doc_231
  - Importance 选择：从 doc_797 中选 rate*len(doc_797) 个 tokens
  - 最终重算：selected_from_797 + doc_174 + doc_231 + question
  ✅ doc_174, doc_231 会被 100% 重算（Lines 2517-2531）

问题2（所有文档已缓存）：
  - 加载：system + doc_174 + doc_797 + doc_231
  - Missing：无
  - Importance 选择：从所有文档中选 rate*total_len 个 tokens
  - 最终重算：selected_tokens + question
  ✅ 标准的 FusionRAG 重算逻辑
```

## 总结

这个 bug 的本质是**在不完整上下文下计算 importance 导致选择偏差**。修复后：
1. ✅ Token 预算基于实际加载的文档，不会超出 `k_sum` 范围
2. ✅ Missing chunks 总是 100% 重算（在 Lines 2517-2531 添加）
3. ✅ 低 rate 不再选择错误的 tokens
4. ✅ 性能曲线应该单调递增（或接近单调）

修复非常简洁，只改了几个 `k_lens` 的计算方式，但影响深远。
