# Online Lazy - 重算Token选择逻辑

## 场景示例

```
passages = [system, doc1, doc2, doc3, question]
已有KV: system ✓, doc1 ✓
缺失KV: doc2 ✗, doc3 ✗

文档长度：
- system: 100 tokens
- doc1: 200 tokens
- doc2: 150 tokens
- doc3: 180 tokens
- question: 50 tokens
```

---

## 执行流程

### ① 加载已有KV
```python
past_key_values.load(system_kv)  # [0:100]
past_key_values.load(doc1_kv)    # [100:300]
past_len = 300  # system + doc1
```

### ② 前向传播（缺失文档 + question）
```python
inputs = [doc2, doc3, question]  # 150 + 180 + 50 = 380 tokens
cache_position = [300, 301, ..., 679]  # 从past_len开始

model.forward(inputs, past_key_values)
# past_key_values现在包含:
# [0:100] system (loaded)
# [100:300] doc1 (loaded)
# [300:450] doc2 (generated)
# [450:630] doc3 (generated)
# [630:680] question (generated)
```

### ③ 计算k_sum（所有文档的importance，不包括question）
```python
# 修正前（错误）：
# k_sum = importance_cache[-1][:cache_position[0]]
# k_sum = importance_cache[-1][:300]  ← 只包含system+doc1，丢失doc2+doc3！

# 修正后（正确）：
question_len = 50
missing_docs_len = 150 + 180 = 330
all_docs_len = past_len + missing_docs_len = 300 + 330 = 630

k_sum = importance_cache[-1][:630]
# k_sum.shape = [630]，包含所有文档:
# [0:100] system importance
# [100:300] doc1 importance
# [300:450] doc2 importance  ← 新生成的！
# [450:630] doc3 importance  ← 新生成的！
```

### ④ 强制新文档重算
```python
passages_len_cumsum = [100, 300, 450, 630]  # 每个passage的累积长度

missing_chunks = [(2, 3, doc2), (3, 4, doc3)]  # (idx, chunk_id, passage)

for (idx, chunk_id, passage) in missing_chunks:
    # doc2 (idx=2):
    chunk_start = passages_len_cumsum[1] = 300
    chunk_end = passages_len_cumsum[2] = 450
    k_sum[300:450] = inf  # doc2全部强制重算

    # doc3 (idx=3):
    chunk_start = passages_len_cumsum[2] = 450
    chunk_end = passages_len_cumsum[3] = 630
    k_sum[450:630] = inf  # doc3全部强制重算

# 现在k_sum:
# [0:100] system importance
# [100:300] doc1 importance
# [300:450] inf  ← doc2强制选择
# [450:630] inf  ← doc3强制选择
```

### ⑤ FusionRAG选择重算token
```python
k_sum_relevant = k_sum[system_len:]  # 排除system
# k_sum_relevant.shape = [530]
# [0:200] doc1 importance
# [200:350] inf (doc2)
# [350:530] inf (doc3)

total_relevant_tokens = 200 + 150 + 180 = 530
k_lens = int(rate * total_relevant_tokens) = int(0.3 * 530) = 159

# 选择top-k个token
selected_indices = topk(k_sum_relevant, k_lens)
# 由于doc2和doc3的分数都是inf，它们会被优先选择
# 结果：
# - doc2的150个token全部选中（因为inf）
# - doc3的180个token全部选中（因为inf）
# - 但总共只需要159个，所以会选择：
#   * 全部330个inf token（doc2+doc3）
#   * 因为330 > 159，topk会选前159个inf token
#
# 等等，这里有问题！topk(k_sum, 159)只会选159个，
# 但doc2+doc3总共330个token。

# 实际上，应该是：
# - 先保证新文档全部选中
# - 然后从旧文档中选择剩余的

# 但当前逻辑是直接topk，这样当rate很小时，
# 可能无法保证新文档全部选中！
```

---

## ✅ 已解决的问题

### 问题：rate太小时，新文档可能无法全部重算

**原问题描述**：
```python
# 假设rate=0.1，total_tokens=530
k_lens = int(0.1 * 530) = 53

# topk只选53个token，但新文档有330个！
# 即使k_sum设为inf，topk也只会选前53个inf token
```

### ✅ 已实现的解决方案

**文件**: `ktransformers/util/utils.py:1180-1185` (group模式) 和 `1232-1237` (非group模式)

```python
# 计算重算budget
k_lens = int(rate * total_relevant_tokens)

# 确保budget足够覆盖所有新文档（ONLINE_LAZY模式）
if missing_chunks:
    missing_docs_len = sum([chunk[2].shape[0] for chunk in missing_chunks])
    if k_lens < missing_docs_len:
        print(f"    → Adjusting budget from {k_lens} to {missing_docs_len} to cover all new documents")
        k_lens = missing_docs_len
```

**效果**：
- ✅ 新文档一定全部重算（rate=1.0）
- ✅ 当rate产生的budget不足时，自动调整为新文档总长度
- ✅ 总budget = max(rate×total_tokens, new_docs_len)
- ✅ 在两个代码路径都实现了（group和非group模式）
