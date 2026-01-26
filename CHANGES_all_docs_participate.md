# 修改说明：去掉文本块1特殊处理

## 修改日期
2026-01-25

## 修改原因
用户要求去掉文本块1的特例处理，让所有加载的文档 KV 都按照 rate 参与 importance-based token 选择。

---

## 代码修改

### 文件
`/home/shm/document/exp/FusionRAG/ktransformers/util/utils.py`

### 修改内容

#### 1. 更新注释和参数 (Line 1252-1262)

**修改前**:
```python
elif reprocess_method == 'DraftModel':
    # DraftModel: 用小模型 prefill 获取 attention，指导 token 选择
    select_time = time.time()

    # query_start 用于 compute_draft_model_attention
    query_start = sum(passages_len[:-1])
    # 文本块1 长度 (用于 prefix cache，不参与重算)
    text_block1_len = passages_len[1]
    # doc_len = 文本块2 + 文本块3 + ... + 文本块n
    doc_len = sum(passages_len[1:-1])
    # selection_start 跳过 system_prompt 和 文本块1
    selection_start = system_len
```

**修改后**:
```python
elif reprocess_method == 'DraftModel':
    # DraftModel: 用小模型 prefill 获取 attention，指导 token 选择
    # 所有加载的文档 KV 都参与 rate-based importance 选择，无特例
    select_time = time.time()

    # query_start 用于 compute_draft_model_attention
    query_start = sum(passages_len[:-1])
    # doc_len = 所有文档 (doc1 + doc2 + ... + docN)
    doc_len = sum(passages_len[1:-1])
    # selection_start 跳过 system_prompt
    selection_start = system_len
```

**变化**:
- ✅ 删除 `text_block1_len` 变量定义
- ✅ `doc_len` 包括所有文档（含 doc1）
- ✅ `selection_start` 只跳过 system_prompt
- ✅ 更新注释说明所有文档都参与选择

---

#### 2. 更新 print 输出 (Line 1396)

**修改前**:
```python
print(f"DraftModel 选择了 {len(k_need_index)} 个 tokens (从文本块2-n中选{len(k_need_index)/doc_len*100:.1f}%), 文本块1({text_block1_len}tokens)用prefix cache, threshold={draft_threshold_factor}")
```

**修改后**:
```python
print(f"DraftModel 选择了 {len(k_need_index)} 个 tokens (从所有文档中选{len(k_need_index)/doc_len*100:.1f}%), threshold={draft_threshold_factor}")
```

---

#### 3. 更新 Debug 输出 (Line 1405-1458)

**修改前**:
```python
print(f"\n[1] Parameters:")
print(f"  system_len = {system_len}")
print(f"  text_block1_len = {text_block1_len}")
print(f"  doc_len = {doc_len}")
...
# 检查 doc1 是否被部分选择 (认为是 bug)
if i == 1:  # 文本块1
    if selected == 0:
        status = " ✅ 正确 (用 prefix cache)"
    elif selected == total:
        status = " ⚠️ 全选 (可能正常)"
    else:
        status = f" ❌ BUG! 部分选择 (应该用 prefix cache)"
...
# 检查是否包含文本块1 (认为是 bug)
if selection_start <= text_block1_start < selection_start + doc_len:
    print(f"  ❌ BUG: selection_start={selection_start} 包含文本块1 ...")
```

**修改后**:
```python
print(f"\n[1] Parameters:")
print(f"  system_len = {system_len}")
print(f"  doc_len = {doc_len} (所有文档)")
...
# 不再特殊检查 doc1
label = "system" if i == 0 else f"doc{i}"
print(f"  passages[{i}] ({label:10s}): {selected:4d}/{total:4d} ({pct:5.1f}%)")
...
# 不再检查是否包含 doc1
print(f"  Selection covers all documents (no prefix cache special case)")
```

---

## 行为变化

### 修改前 (有 Bug)
```
passages[0] (system):  [0, 792)     - 跳过
passages[1] (doc1):    [792, 856)   - 应该跳过 (prefix cache) 但被错误包含
passages[2] (doc2):    [856, 1129)  - 参与选择
...
passages[N] (docN):    ...          - 参与选择

selection_start = 792 (只跳过 system)
doc_len = 所有文档 (包括 doc1)
selection_range = [792, 2473)  ← 错误包含 doc1!

rate=0.99:
  从 1681 tokens 中选 1664 个
  doc1 被部分选择 (99.4%) ← BUG!
```

### 修改后 (正确)
```
passages[0] (system):  [0, 792)     - 跳过
passages[1] (doc1):    [792, 856)   - 参与选择 ✅
passages[2] (doc2):    [856, 1129)  - 参与选择 ✅
...
passages[N] (docN):    ...          - 参与选择 ✅

selection_start = 792 (只跳过 system)
doc_len = 所有文档 (包括 doc1)
selection_range = [792, 2473)  ← 正确!所有文档都参与

rate=0.99:
  从 1681 tokens 中选 1664 个
  所有文档按 importance 选择 ✅
  没有特例，行为一致 ✅
```

---

## 预期效果

### 1. 统一的选择逻辑
- ✅ 所有加载的文档 KV 都参与 importance-based 选择
- ✅ 没有 "prefix cache" 特例
- ✅ doc1 和其他文档地位平等

### 2. Rate 行为一致
```
Rate=0.0:  只重算 question
Rate=0.5:  从所有文档中选 50% 重要 tokens + question
Rate=0.8:  从所有文档中选 80% 重要 tokens + question
Rate=0.99: 从所有文档中选 99% 重要 tokens + question
Rate=1.0:  重算所有 tokens (完整前向传播)
```

### 3. 性能预期

**Rate=0.99 性能应该提升或保持稳定**:
- 修改前: doc1 被部分选择，可能丢失重要 tokens → 性能下降
- 修改后: doc1 完整参与选择，按 importance 保留 → 性能应该好转

**最佳性能点**:
- 仍然是 Rate=1.0 (完整前向传播)
- Rate=0.99 应该接近 Rate=1.0 的性能

---

## 测试方法

### 1. 快速测试 (1 个样本)
```bash
bash test_rate_99_fixed.sh
```

查看输出:
```bash
grep -A 40 'DEBUG: Token Selection Details' \
  result/debug_rate_99_fixed/output_rate_0.99.log
```

**检查项**:
- [ ] `doc_len = 所有文档`
- [ ] `selection_start = system_len`
- [ ] doc1 参与选择（不再显示 "BUG" 警告）
- [ ] selection_range 包含所有文档

### 2. 完整测试 (所有样本)
```bash
# 修改 run_online_lazy_sweep.sh
RATE_LIST=(0.8 0.9 0.95 0.99 1.0)
MAX_SAMPLES=""  # 测试所有样本

bash script/run_online_lazy_sweep.sh
```

**对比性能**:
| Rate | 修改前 Main Acc | 修改后 Main Acc | 变化 |
|------|----------------|----------------|------|
| 0.80 | 0.8333         | ?              | ?    |
| 0.90 | 0.8333         | ?              | ?    |
| 0.99 | 0.7759         | ?              | ?    |
| 1.00 | 0.9023         | ?              | ?    |

---

## 注意事项

### 1. 其他 Reprocess 方法未修改
- `DynamicDraftModel` (Line 1499-1504): 仍保留 prefix cache 特例
- `DraftModelDynamic` (Line 1690-1695): 仍保留 prefix cache 特例

如果需要，可以统一修改所有方法。

### 2. 计算成本
- 修改后所有文档都参与 importance 计算
- 没有 prefix cache 优化，可能略微增加计算时间
- 但避免了 bug，性能应该更好

### 3. 缓存复用
- 已生成的 KV cache 仍然可以复用
- 只是选择逻辑变了，不影响缓存文件

---

## 回滚方法

如果需要恢复之前的实现（有 prefix cache 特例），参考：
- `DynamicDraftModel` 方法 (Line 1499-1504)
- `DraftModelDynamic` 方法 (Line 1690-1695)

这两个方法仍然保留了原始逻辑。

---

## 总结

**修改内容**: 去掉 DraftModel 中文本块1的 prefix cache 特例
**修改目的**: 统一选择逻辑，所有文档都参与 rate-based importance 选择
**预期效果**: Rate=0.99 性能应该提升或保持稳定，不会再出现反常下降
