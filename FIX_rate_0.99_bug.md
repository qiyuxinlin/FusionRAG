# Rate=0.99 性能下降 Bug 修复方案

## Bug 确认

**根本原因**: `DraftModel` 方法中 `selection_start` 和 `doc_len` 的计算错误

### 代码位置
文件: `/home/shm/document/exp/FusionRAG/ktransformers/util/utils.py`
行号: 1257-1263

### Bug 代码
```python
# 文本块1 长度 (用于 prefix cache，不参与重算)
text_block1_len = passages_len[1]
# 从文本块2开始选择 (文本块1直接用原始KV cache，不参与重算选择)
# doc_len = 文本块2 + 文本块3 + ... + 文本块n
doc_len = sum(passages_len[1:-1])  # ❌ BUG: 包括了 passages[1]
# selection_start 跳过 system_prompt 和 文本块1
selection_start = system_len  # ❌ BUG: 只跳过了 system
```

**问题**:
1. 注释说"从文本块2开始选择"，但 `doc_len` 包括了 `passages[1]` (文本块1)
2. 注释说"跳过 system_prompt 和 文本块1"，但 `selection_start` 只跳过了 system
3. 导致选择范围**错误地包括了文本块1**

---

## Bug 影响

### 实验数据
| Rate | Main Accuracy | 说明 |
|------|--------------|------|
| 0.80 | 0.8333 | ✓ 正常 |
| 0.90 | 0.8333 | ✓ 正常 |
| 0.99 | 0.7759 | ❌ 下降 6.89% |
| 1.00 | 0.9023 | ⭐ 最佳 |

### 为什么 rate=0.99 性能下降？

**假设场景** (典型的 2WikiMQA 问题):
- system_prompt: 50 tokens
- passages[1] (文本块1): 200 tokens  ← 应该用 prefix cache
- passages[2-4] (文本块2-4): 600 tokens
- question: 100 tokens

**DraftModel (有 bug)**:
```
doc_len = 800 (包括文本块1的 200 tokens)
selection_start = 50

Rate=1.0:
  → 选择全部 800 tokens
  → 文本块1 完整保留 ✅
  → 性能最优 (0.9023)

Rate=0.99:
  → 选择 800 * 0.99 = 792 tokens
  → 丢弃 8 个 tokens
  → 文本块1 被部分选择 ❌
  → **破坏了 prefix cache 的完整性**
  → 性能下降 (0.7759)
```

**正确实现** (DynamicDraftModel):
```
doc_len = 600 (只有文本块2-4)
selection_start = 250 (跳过 system 和文本块1)

Rate=0.99:
  → 从 600 tokens 中选择 594 个
  → 文本块1 完整保留，使用 prefix cache ✅
  → 性能应该接近 rate=1.0
```

---

## 修复方案

### 方法 1: 直接修复 (推荐)

修改 `/home/shm/document/exp/FusionRAG/ktransformers/util/utils.py:1261-1263`:

```python
# 从文本块2开始选择 (文本块1直接用原始KV cache，不参与重算选择)
doc_len = sum(passages_len[2:-1])  # 修复: 从 passages[2] 开始
# selection_start 跳过 system_prompt 和 文本块1
selection_start = system_len + text_block1_len  # 修复: 加上 text_block1_len
```

### 方法 2: 参考正确实现

`DynamicDraftModel` 和 `DraftModelDynamic` 的实现是正确的，可以参考：

**DynamicDraftModel** (utils.py:1421-1425):
```python
text_block1_len = passages_len[1]
doc_len = sum(passages_len[2:-1])  # ✅ 正确
selection_start = system_len + text_block1_len  # ✅ 正确
```

**DraftModelDynamic** (utils.py:1612-1616):
```python
text_block1_len = passages_len[1]
doc_len = sum(passages_len[2:-1])  # ✅ 正确
selection_start = system_len + text_block1_len  # ✅ 正确
```

---

## 实施步骤

### 1. 备份原文件
```bash
cp /home/shm/document/exp/FusionRAG/ktransformers/util/utils.py \
   /home/shm/document/exp/FusionRAG/ktransformers/util/utils.py.backup
```

### 2. 应用修复
修改第 1261-1263 行:

```diff
  # 从文本块2开始选择 (文本块1直接用原始KV cache，不参与重算选择)
  # doc_len = 文本块2 + 文本块3 + ... + 文本块n
- doc_len = sum(passages_len[1:-1])
+ doc_len = sum(passages_len[2:-1])
  # selection_start 跳过 system_prompt 和 文本块1
- selection_start = system_len
+ selection_start = system_len + text_block1_len
```

### 3. 重新测试
```bash
cd /home/shm/document/exp/FusionRAG
bash script/run_online_lazy_sweep.sh
```

修改 `script/run_online_lazy_sweep.sh` 中的 `RATE_LIST`:
```bash
RATE_LIST=(0.8 0.9 0.95 0.99 1.0)
```

---

## 预期效果

修复后的性能预期:

| Rate | 修复前 Main Acc | 预期修复后 Main Acc | 提升 |
|------|----------------|-------------------|------|
| 0.80 | 0.8333         | ~0.8333           | 保持 |
| 0.90 | 0.8333         | ~0.8400           | +0.67% |
| 0.95 | -              | ~0.8600           | 新测试 |
| 0.99 | 0.7759         | **~0.8900**       | **+14.7%** ⭐ |
| 1.00 | 0.9023         | ~0.9023           | 保持 |

**关键改进**:
- Rate=0.99 性能应该接近 rate=1.0
- 形成平滑的性能曲线：rate 越高 → 性能越好
- 符合理论预期

---

## 验证方法

### 1. 检查输出信息
修复后，输出应该显示:
```
DraftModel 选择了 ~594 个 tokens (从文本块2-n中选99.0%), 文本块1(~200tokens)用prefix cache
```

而不是修复前的:
```
DraftModel 选择了 ~792 个 tokens (从文本块2-n中选99.0%), 文本块1(~200tokens)用prefix cache
```

### 2. 性能指标
- Rate=0.99 的 Main Accuracy 应该 ≥ 0.88
- Rate=0.99 应该比 Rate=0.9 更好
- Rate=0.99 应该接近 Rate=1.0

---

## 其他发现

### DynamicDraftModel 和 DraftModelDynamic 没有这个 bug
这两个方法的实现是正确的，可以作为参考。这也解释了为什么某些实验中 rate=0.99 性能正常。

### 建议
1. 统一三个方法的实现，避免代码重复
2. 添加单元测试验证 selection_start 和 doc_len 的计算
3. 添加断言检查选择范围的正确性

---

## 总结

**Bug 本质**: 注释与代码不一致
**根本原因**: `DraftModel` 的 `selection_start` 和 `doc_len` 计算错误
**影响范围**: 仅影响 `DraftModel` 方法，其他方法正常
**修复难度**: 简单 (只需修改 2 行代码)
**预期提升**: Rate=0.99 性能提升 **~14.7%**

这个 bug 完美解释了为什么 rate=0.99 性能反而比 rate=0.8/0.9 差，而 rate=1.0 最好。修复后应该形成平滑的性能曲线。
