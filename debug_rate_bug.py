#!/usr/bin/env python3
"""
验证 DraftModel 中 selection_start 的 bug

问题：
1. 代码注释说"从文本块2-n中选"，文本块1用prefix cache
2. 但实际 selection_start = system_len 只跳过了 system
3. doc_len = sum(passages_len[1:-1]) 包括了文本块1

这导致：
- selection_start 到 selection_start + doc_len 实际上包括了文本块1
- 而注释说要跳过文本块1
"""

print("="*80)
print("DraftModel selection_start Bug 分析")
print("="*80)

# 模拟一个典型的 passages 结构
print("\n假设 passages 结构:")
print("-" * 80)
passages_len = [50, 200, 200, 200, 200, 100]  # [system, doc1, doc2, doc3, doc4, question]
print(f"passages[0] (system):   {passages_len[0]} tokens")
print(f"passages[1] (文本块1):  {passages_len[1]} tokens")
print(f"passages[2] (文本块2):  {passages_len[2]} tokens")
print(f"passages[3] (文本块3):  {passages_len[3]} tokens")
print(f"passages[4] (文本块4):  {passages_len[4]} tokens")
print(f"passages[5] (question): {passages_len[5]} tokens")

print("\n当前代码的逻辑 (utils.py:1258-1263):")
print("-" * 80)
system_len = passages_len[0]
text_block1_len = passages_len[1]
doc_len = sum(passages_len[1:-1])  # BUG: 包括了文本块1
selection_start = system_len  # BUG: 只跳过了 system

print(f"system_len = {system_len}")
print(f"text_block1_len = {text_block1_len}")
print(f"doc_len = sum(passages_len[1:-1]) = {doc_len}")
print(f"selection_start = system_len = {selection_start}")

print(f"\n选择范围:")
print(f"  selection_start = {selection_start}")
print(f"  selection_start + doc_len = {selection_start + doc_len}")
print(f"  → 实际范围: [{selection_start}, {selection_start + doc_len})")

print(f"\nBUG: 这个范围包括了:")
cumsum = [sum(passages_len[:i+1]) for i in range(len(passages_len))]
print(f"  passages[0] (system):   [0, {cumsum[0]})")
print(f"  passages[1] (文本块1):  [{cumsum[0]}, {cumsum[1]}) ⚠️ 应该跳过但被包括了！")
print(f"  passages[2] (文本块2):  [{cumsum[1]}, {cumsum[2]})")
print(f"  passages[3] (文本块3):  [{cumsum[2]}, {cumsum[3]})")
print(f"  passages[4] (文本块4):  [{cumsum[3]}, {cumsum[4]})")

print("\n"+ "="*80)
print("正确的实现应该是:")
print("="*80)
correct_selection_start = system_len + text_block1_len
correct_doc_len = sum(passages_len[2:-1])

print(f"selection_start = system_len + text_block1_len = {correct_selection_start}")
print(f"doc_len = sum(passages_len[2:-1]) = {correct_doc_len}")
print(f"\n选择范围: [{correct_selection_start}, {correct_selection_start + correct_doc_len})")
print(f"  → 这样就只包括文本块2-4，跳过文本块1")

print("\n" + "="*80)
print("这个 bug 对 rate=0.99 的影响:")
print("="*80)

print("\n当 rate=0.99 时:")
print(f"  目标选择: {correct_doc_len} * 0.99 = {int(correct_doc_len * 0.99)} tokens (从文本块2-4)")
print(f"  但实际 doc_len = {doc_len}")
print(f"  实际选择: {doc_len} * 0.99 = {int(doc_len * 0.99)} tokens")
print(f"  差异: {int(doc_len * 0.99) - int(correct_doc_len * 0.99)} tokens")

print(f"\n问题:")
print(f"  1. 文本块1的 {text_block1_len} 个 tokens 被错误地包含在选择范围中")
print(f"  2. smart_query_selection 会从文本块1中选择一些 tokens")
print(f"  3. 但这些 tokens 可能不是最重要的（因为文本块1应该用 prefix cache）")
print(f"  4. 导致选择质量下降，性能变差")

print("\n" + "="*80)
print("为什么 rate=1.0 没问题？")
print("="*80)

print("""
当 rate=1.0 时:
  1. smart_query_selection 会选择所有 doc_len 个 tokens
  2. 虽然范围包括了文本块1，但 rate=1.0 意味着全选
  3. 所以文本块1的所有 tokens 都被选中
  4. 结果等价于完整前向传播
  5. 性能达到最优

当 rate=0.99 时:
  1. smart_query_selection 只选择 99% 的 tokens
  2. 那 1% 被丢弃的 tokens 可能来自文本块1
  3. 但文本块1本应该用 prefix cache，不应该被部分选择
  4. 部分选择导致文本块1的信息不完整
  5. 性能下降
""")

print("\n" + "="*80)
print("验证假设：检查实际实验日志")
print("="*80)

print("""
查看实验输出中的 "DraftModel 选择了 X 个 tokens" 信息:

如果 bug 存在，应该看到:
  rate=0.99: 选择了 ~594 tokens (从 600 中选 99%)
  rate=1.0:  选择了 ~800 tokens (但实际应该是 600)

如果 bug 不存在，应该看到:
  rate=0.99: 选择了 ~396 tokens (从 400 中选 99%)
  rate=1.0:  选择了 ~400 tokens
""")

print("\n" + "="*80)
print("修复方案")
print("="*80)

print("""
方案 1: 修复 selection_start 和 doc_len (推荐)
--------------------------------------------
在 utils.py:1258-1263 处修改:

```python
# 文本块1 长度 (用于 prefix cache，不参与重算)
text_block1_len = passages_len[1]

# 从文本块2开始选择
doc_len = sum(passages_len[2:-1])  # 修改: 从 passages[2] 开始

# selection_start 跳过 system_prompt 和 文本块1
selection_start = system_len + text_block1_len  # 修改: 加上 text_block1_len
```

方案 2: 明确文本块1也参与选择
----------------------------
如果文本块1也要参与重算选择，则修改注释和打印信息:

```python
# 所有文档都参与选择
doc_len = sum(passages_len[1:-1])  # 包括文本块1
selection_start = system_len  # 只跳过 system

# 修改打印信息
print(f"DraftModel 选择了 {len(k_need_index)} 个 tokens "
      f"(从所有文档中选 {len(k_need_index)/doc_len*100:.1f}%)")
```

推荐使用方案 1，因为:
1. prefix cache 的目的就是避免重算第一个文档
2. 减少计算量
3. 保持代码注释和实现一致
""")
