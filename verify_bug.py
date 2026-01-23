#!/usr/bin/env python3
"""
验证 cache_position bug 的快速测试
"""

# 模拟场景
system_len = 100
doc1_len = 200
doc2_len = 200  # missing
doc3_len = 200
question_len = 50

passages_lens = [system_len, doc1_len, doc2_len, doc3_len, question_len]
passages_len_cumsum = []
cumsum = 0
for l in passages_lens:
    cumsum += l
    passages_len_cumsum.append(cumsum)

print("Passages 结构:")
print(f"  System: [0, {passages_len_cumsum[0]})")
print(f"  Doc1: [{passages_len_cumsum[0]}, {passages_len_cumsum[1]})")
print(f"  Doc2 (missing): [{passages_len_cumsum[1]}, {passages_len_cumsum[2]})")
print(f"  Doc3: [{passages_len_cumsum[2]}, {passages_len_cumsum[3]})")
print(f"  Question: [{passages_len_cumsum[3]}, {passages_len_cumsum[4]})")
print(f"  Total passages length: {passages_len_cumsum[-1]}")
print()

# past_key_values 加载阶段（跳过 Doc2）
past_len = system_len + doc1_len + doc3_len
print(f"past_key_values 加载的内容:")
print(f"  System: [0, {system_len})")
print(f"  Doc1: [{system_len}, {system_len + doc1_len})")
print(f"  Doc3: [{system_len + doc1_len}, {past_len})")
print(f"  Total past_len: {past_len}")
print()

# 重算阶段：计算 Doc2 (missing) 的 cache_position
missing_chunks = [(2, 'doc2', None)]  # idx=2 是 Doc2
k_need_index = []

for idx, doc_id, _ in missing_chunks:
    # 当前代码的计算方式
    chunk_start = passages_len_cumsum[idx-1] if idx > 0 else 0
    chunk_end = passages_len_cumsum[idx]
    k_need_index.extend(range(chunk_start, chunk_end))

    print(f"Bug: Doc2 的 cache_position = [{chunk_start}, {chunk_end})")
    print(f"  这会写入 past_key_values 的 [{chunk_start}, {chunk_end}) 位置")
    print(f"  但这个位置实际上是 Doc3 的 KV！")
    print()

    # 正确的计算方式
    correct_start = past_len  # Doc2 应该追加到 past_key_values 的末尾
    correct_end = past_len + doc2_len
    print(f"正确: Doc2 的 cache_position 应该是 [{correct_start}, {correct_end})")
    print(f"  这会追加到 past_key_values 的末尾")
    print()

print("=" * 60)
print("结论：当前代码有严重 bug！")
print("Missing chunks 的 cache_position 基于 passages 的位置计算，")
print("但 past_key_values 中已经跳过了 missing chunks，")
print("导致位置不匹配，missing chunks 会覆盖已加载的文档！")
