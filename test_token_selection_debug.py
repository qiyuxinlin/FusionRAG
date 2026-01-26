#!/usr/bin/env python3
import sys
import os
import torch
import json
from pathlib import Path

# 设置路径
sys.path.insert(0, '/home/shm/document/exp/FusionRAG')
os.chdir('/home/shm/document/exp/FusionRAG')

from transformers import AutoModelForCausalLM, AutoTokenizer
from ktransformers.util.utils import load_kv_and_generate
import numpy as np

print("="*80)
print("Token Selection Debug Test - Rate=0.99")
print("="*80)

# 配置
MODEL_PATH = "/mnt/data/models/Qwen2.5-7B-Instruct"
DRAFT_MODEL_PATH = "/mnt/data/models/Qwen2.5-3B-Instruct"
DATA_PATH = "/home/shm/document/exp/FusionRAG/data/2wikimqa_reflect_optimized.json"
CACHE_PATH = "/mnt/data3/tmp/fusionrag_online_lazy"
DEVICE = "cuda:0"

print(f"\n加载模型...")
print(f"  Main model: {MODEL_PATH}")
print(f"  Draft model: {DRAFT_MODEL_PATH}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 加载数据（只测试第一个样本）
with open(DATA_PATH, 'r') as f:
    data = json.load(f)

sample = data[0]
print(f"\n测试样本:")
print(f"  Question: {sample['question'][:100]}...")
print(f"  Gold docs: {sample.get('gold_docs', 'N/A')}")

# 模拟 passages 构建
# 注意：这里简化处理，实际代码中需要加载真实的 doc_pool
system_prompt = "You are a helpful assistant."
question = sample['question']

# 模拟文档（使用简化版本）
passages_text = [
    system_prompt,  # passages[0]
    "Document 1 content here. " * 20,  # passages[1] - 文本块1
    "Document 2 content here. " * 20,  # passages[2] - 文本块2
    "Document 3 content here. " * 20,  # passages[3] - 文本块3
    "Document 4 content here. " * 20,  # passages[4] - 文本块4
    question  # passages[5]
]

# Tokenize
passages = []
for text in passages_text:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    passages.append(torch.tensor(tokens))

print(f"\n构建的 passages:")
for i, p in enumerate(passages):
    label = "system" if i == 0 else ("question" if i == len(passages)-1 else f"文本块{i}")
    print(f"  passages[{i}] ({label}): {len(p)} tokens")

print(f"\n总输入长度: {sum(len(p) for p in passages)} tokens")

# 关键参数
rate = 0.99
system_len = len(passages[0])
passages_len = [len(p) for p in passages]

# ========== 模拟 DraftModel 的选择逻辑 ==========
print(f"\n{'='*80}")
print(f"模拟 DraftModel 选择逻辑 (BUG 版本)")
print(f"{'='*80}")

text_block1_len = passages_len[1]
doc_len_bug = sum(passages_len[1:-1])  # BUG: 包括 passages[1]
selection_start_bug = system_len  # BUG: 只跳过 system

print(f"\nBUG 版本参数:")
print(f"  text_block1_len = {text_block1_len}")
print(f"  doc_len = {doc_len_bug} (包括文本块1)")
print(f"  selection_start = {selection_start_bug} (只跳过 system)")
print(f"  rate = {rate}")
print(f"  target_tokens = {int(doc_len_bug * rate)}")

# 模拟 attention 分布
np.random.seed(42)
attention_scores = np.random.exponential(0.001, doc_len_bug)
# 添加一些峰值
peaks = np.random.choice(doc_len_bug, 10, replace=False)
for p in peaks:
    attention_scores[p] += 0.01

# 简化的选择逻辑（只用 topk）
target_count = int(doc_len_bug * rate)
sorted_indices = np.argsort(attention_scores)[::-1]
selected_positions = sorted(sorted_indices[:target_count])

# 转换为全局位置
k_need_index_bug = [p + selection_start_bug for p in selected_positions]

print(f"\n选择结果:")
print(f"  选择了 {len(k_need_index_bug)} 个 tokens")
print(f"  范围: [{min(k_need_index_bug)}, {max(k_need_index_bug)}]")

# 分析每个文档的选择情况
cumsum = [0]
for p in passages:
    cumsum.append(cumsum[-1] + len(p))

print(f"\n每个文档被选中的情况:")
k_need_set = set(k_need_index_bug)
for i in range(len(passages) - 1):
    doc_start = cumsum[i]
    doc_end = cumsum[i+1]
    selected = len(k_need_set & set(range(doc_start, doc_end)))
    total = doc_end - doc_start
    pct = selected / total * 100 if total > 0 else 0

    label = "system" if i == 0 else f"文本块{i}"
    status = ""
    if i == 1:  # 文本块1
        if selected == total:
            status = " ✅"
        elif selected > 0:
            status = f" ⚠️ BUG: 部分选择！"
        else:
            status = " ✅"

    print(f"  passages[{i}] ({label:10s}): {selected:4d}/{total:4d} ({pct:5.1f}%){status}")

# ========== 正确版本对比 ==========
print(f"\n{'='*80}")
print(f"正确版本对比")
print(f"{'='*80}")

doc_len_correct = sum(passages_len[2:-1])  # 正确: 从 passages[2] 开始
selection_start_correct = system_len + text_block1_len  # 正确

print(f"\n正确版本参数:")
print(f"  doc_len = {doc_len_correct} (只有文本块2-4)")
print(f"  selection_start = {selection_start_correct} (跳过 system 和文本块1)")
print(f"  target_tokens = {int(doc_len_correct * rate)}")

# 对比
print(f"\n差异对比:")
print(f"  doc_len: {doc_len_bug} (BUG) vs {doc_len_correct} (正确)")
print(f"  差异: {doc_len_bug - doc_len_correct} tokens")
print(f"  target_tokens: {int(doc_len_bug * rate)} (BUG) vs {int(doc_len_correct * rate)} (正确)")
print(f"  差异: {int(doc_len_bug * rate) - int(doc_len_correct * rate)} tokens")

print(f"\n{'='*80}")
print(f"结论")
print(f"{'='*80}")

passages_1_selected = len(k_need_set & set(range(cumsum[1], cumsum[2])))
if passages_1_selected > 0:
    print(f"❌ BUG 确认！")
    print(f"   文本块1有 {passages_1_selected}/{passages_len[1]} tokens 被选中")
    print(f"   文本块1应该完整使用 prefix cache")
    print(f"   但由于 BUG，文本块1被部分包含在选择范围内")
else:
    print(f"✅ 正确！文本块1未被选中")

print(f"\n总输入: {sum(passages_len)} tokens")
print(f"需要重算: {len(k_need_index_bug)} tokens")
print(f"不需重算: {sum(passages_len) - len(k_need_index_bug)} tokens")

