#!/usr/bin/env python3
"""
调试采样逻辑 - 检查实际使用了多少token
"""
import torch
from transformers import AutoTokenizer

model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# 读取文件
with open("/home/shm/document/exp/FusionRAG/example_texts.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

texts = {}
for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if ':' in line and len(line.split(':', 1)) == 2:
        label, text = line.split(':', 1)
        texts[label.strip()] = text.strip()

print("="*80)
print("Token采样分析")
print("="*80)

max_tokens = 50000  # shell脚本中的默认值

for label, text in texts.items():
    tokens = tokenizer.encode(text, add_special_tokens=False)
    seq_len = len(tokens)

    # 模拟采样逻辑
    if max_tokens is not None and seq_len > max_tokens:
        indices = torch.linspace(0, seq_len-1, max_tokens).long()
        actual_count = max_tokens
        print(f"\n{label}:")
        print(f"  原始token数: {seq_len}")
        print(f"  采样后: {actual_count}")
        print(f"  采样范围: {indices[0].item()} 到 {indices[-1].item()}")
        print(f"  ⚠️  被限制了！")
    else:
        actual_count = seq_len
        print(f"\n{label}:")
        print(f"  原始token数: {seq_len}")
        print(f"  采样后: {actual_count}")
        print(f"  ✓ 使用全部token")

print("\n" + "="*80)
print("预期行为:")
print("  - 如果 max_tokens=50000，应该使用全部token")
print("  - AI_Tech2的token数应该是AI_Tech的2倍")
print("="*80)
