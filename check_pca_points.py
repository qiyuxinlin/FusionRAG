#!/usr/bin/env python3
"""
检查PCA可视化中每个文本的实际点数
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
print("文本Token数统计")
print("="*80)

for label, text in texts.items():
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"\n{label}:")
    print(f"  字符数: {len(text)}")
    print(f"  Token数: {len(tokens)}")
    print(f"  文本预览: {text[:100]}...")

print("\n" + "="*80)
print("预期:")
print("  - 如果 max_tokens=50000，两个文本都不会被限制")
print("  - AI_Tech 应该有约 80-90 个点")
print("  - AI_Tech2 应该有约 160-180 个点（是AI_Tech的2倍）")
print("\n但PCA可视化位置可能相近，因为内容相同！")
print("="*80)
