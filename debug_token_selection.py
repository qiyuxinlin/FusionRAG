#!/usr/bin/env python3
"""
断点分析脚本：检查 rate=0.99 下的 token 选择详情

功能：
1. 在关键位置插入 debug 输出
2. 分析 k_need_index 的组成
3. 对比原始输入 token 长度
4. 可视化选择的 tokens 分布
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/shm/document/exp/FusionRAG')

def create_debug_patch():
    """
    创建一个临时的 debug 版本，添加详细输出
    """
    patch_code = """
# ==================== DEBUG PATCH START ====================
# 在 utils.py 的 DraftModel 方法中，line 1396 之后添加以下代码：

print(f"\\n{'='*80}")
print(f"DEBUG: DraftModel Token Selection Analysis")
print(f"{'='*80}")

# 1. passages 结构分析
print(f"\\n[1] Passages 结构:")
print(f"  len(passages) = {len(passages)}")
for i, p in enumerate(passages):
    p_len = p.shape[0] if hasattr(p, 'shape') else len(p)
    if i == 0:
        print(f"  passages[{i}] (system):      {p_len} tokens")
    elif i == len(passages) - 1:
        print(f"  passages[{i}] (question):    {p_len} tokens")
    else:
        print(f"  passages[{i}] (文本块{i}):     {p_len} tokens")

# 2. 选择参数
print(f"\\n[2] 选择参数:")
print(f"  system_len = {system_len}")
print(f"  text_block1_len = {text_block1_len}")
print(f"  doc_len = {doc_len}")
print(f"  selection_start = {selection_start}")
print(f"  rate = {rate}")
print(f"  target tokens = {int(doc_len * rate)}")

# 3. 选择范围
print(f"\\n[3] 选择范围:")
print(f"  selection_start = {selection_start}")
print(f"  selection_start + doc_len = {selection_start + doc_len}")
cumsum = [0]
for i, p in enumerate(passages):
    p_len = p.shape[0] if hasattr(p, 'shape') else len(p)
    cumsum.append(cumsum[-1] + p_len)
print(f"\\n  Cumulative positions:")
for i in range(len(passages)):
    label = "system" if i == 0 else ("question" if i == len(passages)-1 else f"文本块{i}")
    print(f"    passages[{i}] ({label}): [{cumsum[i]}, {cumsum[i+1]})")

print(f"\\n  ⚠️  Selection range [{selection_start}, {selection_start + doc_len})")
print(f"      covers passages[{selection_start//100}] to passages[{(selection_start + doc_len)//100}]")

# 4. k_need_index 详细分析
print(f"\\n[4] k_need_index 分析:")
print(f"  len(k_need_index) = {len(k_need_index)}")
if isinstance(k_need_index, torch.Tensor):
    k_need_list = k_need_index.tolist()
else:
    k_need_list = list(k_need_index)

print(f"  First 20: {k_need_list[:20]}")
print(f"  Last 20:  {k_need_list[-20:]}")
print(f"  Min: {min(k_need_list)}, Max: {max(k_need_list)}")

# 5. 分析每个文档被选中的 tokens
print(f"\\n[5] 每个文档被选中的 tokens:")
k_need_set = set(k_need_list)
for i in range(len(passages) - 1):  # 不包括 question
    doc_start = cumsum[i]
    doc_end = cumsum[i+1]
    doc_tokens_in_selection = len(k_need_set & set(range(doc_start, doc_end)))
    doc_total = doc_end - doc_start
    percentage = doc_tokens_in_selection / doc_total * 100 if doc_total > 0 else 0

    label = "system" if i == 0 else f"文本块{i}"
    status = ""
    if i == 1:
        if doc_tokens_in_selection == doc_total:
            status = " ✅ 完整选择"
        elif doc_tokens_in_selection > 0:
            status = f" ⚠️ 部分选择 (应该用 prefix cache)"
        else:
            status = " ✅ 未选择 (使用 prefix cache)"

    print(f"  passages[{i}] ({label:12s}): {doc_tokens_in_selection:4d}/{doc_total:4d} ({percentage:5.1f}%){status}")

# 6. BUG 验证
print(f"\\n[6] BUG 验证:")
passages_1_start = cumsum[1]
passages_1_end = cumsum[2]
passages_1_in_selection = len(k_need_set & set(range(passages_1_start, passages_1_end)))

if passages_1_in_selection > 0:
    print(f"  ❌ BUG 存在！文本块1有 {passages_1_in_selection} 个 tokens 被选中")
    print(f"     文本块1应该完整使用 prefix cache，不应该参与选择")
else:
    print(f"  ✅ 正确！文本块1没有被选中，使用 prefix cache")

# 7. 总 token 长度对比
print(f"\\n[7] Token 长度对比:")
total_input_len = sum([p.shape[0] if hasattr(p, 'shape') else len(p) for p in passages])
print(f"  原始输入总长度: {total_input_len} tokens")
print(f"  k_need_index 长度: {len(k_need_index)} tokens")
print(f"  差异: {total_input_len - len(k_need_index)} tokens 不需要重算")

# 8. 可视化选择分布
print(f"\\n[8] 选择分布可视化 (每个 '█' 代表 10 个 tokens):")
for i in range(len(passages) - 1):
    doc_start = cumsum[i]
    doc_end = cumsum[i+1]
    doc_len_i = doc_end - doc_start

    # 创建可视化
    viz = []
    for pos in range(doc_start, doc_end, 10):
        block_selected = len(k_need_set & set(range(pos, min(pos+10, doc_end))))
        if block_selected >= 5:
            viz.append('█')
        elif block_selected > 0:
            viz.append('▓')
        else:
            viz.append('░')

    label = "system" if i == 0 else f"doc{i}"
    print(f"  {label:8s} [{''.join(viz)}]")

print(f"  Legend: █=selected(5+/10), ▓=partial(1-4/10), ░=not selected")

print(f"\\n{'='*80}")
print(f"END DEBUG")
print(f"{'='*80}\\n")

# ==================== DEBUG PATCH END ====================
"""
    return patch_code


def create_instrumented_script():
    """
    创建一个带有详细 debug 输出的测试脚本
    """
    script_content = """#!/usr/bin/env python3
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

print(f"\\n加载模型...")
print(f"  Main model: {MODEL_PATH}")
print(f"  Draft model: {DRAFT_MODEL_PATH}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 加载数据（只测试第一个样本）
with open(DATA_PATH, 'r') as f:
    data = json.load(f)

sample = data[0]
print(f"\\n测试样本:")
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

print(f"\\n构建的 passages:")
for i, p in enumerate(passages):
    label = "system" if i == 0 else ("question" if i == len(passages)-1 else f"文本块{i}")
    print(f"  passages[{i}] ({label}): {len(p)} tokens")

print(f"\\n总输入长度: {sum(len(p) for p in passages)} tokens")

# 关键参数
rate = 0.99
system_len = len(passages[0])
passages_len = [len(p) for p in passages]

# ========== 模拟 DraftModel 的选择逻辑 ==========
print(f"\\n{'='*80}")
print(f"模拟 DraftModel 选择逻辑 (BUG 版本)")
print(f"{'='*80}")

text_block1_len = passages_len[1]
doc_len_bug = sum(passages_len[1:-1])  # BUG: 包括 passages[1]
selection_start_bug = system_len  # BUG: 只跳过 system

print(f"\\nBUG 版本参数:")
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

print(f"\\n选择结果:")
print(f"  选择了 {len(k_need_index_bug)} 个 tokens")
print(f"  范围: [{min(k_need_index_bug)}, {max(k_need_index_bug)}]")

# 分析每个文档的选择情况
cumsum = [0]
for p in passages:
    cumsum.append(cumsum[-1] + len(p))

print(f"\\n每个文档被选中的情况:")
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
print(f"\\n{'='*80}")
print(f"正确版本对比")
print(f"{'='*80}")

doc_len_correct = sum(passages_len[2:-1])  # 正确: 从 passages[2] 开始
selection_start_correct = system_len + text_block1_len  # 正确

print(f"\\n正确版本参数:")
print(f"  doc_len = {doc_len_correct} (只有文本块2-4)")
print(f"  selection_start = {selection_start_correct} (跳过 system 和文本块1)")
print(f"  target_tokens = {int(doc_len_correct * rate)}")

# 对比
print(f"\\n差异对比:")
print(f"  doc_len: {doc_len_bug} (BUG) vs {doc_len_correct} (正确)")
print(f"  差异: {doc_len_bug - doc_len_correct} tokens")
print(f"  target_tokens: {int(doc_len_bug * rate)} (BUG) vs {int(doc_len_correct * rate)} (正确)")
print(f"  差异: {int(doc_len_bug * rate) - int(doc_len_correct * rate)} tokens")

print(f"\\n{'='*80}")
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

print(f"\\n总输入: {sum(passages_len)} tokens")
print(f"需要重算: {len(k_need_index_bug)} tokens")
print(f"不需重算: {sum(passages_len) - len(k_need_index_bug)} tokens")

"""
    return script_content


def main():
    """
    主函数：创建分析工具
    """
    print("="*80)
    print("创建 Token Selection Debug 工具")
    print("="*80)

    # 1. 创建 debug patch 说明
    patch_file = "/home/shm/document/exp/FusionRAG/DEBUG_PATCH.txt"
    with open(patch_file, 'w') as f:
        f.write(create_debug_patch())
    print(f"\n✓ 创建 debug patch 说明: {patch_file}")
    print("  可以手动将此代码插入 utils.py:1396 之后")

    # 2. 创建独立测试脚本
    test_script = "/home/shm/document/exp/FusionRAG/test_token_selection_debug.py"
    with open(test_script, 'w') as f:
        f.write(create_instrumented_script())
    os.chmod(test_script, 0o755)
    print(f"\n✓ 创建独立测试脚本: {test_script}")
    print("  运行: python3 test_token_selection_debug.py")

    # 3. 创建一个更简单的 inline 注入版本
    print("\n" + "="*80)
    print("方案 1: 修改 utils.py 添加 debug 输出 (推荐)")
    print("="*80)
    print("""
在 /home/shm/document/exp/FusionRAG/ktransformers/util/utils.py 的 1396 行之后
(即 k_need_index = torch.tensor(selected_indices, device='cpu') 之后)
添加以下代码:

```python
# ==================== DEBUG START ====================
if rate >= 0.99:  # 只在 rate 接近 1.0 时输出
    print(f"\\n{'='*60}")
    print(f"DEBUG: Token Selection Details (rate={rate})")
    print(f"{'='*60}")

    # 参数
    print(f"\\nParameters:")
    print(f"  system_len = {system_len}")
    print(f"  text_block1_len = {text_block1_len}")
    print(f"  doc_len = {doc_len}")
    print(f"  selection_start = {selection_start}")
    print(f"  rate = {rate}")

    # k_need_index 分析
    k_list = k_need_index.tolist() if isinstance(k_need_index, torch.Tensor) else list(k_need_index)
    print(f"\\nk_need_index:")
    print(f"  Length: {len(k_list)}")
    print(f"  Range: [{min(k_list)}, {max(k_list)}]")

    # 每个文档的选择情况
    cumsum = [sum([passages[j].shape[0] for j in range(i+1)]) for i in range(len(passages))]
    k_set = set(k_list)

    print(f"\\nPer-document selection:")
    for i in range(len(passages)-1):
        start = cumsum[i-1] if i > 0 else 0
        end = cumsum[i]
        selected = len(k_set & set(range(start, end)))
        total = end - start
        pct = selected/total*100 if total > 0 else 0

        label = "sys" if i == 0 else f"doc{i}"
        warn = " ⚠️ BUG!" if i == 1 and selected > 0 and selected < total else ""
        print(f"  passages[{i}] ({label}): {selected:4d}/{total:4d} ({pct:5.1f}%){warn}")

    print(f"{'='*60}\\n")
# ==================== DEBUG END ====================
```
    """)

    print("\n" + "="*80)
    print("方案 2: 运行简化测试脚本")
    print("="*80)
    print(f"\n运行命令:")
    print(f"  python3 {test_script}")

    print("\n" + "="*80)
    print("方案 3: 运行实际实验并查看输出")
    print("="*80)
    print("""
修改 run_online_lazy_sweep.sh:
  RATE_LIST=(0.99)
  MAX_SAMPLES="1"  # 只测试 1 个样本

然后运行:
  bash script/run_online_lazy_sweep.sh 2>&1 | tee debug_output.log

查看 debug_output.log 中的 "DraftModel 选择了 X 个 tokens" 信息
    """)


if __name__ == '__main__':
    main()
