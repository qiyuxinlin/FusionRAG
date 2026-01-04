"""检查高 attention 位置对应的文本内容"""
import torch
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/models/Qwen2.5-3B-Instruct", trust_remote_code=True)

with open('/mnt/data/wjh/FusionRAG/result_reflect.json', 'r') as f:
    data = json.load(f)

# System prompt
with open('./config/dataset2prompt_few-shot.json', 'r', encoding='utf-8') as f:
    prompt_config = json.load(f)
system_text = prompt_config["system_prompt"]["Qwen2.5"]["2wikimqa"]
system_tokens = tokenizer.encode(system_text, add_special_tokens=True)
system_len = len(system_tokens)

sample = data[0]
intermediate_context = sample.get('intermediate_context', [])

question_docs = []
doc_to_idx = {}
for sub_q in intermediate_context:
    docs = sub_q.get('retrieve docs', [])
    for doc in docs:
        if doc not in doc_to_idx:
            question_docs.append(doc)
            doc_to_idx[doc] = len(question_docs)

# 构建文档 tokens
doc_tensors = []
for doc in question_docs:
    doc_text = f"Document: {doc}\n"
    doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
    doc_tensors.append(torch.tensor(doc_tokens, dtype=torch.long))

sub_q = intermediate_context[0]
docs = sub_q.get('retrieve docs', [])
doc_chunk_ids = [doc_to_idx[doc] for doc in docs if doc in doc_to_idx]
sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

# 合并文档 tokens
doc_token_ids = torch.cat(sub_q_doc_tensors).tolist()

# Layer 23 Top attention 位置
top_positions = [488, 1278, 1220, 1312, 1179, 1287, 1221, 1174, 1230, 1178, 15, 16, 17]

print("=" * 80)
print("高 Attention 位置对应的文本内容")
print("=" * 80)

for pos in top_positions:
    if pos < len(doc_token_ids):
        # 获取上下文（前后各10个token）
        start = max(0, pos - 10)
        end = min(len(doc_token_ids), pos + 10)
        
        context_tokens = doc_token_ids[start:end]
        context_text = tokenizer.decode(context_tokens)
        
        # 标记当前token
        current_token = tokenizer.decode([doc_token_ids[pos]])
        
        print(f"\nPosition {pos}:")
        print(f"  Token: '{current_token}'")
        print(f"  Context: ...{context_text}...")

# 检查答案在文档中的所有出现
print("\n" + "=" * 80)
print("答案 'National Cycle Network' 在文档中的所有出现")
print("=" * 80)

full_doc_text = tokenizer.decode(doc_token_ids)
search_term = "National Cycle Network"
start_pos = 0
occurrences = []

while True:
    pos = full_doc_text.lower().find(search_term.lower(), start_pos)
    if pos == -1:
        break
    occurrences.append(pos)
    start_pos = pos + 1

print(f"Found {len(occurrences)} occurrences at character positions: {occurrences}")

# 找到每个出现对应的 token 位置
for i, char_pos in enumerate(occurrences):
    # 将字符位置映射到 token 位置
    current_char = 0
    for token_idx, token_id in enumerate(doc_token_ids):
        token_text = tokenizer.decode([token_id])
        if current_char <= char_pos < current_char + len(token_text):
            context_start = max(0, token_idx - 5)
            context_end = min(len(doc_token_ids), token_idx + 10)
            context = tokenizer.decode(doc_token_ids[context_start:context_end])
            print(f"\nOccurrence {i+1} at token position ~{token_idx}:")
            print(f"  Context: ...{context}...")
            break
        current_char += len(token_text)
