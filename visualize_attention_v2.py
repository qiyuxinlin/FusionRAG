"""
可视化对比 Layer 23 和 Layer 35 的 attention 分布 (改进版)
- 颜色区分更明显
- 精确定位答案位置
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def compute_attention_for_layers(model, input_ids, query_start, device, config, target_layers):
    """只计算指定层的 attention"""
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    seq_len = input_ids.shape[1]
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    
    layer_attention = {}
    
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        cos, sin = model.model.rotary_emb(hidden_states, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            bsz, q_len, _ = hidden_states.size()
            
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)
            
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)
            
            n_rep = num_heads // num_kv_heads
            key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
            value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)
            
            if layer_idx in target_layers:
                query_states_subset = query_states[:, :, query_start:, :]
                attn_weights = torch.matmul(
                    query_states_subset.float(),
                    key_states_expanded.float().transpose(2, 3)
                ) / (head_dim ** 0.5)
                
                query_positions = torch.arange(query_start, seq_len, device=device)
                key_positions = torch.arange(seq_len, device=device)
                causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
                
                layer_attention[layer_idx] = attn_weights[0].cpu().numpy()
                
                attn_output_subset = torch.matmul(attn_weights.to(value_states_expanded.dtype), value_states_expanded)
                
                if query_start > 0:
                    attn_output_prefix = F.scaled_dot_product_attention(
                        query_states[:, :, :query_start, :],
                        key_states_expanded[:, :, :query_start, :],
                        value_states_expanded[:, :, :query_start, :],
                        is_causal=True
                    )
                    attn_output = torch.cat([attn_output_prefix, attn_output_subset], dim=2)
                else:
                    attn_output = attn_output_subset
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states_expanded, value_states_expanded, is_causal=True
                )
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)
            
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
    
    return layer_attention


def find_answer_token_positions(tokenizer, doc_token_ids, answer_text):
    """精确找到答案在文档 tokens 中的位置"""
    # 解码整个文档
    full_doc_text = tokenizer.decode(doc_token_ids)
    
    # 在文档中找答案的字符位置
    answer_lower = answer_text.lower()
    doc_lower = full_doc_text.lower()
    
    # 尝试找完整答案
    char_start = doc_lower.find(answer_lower)
    
    # 如果找不到完整答案，尝试找核心部分 "National Cycle Network"
    if char_start == -1:
        core_answer = "national cycle network"
        char_start = doc_lower.find(core_answer)
        if char_start != -1:
            char_end = char_start + len(core_answer)
            answer_text = core_answer
        else:
            return [], "Answer not found in document"
    else:
        char_end = char_start + len(answer_text)
    
    # 将字符位置映射到 token 位置
    # 逐个 token 解码，累计字符长度
    answer_token_positions = []
    current_char_pos = 0
    
    for token_idx, token_id in enumerate(doc_token_ids):
        token_text = tokenizer.decode([token_id])
        token_start = current_char_pos
        token_end = current_char_pos + len(token_text)
        
        # 检查这个 token 是否与答案区域重叠
        if token_start < char_end and token_end > char_start:
            answer_token_positions.append(token_idx)
        
        current_char_pos = token_end
    
    # 获取答案对应的文本
    if answer_token_positions:
        answer_tokens_text = tokenizer.decode([doc_token_ids[i] for i in answer_token_positions])
    else:
        answer_tokens_text = ""
    
    return answer_token_positions, answer_tokens_text


def main():
    device = "cuda:0"
    model_path = "/mnt/data/models/Qwen2.5-3B-Instruct"
    data_path = "/mnt/data/wjh/FusionRAG/result_reflect.json"
    
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    
    print("Loading data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # System prompt
    prompt_config_path = "./config/dataset2prompt_few-shot.json"
    with open(prompt_config_path, 'r', encoding='utf-8') as f:
        prompt_config = json.load(f)
    system_text = prompt_config["system_prompt"]["Qwen2.5"]["2wikimqa"]
    system_tokens = tokenizer.encode(system_text, add_special_tokens=True)
    system_len = len(system_tokens)
    
    # 取第一个样本
    sample = data[0]
    intermediate_context = sample.get('intermediate_context', [])
    
    # 收集文档
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
    
    # 第一个子问题
    sub_q = intermediate_context[0]
    query = sub_q['query']
    answer = sub_q['answer']
    if query.startswith("Intermediate query"):
        colon_pos = query.find(":")
        if colon_pos != -1:
            query = query[colon_pos + 1:].strip()
    if answer.startswith("Intermediate answer"):
        colon_pos = answer.find(":")
        if colon_pos != -1:
            answer = answer[colon_pos + 1:].strip()
    
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    
    # 获取子问题使用的文档
    docs = sub_q.get('retrieve docs', [])
    doc_chunk_ids = [doc_to_idx[doc] for doc in docs if doc in doc_to_idx]
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]
    
    # 构建完整输入
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)
    
    system_tensor = torch.tensor(system_tokens, dtype=torch.long)
    all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    input_ids = torch.cat(all_tokens).unsqueeze(0)
    
    total_len = input_ids.shape[1]
    doc_len = sum(len(t) for t in sub_q_doc_tensors)
    query_start = total_len - len(question_tokens)
    doc_start = system_len
    doc_end = system_len + doc_len
    
    print(f"Input: {total_len} tokens, doc region: [{doc_start}, {doc_end})")
    
    # 精确找到答案位置
    doc_token_ids = input_ids[0, doc_start:doc_end].tolist()
    answer_positions, answer_tokens_text = find_answer_token_positions(tokenizer, doc_token_ids, "National Cycle Network")
    
    print(f"\nAnswer location:")
    print(f"  Token positions: {answer_positions}")
    print(f"  Decoded text: '{answer_tokens_text}'")
    print(f"  Number of tokens: {len(answer_positions)}")
    
    # 计算 Layer 23 和 Layer 35 的 attention
    print("\nComputing attention for Layer 23 and Layer 35...")
    target_layers = [23, 35]
    layer_attention = compute_attention_for_layers(model, input_ids, query_start, device, config, target_layers)
    
    # 提取 query→doc attention
    layer_23_attn = layer_attention[23][:, :, doc_start:doc_end].mean(axis=(0, 1))
    layer_35_attn = layer_attention[35][:, :, doc_start:doc_end].mean(axis=(0, 1))
    
    # 创建可视化 - 使用更明显的颜色区分
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 1. Layer 23 attention 分布 (蓝色)
    ax1 = axes[0]
    ax1.bar(range(len(layer_23_attn)), layer_23_attn, color='steelblue', alpha=0.8, width=1.0, label='Attention')
    # 标记答案位置 - 用黑色粗线
    for pos in answer_positions:
        if pos < len(layer_23_attn):
            ax1.axvline(x=pos, color='black', alpha=0.8, linewidth=2)
    # 添加答案区域背景
    if answer_positions:
        ax1.axvspan(min(answer_positions)-0.5, max(answer_positions)+0.5, alpha=0.2, color='yellow', label='Answer region')
    ax1.set_title(f'Layer 23 (Entropy=4.75, LOWEST) - Black lines = Answer tokens "{answer_tokens_text}"', fontsize=12)
    ax1.set_xlabel('Document Token Position')
    ax1.set_ylabel('Attention Score')
    ax1.set_xlim(0, len(layer_23_attn))
    ax1.legend(loc='upper right')
    
    # 2. Layer 35 attention 分布 (绿色)
    ax2 = axes[1]
    ax2.bar(range(len(layer_35_attn)), layer_35_attn, color='seagreen', alpha=0.8, width=1.0, label='Attention')
    for pos in answer_positions:
        if pos < len(layer_35_attn):
            ax2.axvline(x=pos, color='black', alpha=0.8, linewidth=2)
    if answer_positions:
        ax2.axvspan(min(answer_positions)-0.5, max(answer_positions)+0.5, alpha=0.2, color='yellow', label='Answer region')
    ax2.set_title(f'Layer 35 (Entropy=6.55, HIGHEST) - Black lines = Answer tokens', fontsize=12)
    ax2.set_xlabel('Document Token Position')
    ax2.set_ylabel('Attention Score')
    ax2.set_xlim(0, len(layer_35_attn))
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/mnt/data/wjh/FusionRAG/attention_comparison_v2.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to /mnt/data/wjh/FusionRAG/attention_comparison_v2.png")
    
    # 统计分析
    answer_set = set(answer_positions)
    
    layer_23_answer_attn = [layer_23_attn[i] for i in answer_positions if i < len(layer_23_attn)]
    layer_23_other_attn = [layer_23_attn[i] for i in range(len(layer_23_attn)) if i not in answer_set]
    layer_35_answer_attn = [layer_35_attn[i] for i in answer_positions if i < len(layer_35_attn)]
    layer_35_other_attn = [layer_35_attn[i] for i in range(len(layer_35_attn)) if i not in answer_set]
    
    # Top-k 分析
    k = int(len(layer_23_attn) * 0.3)  # top 30%
    layer_23_top_k = set(np.argsort(layer_23_attn)[-k:])
    layer_35_top_k = set(np.argsort(layer_35_attn)[-k:])
    
    layer_23_answer_in_top = len(answer_set & layer_23_top_k)
    layer_35_answer_in_top = len(answer_set & layer_35_top_k)
    
    print(f"\n{'='*60}")
    print("Statistics")
    print(f"{'='*60}")
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    print(f"Answer tokens: '{answer_tokens_text}' ({len(answer_positions)} tokens)")
    
    print(f"\nLayer 23 (Low Entropy = 4.75):")
    print(f"  - Avg attention on ANSWER tokens: {np.mean(layer_23_answer_attn):.6f}")
    print(f"  - Avg attention on OTHER tokens:  {np.mean(layer_23_other_attn):.6f}")
    print(f"  - Ratio (answer/other): {np.mean(layer_23_answer_attn)/np.mean(layer_23_other_attn):.2f}x")
    print(f"  - Answer tokens in top-{k} (30%): {layer_23_answer_in_top}/{len(answer_positions)}")
    
    print(f"\nLayer 35 (High Entropy = 6.55):")
    print(f"  - Avg attention on ANSWER tokens: {np.mean(layer_35_answer_attn):.6f}")
    print(f"  - Avg attention on OTHER tokens:  {np.mean(layer_35_other_attn):.6f}")
    print(f"  - Ratio (answer/other): {np.mean(layer_35_answer_attn)/np.mean(layer_35_other_attn):.2f}x")
    print(f"  - Answer tokens in top-{k} (30%): {layer_35_answer_in_top}/{len(answer_positions)}")
    
    # 打印 Layer 23 attention 最高的 tokens
    print(f"\n{'='*60}")
    print("Layer 23 Top-10 Attention Tokens")
    print(f"{'='*60}")
    top_10_indices = np.argsort(layer_23_attn)[-10:][::-1]
    for rank, idx in enumerate(top_10_indices):
        token_text = tokenizer.decode([doc_token_ids[idx]])
        is_answer = "★ ANSWER" if idx in answer_set else ""
        print(f"  {rank+1}. Position {idx}: attention={layer_23_attn[idx]:.6f}, token='{token_text}' {is_answer}")


if __name__ == "__main__":
    main()
