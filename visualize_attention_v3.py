"""可视化 - 标记所有答案出现位置"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def compute_attention_for_layers(model, input_ids, query_start, device, config, target_layers):
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
                attn_weights = torch.matmul(query_states_subset.float(), key_states_expanded.float().transpose(2, 3)) / (head_dim ** 0.5)
                
                query_positions = torch.arange(query_start, seq_len, device=device)
                key_positions = torch.arange(seq_len, device=device)
                causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
                
                layer_attention[layer_idx] = attn_weights[0].cpu().numpy()
                attn_output_subset = torch.matmul(attn_weights.to(value_states_expanded.dtype), value_states_expanded)
                
                if query_start > 0:
                    attn_output_prefix = F.scaled_dot_product_attention(
                        query_states[:, :, :query_start, :], key_states_expanded[:, :, :query_start, :],
                        value_states_expanded[:, :, :query_start, :], is_causal=True)
                    attn_output = torch.cat([attn_output_prefix, attn_output_subset], dim=2)
                else:
                    attn_output = attn_output_subset
            else:
                attn_output = F.scaled_dot_product_attention(query_states, key_states_expanded, value_states_expanded, is_causal=True)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)
            
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
    
    return layer_attention

def find_all_answer_positions(tokenizer, doc_token_ids, answer_text):
    """找到答案在文档中的所有出现位置"""
    full_doc_text = tokenizer.decode(doc_token_ids)
    answer_lower = answer_text.lower()
    doc_lower = full_doc_text.lower()
    
    all_positions = []
    start_search = 0
    
    while True:
        char_pos = doc_lower.find(answer_lower, start_search)
        if char_pos == -1:
            break
        
        char_end = char_pos + len(answer_text)
        
        # 映射到 token 位置
        current_char = 0
        occurrence_tokens = []
        for token_idx, token_id in enumerate(doc_token_ids):
            token_text = tokenizer.decode([token_id])
            token_start = current_char
            token_end = current_char + len(token_text)
            
            if token_start < char_end and token_end > char_pos:
                occurrence_tokens.append(token_idx)
            
            current_char = token_end
        
        if occurrence_tokens:
            all_positions.append(occurrence_tokens)
        
        start_search = char_pos + 1
    
    return all_positions

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
    
    doc_tensors = []
    for doc in question_docs:
        doc_text = f"Document: {doc}\n"
        doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
        doc_tensors.append(torch.tensor(doc_tokens, dtype=torch.long))
    
    sub_q = intermediate_context[0]
    query = sub_q['query']
    if query.startswith("Intermediate query"):
        query = query[query.find(":")+1:].strip()
    
    docs = sub_q.get('retrieve docs', [])
    doc_chunk_ids = [doc_to_idx[doc] for doc in docs if doc in doc_to_idx]
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]
    
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
    
    doc_token_ids = input_ids[0, doc_start:doc_end].tolist()
    
    # 找到所有答案位置
    all_answer_positions = find_all_answer_positions(tokenizer, doc_token_ids, "National Cycle Network")
    print(f"Found {len(all_answer_positions)} occurrences of 'National Cycle Network'")
    for i, pos_list in enumerate(all_answer_positions):
        print(f"  Occurrence {i+1}: tokens {pos_list[0]}-{pos_list[-1]}")
    
    # 合并所有答案位置
    all_answer_tokens = set()
    for pos_list in all_answer_positions:
        all_answer_tokens.update(pos_list)
    
    # 计算 attention
    print("\nComputing attention...")
    target_layers = [23, 35]
    layer_attention = compute_attention_for_layers(model, input_ids, query_start, device, config, target_layers)
    
    layer_23_attn = layer_attention[23][:, :, doc_start:doc_end].mean(axis=(0, 1))
    layer_35_attn = layer_attention[35][:, :, doc_start:doc_end].mean(axis=(0, 1))
    
    # 创建可视化
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    colors = ['red', 'orange', 'purple', 'brown']  # 不同出现用不同颜色
    
    # Layer 23
    ax1 = axes[0]
    ax1.bar(range(len(layer_23_attn)), layer_23_attn, color='steelblue', alpha=0.7, width=1.0)
    
    for i, pos_list in enumerate(all_answer_positions):
        for pos in pos_list:
            ax1.axvline(x=pos, color=colors[i % len(colors)], alpha=0.8, linewidth=2)
        # 添加标签
        mid_pos = (pos_list[0] + pos_list[-1]) / 2
        ax1.annotate(f'#{i+1}', xy=(mid_pos, ax1.get_ylim()[1]*0.9), fontsize=10, color=colors[i % len(colors)], ha='center', fontweight='bold')
    
    ax1.set_title('Layer 23 (Entropy=4.75, LOWEST) - Colored lines = 4 occurrences of "National Cycle Network"', fontsize=12)
    ax1.set_xlabel('Document Token Position')
    ax1.set_ylabel('Attention Score')
    ax1.set_xlim(0, len(layer_23_attn))
    
    # Layer 35
    ax2 = axes[1]
    ax2.bar(range(len(layer_35_attn)), layer_35_attn, color='seagreen', alpha=0.7, width=1.0)
    
    for i, pos_list in enumerate(all_answer_positions):
        for pos in pos_list:
            ax2.axvline(x=pos, color=colors[i % len(colors)], alpha=0.8, linewidth=2)
        mid_pos = (pos_list[0] + pos_list[-1]) / 2
        ax2.annotate(f'#{i+1}', xy=(mid_pos, ax2.get_ylim()[1]*0.9), fontsize=10, color=colors[i % len(colors)], ha='center', fontweight='bold')
    
    ax2.set_title('Layer 35 (Entropy=6.55, HIGHEST) - Same answer positions marked', fontsize=12)
    ax2.set_xlabel('Document Token Position')
    ax2.set_ylabel('Attention Score')
    ax2.set_xlim(0, len(layer_35_attn))
    
    plt.tight_layout()
    plt.savefig('/mnt/data/wjh/FusionRAG/attention_comparison_v3.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to /mnt/data/wjh/FusionRAG/attention_comparison_v3.png")
    
    # 统计每次出现的 attention
    print(f"\n{'='*70}")
    print("Attention on each occurrence of 'National Cycle Network'")
    print(f"{'='*70}")
    
    for i, pos_list in enumerate(all_answer_positions):
        l23_attn = np.mean([layer_23_attn[p] for p in pos_list])
        l35_attn = np.mean([layer_35_attn[p] for p in pos_list])
        print(f"\nOccurrence #{i+1} (tokens {pos_list[0]}-{pos_list[-1]}):")
        print(f"  Layer 23: {l23_attn:.6f}")
        print(f"  Layer 35: {l35_attn:.6f}")
        print(f"  Ratio (L23/L35): {l23_attn/l35_attn:.2f}x")
    
    # 整体统计
    other_tokens = [i for i in range(len(layer_23_attn)) if i not in all_answer_tokens]
    l23_answer_avg = np.mean([layer_23_attn[p] for p in all_answer_tokens])
    l23_other_avg = np.mean([layer_23_attn[p] for p in other_tokens])
    l35_answer_avg = np.mean([layer_35_attn[p] for p in all_answer_tokens])
    l35_other_avg = np.mean([layer_35_attn[p] for p in other_tokens])
    
    print(f"\n{'='*70}")
    print("Overall Statistics")
    print(f"{'='*70}")
    print(f"\nLayer 23:")
    print(f"  Avg on ALL answer tokens: {l23_answer_avg:.6f}")
    print(f"  Avg on other tokens:      {l23_other_avg:.6f}")
    print(f"  Ratio: {l23_answer_avg/l23_other_avg:.2f}x")
    
    print(f"\nLayer 35:")
    print(f"  Avg on ALL answer tokens: {l35_answer_avg:.6f}")
    print(f"  Avg on other tokens:      {l35_other_avg:.6f}")
    print(f"  Ratio: {l35_answer_avg/l35_other_avg:.2f}x")


if __name__ == "__main__":
    main()
