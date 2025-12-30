"""
用 test_fusionrag_reflect.py 相同的输入构造方式分析熵选层
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def compute_draft_attention(model, input_ids, query_start, device, config):
    """计算后50%层的 attention (和 utils.py 中一致)"""
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
        
        # 获取 rotary_emb
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
            
            # 计算所有层的 attention (不再限制后50%)
            if True:  # 原来是 layer_idx >= num_layers // 2
                # 只计算 query positions 的 attention
                query_states_subset = query_states[:, :, query_start:, :]
                attn_weights = torch.matmul(
                    query_states_subset.float(),
                    key_states_expanded.float().transpose(2, 3)
                ) / (head_dim ** 0.5)
                
                # Causal mask
                query_positions = torch.arange(query_start, seq_len, device=device)
                key_positions = torch.arange(seq_len, device=device)
                causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
                
                # 保存
                layer_attention[layer_idx] = attn_weights[0].cpu().numpy()
                
                # 使用计算好的 attention 继续
                attn_output_subset = torch.matmul(attn_weights.to(value_states_expanded.dtype), value_states_expanded)
                
                # 前面的 tokens 用 SDPA
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
                # 前 50% 的层用 SDPA
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
            
            if layer_idx % 8 == 0:
                print(f"  Layer {layer_idx} done")
    
    print(f"  Layer {num_layers-1} done")
    return layer_attention


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
    
    # System prompt (和 test_fusionrag_reflect.py 一致，从配置文件加载)
    prompt_config_path = "./config/dataset2prompt_few-shot.json"
    with open(prompt_config_path, 'r', encoding='utf-8') as f:
        prompt_config = json.load(f)
    system_text = prompt_config["system_prompt"]["Qwen2.5"]["2wikimqa"]
    system_tokens = tokenizer.encode(system_text, add_special_tokens=True)
    system_len = len(system_tokens)
    print(f"System prompt: {system_len} tokens")
    
    # 分析多个样本
    num_samples = 1  # 只分析第一个样本，看所有层的熵
    all_layer_selections = []
    
    for main_q_idx in range(min(num_samples, len(data))):
        sample = data[main_q_idx]
        intermediate_context = sample.get('intermediate_context', [])
        
        # 收集所有文档 (和 test_fusionrag_reflect.py 一致)
        question_docs = []
        doc_to_idx = {}
        
        for sub_q in intermediate_context:
            docs = sub_q.get('retrieve docs', [])
            for doc in docs:
                if doc not in doc_to_idx:
                    question_docs.append(doc)
                    doc_to_idx[doc] = len(question_docs)
        
        if len(question_docs) == 0:
            continue
        
        # 构建文档 tokens (和 test_fusionrag_reflect.py 一致)
        doc_tensors = []
        for doc in question_docs:
            doc_text = f"Document: {doc}\n"
            doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            doc_tensors.append(torch.tensor(doc_tokens, dtype=torch.long))
        
        # 分析每个子问题
        for sub_q_idx, sub_q in enumerate(intermediate_context):
            print(f"\n{'='*80}")
            print(f"Main Question {main_q_idx+1}, Sub-question {sub_q_idx+1}")
            print(f"{'='*80}")
            
            query = sub_q['query']
            if query.startswith("Intermediate query"):
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()
            
            print(f"Question: {query[:80]}...")
            
            # 获取子问题使用的文档
            docs = sub_q.get('retrieve docs', [])
            doc_chunk_ids = [doc_to_idx[doc] for doc in docs if doc in doc_to_idx]
            sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]
            
            # 构建完整输入: system + docs + question
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
            
            print(f"Input: {total_len} tokens (doc: {doc_len}, query: {len(question_tokens)})")
            
            # 计算 attention
            layer_attention = compute_draft_attention(model, input_ids, query_start, device, config)
            
            # 计算每层的熵
            layer_entropy = {}
            for layer_idx, attn in layer_attention.items():
                # attn: [num_heads, query_len, seq_len]
                query_to_doc = attn[:, :, doc_start:doc_end]  # [num_heads, query_len, doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
                
                p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
                p = np.clip(p, 1e-10, 1.0)
                entropy = -np.sum(p * np.log(p))
                layer_entropy[layer_idx] = entropy
            
            # 选择熵最低的4层
            sorted_layers = sorted(layer_entropy.items(), key=lambda x: x[1])
            selected_layers = [layer_idx for layer_idx, _ in sorted_layers[:4]]
            
            print(f"\nLayer entropy analysis:")
            for layer_idx, entropy in sorted_layers:
                marker = " <-- selected" if layer_idx in selected_layers else ""
                print(f"  Layer {layer_idx}: entropy={entropy:.4f}{marker}")
            
            print(f"\nSelected layers (lowest entropy): {selected_layers}")
            all_layer_selections.append(selected_layers)
            
            torch.cuda.empty_cache()
            
            # 只分析第一个子问题
            break
    
    # 汇总
    print(f"\n{'='*80}")
    print("Summary of Layer Selections")
    print(f"{'='*80}")
    for i, layers in enumerate(all_layer_selections):
        print(f"  Sample {i+1}: {layers}")
    
    # 统计
    layer_counts = {}
    for layers in all_layer_selections:
        for layer in layers:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    print(f"\nLayer selection frequency:")
    for layer_idx in sorted(layer_counts.keys()):
        count = layer_counts[layer_idx]
        bar = "█" * count
        print(f"  Layer {layer_idx}: {count} times {bar}")


if __name__ == "__main__":
    main()
