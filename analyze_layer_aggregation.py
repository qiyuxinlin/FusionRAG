"""
分析多层聚合 vs 单层的效果差异
重点研究：为什么用4层比只用最后1层效果好
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def compute_layer_attention(model, input_ids, layer_indices, device, config):
    """只计算指定层的attention"""
    model = model.to(device)
    input_ids = input_ids.to(device)

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    layer_attention = {}

    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        bsz, q_len, _ = hidden_states.shape

        for layer_idx, layer in enumerate(model.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            q_proj = layer.self_attn.q_proj(hidden_states)
            k_proj = layer.self_attn.k_proj(hidden_states)
            v_proj = layer.self_attn.v_proj(hidden_states)
            
            query_states = q_proj.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = k_proj.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = v_proj.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            
            cos, sin = model.model.rotary_emb(value_states, position_ids=torch.arange(q_len, device=device).unsqueeze(0))
            if cos.dim() == 2:
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            elif cos.dim() == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)
            
            if layer_idx in layer_indices:
                n_rep = num_heads // num_kv_heads
                key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
                
                attn_weights = torch.matmul(query_states.float(), key_states_expanded.float().transpose(2, 3)) / (head_dim ** 0.5)
                causal_mask = torch.triu(torch.ones(q_len, q_len, device=device), diagonal=1).bool()
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
                
                # 平均所有头
                layer_attention[layer_idx] = attn_weights[0].mean(dim=0).cpu().numpy()
            else:
                # 简化计算 - 使用 SDPA
                n_rep = num_heads // num_kv_heads
                key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
                value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)
                attn_output = F.scaled_dot_product_attention(query_states, key_states_expanded, value_states_expanded, is_causal=True)
            
            if layer_idx not in layer_indices:
                n_rep = num_heads // num_kv_heads
                value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)
                attn_output = F.scaled_dot_product_attention(query_states, key_states.repeat_interleave(n_rep, dim=1), value_states_expanded, is_causal=True)
            else:
                value_states_expanded = value_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
                attn_output = torch.matmul(attn_weights.to(value_states_expanded.dtype), value_states_expanded)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)
            
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
    
    return layer_attention


def analyze_token_selection(attn_matrix, query_start, doc_start, doc_end, ratio=0.3):
    """分析给定attention矩阵会选择哪些tokens"""
    # 取最后一个query token对文档区域的attention
    query_attn = attn_matrix[-1, doc_start:doc_end]
    total_tokens = len(query_attn)
    keep_count = max(1, int(total_tokens * ratio))
    
    # 选择top-k tokens
    top_indices = np.argsort(query_attn)[-keep_count:]
    selected_positions = set(top_indices)
    
    return selected_positions, query_attn


def main():
    device = "cuda:0"
    model_path = "/mnt/data/models/Qwen2.5-3B-Instruct"
    data_path = "/mnt/data/wjh/FusionRAG/result_reflect.json"

    print("Loading model...")
    from transformers import AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    
    print("Loading data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    system_text = "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the provided documents.\n"
    system_tokens = tokenizer.encode(system_text, add_special_tokens=False)
    system_len = len(system_tokens)
    
    # 分析多个样本
    num_samples = 10
    all_results = []
    
    for sample_idx in range(min(num_samples, len(data))):
        sample = data[sample_idx]
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx + 1}")
        print(f"{'='*60}")
        
        doc_texts = list(sample.get('retrieved_results', []))
        for ctx in sample.get('intermediate_context', []):
            doc_texts.extend(ctx.get('retrieve docs', []))
        doc_texts = list(dict.fromkeys(doc_texts))[:20]
        
        if len(doc_texts) < 3:
            continue
        
        doc_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(doc_texts)])
        doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_tokens) > 1500:
            doc_tokens = doc_tokens[:1500]
        doc_len = len(doc_tokens)
        
        question = sample.get('question', '')
        query_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\nAnswer: "
        query_tokens = tokenizer.encode(query_text, add_special_tokens=False)
        
        full_tokens = system_tokens + doc_tokens + query_tokens
        input_ids = torch.tensor([full_tokens], dtype=torch.long)
        
        total_len = len(full_tokens)
        doc_start = system_len
        doc_end = system_len + doc_len
        query_start = system_len + doc_len
        
        print(f"Input: {total_len} tokens, doc region: [{doc_start}, {doc_end})")
        
        # 计算最后4层的attention
        layer_indices = [32, 33, 34, 35]
        layer_attention = compute_layer_attention(model, input_ids, layer_indices, device, config)
        
        # 分析单层选择 vs 多层聚合
        ratio = 0.3
        
        # 1. 只用最后一层 (Layer 35)
        last_layer_attn = layer_attention[35]
        last_layer_selected, last_layer_scores = analyze_token_selection(
            last_layer_attn, query_start, doc_start, doc_end, ratio
        )
        
        # 2. 多层聚合 (union of top-k from each layer)
        multi_layer_selected = set()
        for layer_idx in layer_indices:
            selected, _ = analyze_token_selection(
                layer_attention[layer_idx], query_start, doc_start, doc_end, ratio
            )
            multi_layer_selected = multi_layer_selected.union(selected)
        
        # 3. 分析重叠和差异
        overlap = last_layer_selected.intersection(multi_layer_selected)
        only_last = last_layer_selected - multi_layer_selected
        only_multi = multi_layer_selected - last_layer_selected
        
        print(f"\nToken Selection Analysis (ratio={ratio}):")
        print(f"  最后一层选择: {len(last_layer_selected)} tokens")
        print(f"  多层聚合选择: {len(multi_layer_selected)} tokens")
        print(f"  重叠tokens: {len(overlap)}")
        print(f"  仅最后一层: {len(only_last)}")
        print(f"  仅多层聚合: {len(only_multi)}")
        print(f"  多层额外覆盖: +{len(only_multi)} tokens ({len(only_multi)/len(last_layer_selected)*100:.1f}% more)")
        
        # 4. 分析各层选择的独特贡献
        print("\n各层独特贡献分析:")
        for layer_idx in layer_indices:
            selected, _ = analyze_token_selection(
                layer_attention[layer_idx], query_start, doc_start, doc_end, ratio
            )
            unique = selected - last_layer_selected
            if layer_idx != 35:
                print(f"  Layer {layer_idx}: {len(selected)} tokens, 其中 {len(unique)} 个是最后一层没选到的")
        
        all_results.append({
            'sample_idx': sample_idx,
            'last_layer_count': len(last_layer_selected),
            'multi_layer_count': len(multi_layer_selected),
            'extra_coverage': len(only_multi)
        })
        
        torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "=" * 60)
    print("汇总分析")
    print("=" * 60)
    avg_last = np.mean([r['last_layer_count'] for r in all_results])
    avg_multi = np.mean([r['multi_layer_count'] for r in all_results])
    avg_extra = np.mean([r['extra_coverage'] for r in all_results])
    
    print(f"平均最后一层选择: {avg_last:.1f} tokens")
    print(f"平均多层聚合选择: {avg_multi:.1f} tokens")
    print(f"平均额外覆盖: {avg_extra:.1f} tokens ({avg_extra/avg_last*100:.1f}% more)")
    
    print("\n关键结论:")
    print("多层聚合通过union操作，可以捕获更多潜在相关的tokens")
    print("这些额外的tokens可能包含对回答问题有用的信息")


if __name__ == "__main__":
    main()
