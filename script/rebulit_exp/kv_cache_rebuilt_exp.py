#!/usr/bin/env python3
"""
KV Cache 重建实验脚本

此脚本测试从 KV cache 重建文本的能力，包含两个阶段：
1. 阶段1：构建上下文 [doc1, ..., docK, doc_o] 并只保存 doc_o 的 KV cache
2. 阶段2：加载 doc_o 的 KV cache，使用 repeat_prompt 尝试重建文档

Author: Claude Code
Version: 1.0.0
"""

import os
import sys
import json
import random
import time
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoConfig

# 将项目目录添加到路径
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_dir)

from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect import load_model
from ktransformers.util.utils_v2 import compute_f1


# ============================================================
# RoPE 辅助函数
# ============================================================

def rotate_half(x):
    """旋转输入张量的一半维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(x, cos, sin):
    """对单个张量（key 或 value）应用旋转位置编码"""
    cos = cos.unsqueeze(1)  # 添加 head 维度
    sin = sin.unsqueeze(1)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


# ============================================================
# 语义相似度函数
# ============================================================

def compute_semantic_similarity(text1: str, text2: str, bge_model) -> float:
    """
    使用 BGE 模型计算两个文本之间的语义相似度

    参数:
        text1: 第一个文本
        text2: 第二个文本
        bge_model: BGE 模型 (BGEM3FlagModel 或 FlagModel)

    返回:
        float: 余弦相似度 [0, 1]
    """
    # BGE-M3 encode 方法 - 返回字典格式
    result = bge_model.encode([text1, text2])

    # 调试：打印返回结构（只第一次）
    if not hasattr(compute_semantic_similarity, '_printed_type'):
        print(f"[DEBUG] BGE encode 返回类型: {type(result)}")
        if isinstance(result, dict):
            print(f"[DEBUG] BGE encode 键: {result.keys()}")
        compute_semantic_similarity._printed_type = True

    # 处理返回值
    if isinstance(result, dict):
        # 尝试各种可能的键
        for key in ['dense', 'densen']:

            if key in result:
                embeddings = result[key]
                break
        else:
            # 如果都没有，使用第一个 numpy 数组
            for value in result.values():
                if isinstance(value, np.ndarray) and len(value.shape) == 2:
                    embeddings = value
                    break
            else:
                raise ValueError(f"无法从 BGE 返回值中提取嵌入，可用键: {result.keys()}")
    else:
        embeddings = result

    similarity = (embeddings[0] @ embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)


def classify_match(f1_score: float, semantic_sim: float,
                  f1_threshold: float = 0.9,
                  semantic_threshold: float = 0.85) -> str:
    """
    根据 F1 分数和语义相似度对匹配类型进行分类

    返回:
        str: 'literal_match', 'semantic_match', 'partial_match', 'no_match' 之一
    """
    if f1_score >= f1_threshold:
        return "literal_match"  # 字面匹配
    elif semantic_sim >= semantic_threshold:
        return "semantic_match"  # 语义匹配
    elif semantic_sim >= 0.6:
        return "partial_match"  # 部分匹配
    else:
        return "no_match"  # 不匹配


# ============================================================
# 数据选择函数
# ============================================================

def select_target_document(docs: List[Dict], seed: int = None) -> Dict:
    """从文档池中随机选择一个目标文档 (doc_o)"""
    if seed is not None:
        random.seed(seed)
    return random.choice(docs)


def select_random_documents(docs: List[Dict], K: int, exclude_id: int, seed: int = None) -> List[Dict]:
    """随机选择 K 个文档，排除目标文档"""
    if seed is not None:
        random.seed(seed)
    available = [d for d in docs if d['id'] != exclude_id]
    return random.sample(available, min(K, len(available)))


# ============================================================
# 阶段1：构建上下文并保存 KV Cache
# ============================================================

def build_and_save_target_kv_cache(
    model,
    tokenizer,
    past_key_values,
    target_doc: Dict,
    random_docs: List[Dict],
    config
) -> Tuple[str, int, int]:
    """
    构建上下文序列: [random_doc_1, ..., random_doc_K, target_doc]
    并只保存 target_doc 的 KV cache（含 RoPE 调整）

    参数:
        model: 语言模型
        tokenizer: 分词器
        past_key_values: 静态缓存对象
        target_doc: 目标文档 (doc_o)
        random_docs: K 个随机文档列表
        config: 实验配置

    返回:
        cache_save_dir: 缓存保存目录
        target_doc_id: 目标文档 ID
        system_len: K 个随机文档的总长度 (tokens)
    """
    device = config.device if not config.use_multi_gpu else "cuda:0"

    # 1. 分词所有文档
    random_texts = [d['text'] for d in random_docs]
    target_text = target_doc['text']

    random_tokens_list = [tokenizer.encode(text, add_special_tokens=False) for text in random_texts]
    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)

    # 2. 构建 token 序列: [doc1, ..., docK, doc_o]
    system_tokens = []
    for tokens in random_tokens_list:
        system_tokens.extend(tokens)

    system_len = len(system_tokens)
    target_len = len(target_tokens)
    full_sequence = system_tokens + target_tokens
    input_tensor = torch.tensor(full_sequence).unsqueeze(0)  # [1, seq_len]

    print(f"  System (K个随机文档) 长度: {system_len} tokens")
    print(f"  Target (doc_o) 长度: {target_len} tokens")
    print(f"  完整序列长度: {len(full_sequence)} tokens")

    # 3. 为此实验创建保存目录
    experiment_dir = f"{config.cache_save_path}/exp_doc{target_doc['id']}"
    os.makedirs(experiment_dir, exist_ok=True)

    # 4. 重置缓存并运行前向传播
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    batch_size, seq_length = input_tensor.shape
    inputs = input_tensor.to(device)
    cache_position = torch.arange(seq_length, device=device)

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(inputs)
        _ = model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True
        )[0]

    # 5. 只提取目标部分 [system_len:system_len+target_len]
    past_len = past_key_values.past_tokens[0]
    num_layers = len(past_key_values.key_cache)

    target_key_cache = []
    target_value_cache = []

    for layer_idx in range(num_layers):
        # 提取目标部分
        layer_key = past_key_values.key_cache[layer_idx][:, :, system_len:past_len, :].cpu()
        layer_value = past_key_values.value_cache[layer_idx][:, :, system_len:past_len, :].cpu()
        target_key_cache.append(layer_key)
        target_value_cache.append(layer_value)

    # 6. 应用 RoPE 调整：从绝对位置转换到相对位置 0
    if config.revert_rope:
        print(f"  正在应用 RoPE 调整: 从位置 [{system_len}:{system_len+target_len}] 转换到 [0:{target_len}]")
        for layer_idx in range(num_layers):
            rotary_emb = model.model.layers[layer_idx].self_attn.rotary_emb

            # 获取 rotary 设备
            if hasattr(rotary_emb, 'inv_freq') and rotary_emb.inv_freq is not None:
                rotary_device = rotary_emb.inv_freq.device
            else:
                rotary_device = next(model.model.layers[layer_idx].parameters()).device

            # 移动到 rotary 设备进行 RoPE 计算
            layer_key = target_key_cache[layer_idx].to(rotary_device)
            layer_value = target_value_cache[layer_idx].to(rotary_device)

            # 原始绝对位置: [system_len, system_len+1, ..., system_len+target_len-1]
            original_position_ids = torch.arange(system_len, system_len + target_len,
                                                 device=rotary_device).unsqueeze(0)
            # 目标相对位置: [0, 1, 2, ..., target_len-1]
            target_position_ids = torch.arange(0, target_len, device=rotary_device).unsqueeze(0)

            # 获取两个位置的 cos/sin
            original_cos, original_sin = rotary_emb(layer_value, original_position_ids)
            target_cos, target_sin = rotary_emb(layer_value, target_position_ids)

            # 移除原始 RoPE (使用 -sin)
            layer_key = apply_rotary_pos_emb_single(layer_key, original_cos, -original_sin)
            # 在位置 0 应用新的 RoPE
            layer_key = apply_rotary_pos_emb_single(layer_key, target_cos, target_sin)

            # 移回 CPU
            target_key_cache[layer_idx] = layer_key.cpu()
            target_value_cache[layer_idx] = layer_value.cpu()

    # 堆叠并保存
    target_key_cache = torch.stack(target_key_cache)
    target_value_cache = torch.stack(target_value_cache)

    key_path = f'{experiment_dir}/{target_doc["id"]}_1_key.pt'
    value_path = f'{experiment_dir}/{target_doc["id"]}_1_value.pt'

    torch.save(target_key_cache.clone(), key_path)
    torch.save(target_value_cache.clone(), value_path)
    print(f"  KV cache 已保存到: {experiment_dir}")

    # 7. 保存元数据
    metadata = {
        'target_doc_id': target_doc['id'],
        'target_text': target_text,
        'target_tokens': target_tokens,
        'system_len': system_len,
        'target_len': target_len,
        'K': len(random_docs),
        'random_doc_ids': [d['id'] for d in random_docs]
    }
    metadata_path = f"{experiment_dir}/metadata.pt"
    torch.save(metadata, metadata_path)

    return experiment_dir, target_doc['id'], system_len


# ============================================================
# 阶段2：加载 KV Cache 并测试重建
# ============================================================

def generate_with_cache(
    model,
    tokenizer,
    past_key_values,
    doc_tokens: torch.Tensor,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    device: str
) -> str:
    """
    使用预加载的 KV cache 生成文本

    输入序列：只用 prompt_tokens 作为输入
    模型会通过 attention 访问 past_key_values 中的 doc_o KV cache

    参数:
        model: 语言模型
        tokenizer: 分词器
        past_key_values: 已预加载 doc_o KV 的静态缓存
        doc_tokens: 文档的 token IDs（仅用于获取长度）
        prompt_tokens: prompt 的 token IDs
        max_new_tokens: 最大生成 token 数
        device: 设备

    返回:
        str: 生成的文本
    """
    from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper

    doc_len = len(doc_tokens)
    prompt_len = len(prompt_tokens)

    tokens = []

    with torch.no_grad():
        # 处理 prompt tokens，模型会 attention 到 KV cache
        prompt_inputs = prompt_tokens.unsqueeze(0).to(device)
        cache_position = torch.arange(0, prompt_len, device=device)
        position_ids = cache_position.unsqueeze(0)

        inputs_embeds = model.model.embed_tokens(prompt_inputs)

        logits = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True
        )[0]

        logits_warper = LogitsProcessorList([
            TemperatureLogitsWarper(0.01),
            TopKLogitsWarper(top_k=1)
        ])

        next_token_scores = logits_warper(prompt_inputs, logits[:, -1, :])
        next_token = torch.argmax(next_token_scores, dim=-1)
        tokens.append(next_token)

        current_pos = prompt_len
        cache_position = torch.tensor([current_pos], device=device)
        position_ids = cache_position.unsqueeze(0)

        for step in range(max_new_tokens):
            if next_token.item() == tokenizer.eos_token_id:
                break

            inputs_embeds = model.model.embed_tokens(next_token.unsqueeze(0))
            logits = model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True
            )[0]

            next_token_scores = logits_warper(prompt_inputs, logits[:, -1, :])
            next_token = torch.argmax(next_token_scores, dim=-1)
            tokens.append(next_token)

            current_pos += 1
            cache_position = torch.tensor([current_pos], device=device)
            position_ids = cache_position.unsqueeze(0)

    generated_tokens = torch.cat(tokens, dim=-1)
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text

        # 前向传播，使用已有的 KV cache
        # 参考 prefill_and_generate，只传 cache_position，不传 position_ids
        logits = model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True
        )[0][:, -1, :].unsqueeze(0).clone()

        # 更新 past_tokens
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.past_tokens[layer_idx] += prompt_len

        # 设置 logits warper
        logits_warper = LogitsProcessorList([
            TemperatureLogitsWarper(0.01),
            TopKLogitsWarper(top_k=1)
        ])

        # 获取第一个生成的 token
        next_token_scores = logits_warper(prompt_inputs, logits[:, -1, :])
        next_token = torch.argmax(next_token_scores, dim=-1)
        tokens.append(next_token)

        # 步骤2：自回归生成
        current_pos = doc_len + prompt_len
        cache_position = torch.tensor([current_pos], device=device)
        position_ids = cache_position.unsqueeze(0)

        for step in range(max_new_tokens):
            # 检查 EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"  在步骤 {step} 遇到 EOS token，停止生成")
                break

            # 编码当前 token
            inputs_embeds = model.model.embed_tokens(next_token.unsqueeze(0))
            logits = model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True
            )[0]

            # 获取下一个 token
            next_token_scores = logits_warper(prompt_inputs, logits[:, -1, :])
            next_token = torch.argmax(next_token_scores, dim=-1)
            tokens.append(next_token)

            # 更新位置
            current_pos += 1
            cache_position = torch.tensor([current_pos], device=device)
            position_ids = cache_position.unsqueeze(0)

    # 解码：只解码生成的 tokens
    generated_tokens = torch.cat(tokens, dim=-1)
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


def load_kv_and_reconstruct(
    model,
    tokenizer,
    past_key_values,
    cache_dir: str,
    target_doc_id: int,
    repeat_prompt: str,
    config
) -> Tuple[str, Dict]:
    """
    加载 doc_o 的 KV cache 并使用 repeat_prompt 尝试重建文档

    参数:
        model: 语言模型
        tokenizer: 分词器
        past_key_values: 静态缓存对象
        cache_dir: 包含已保存缓存的目录
        target_doc_id: 目标文档 ID
        repeat_prompt: 要求重复文档的 prompt
        config: 实验配置

    返回:
        reconstructed_text: 生成的文本
        metrics: 包含评估指标的字典
    """
    device = config.device if not config.use_multi_gpu else "cuda:0"

    # 1. 加载元数据
    metadata = torch.load(f"{cache_dir}/metadata.pt")
    target_text = metadata['target_text']
    target_tokens = metadata['target_tokens']
    target_len = metadata['target_len']

    print(f"  目标文本长度: {len(target_text)} 字符")
    print(f"  目标 tokens 长度: {target_len}")

    # 2. 从磁盘加载 KV cache
    key_cache = torch.load(f"{cache_dir}/{target_doc_id}_1_key.pt", weights_only=True)
    value_cache = torch.load(f"{cache_dir}/{target_doc_id}_1_value.pt", weights_only=True)

    print(f"  加载的 KV cache 形状: {key_cache.shape}")  # [num_layers, batch, heads, seq_len, head_dim]

    # 3. 重置缓存并在位置 [0:target_len] 加载 doc_o 的 KV
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    for layer_idx in range(len(past_key_values.key_cache)):
        # key_cache 形状: [num_layers, batch, heads, seq_len, head_dim]
        k = key_cache[layer_idx].to(device)
        v = value_cache[layer_idx].to(device)

        seq_len = k.shape[2]  # 应该等于 target_len
        print(f"  Layer {layer_idx}: seq_len={seq_len}, target_len={target_len}")

        # 复制到缓存，从位置 0 开始
        past_key_values.key_cache[layer_idx].narrow(2, 0, seq_len).copy_(k.squeeze(0))
        past_key_values.value_cache[layer_idx].narrow(2, 0, seq_len).copy_(v.squeeze(0))
        past_key_values.past_tokens[layer_idx] = seq_len

    past_len = target_len
    print(f"  加载后 past_len: {past_len}")

    # 4. 重构策略：使用 KV cache 重建文档
    #
    # 输入序列：[doc_o KV cache, prompt 要求重复上面的内容]
    # - doc_o KV cache 在位置 [0:target_len]
    # - prompt 要求重复上面的文本，输出 JSON 格式
    # - 从 JSON 中提取 repeated_text 字段作为重建结果

    print(f"  [调试] KV cache 状态:")
    print(f"    past_tokens[0] = {past_key_values.past_tokens[0]} (应该是 {target_len})")
    print(f"    key_cache shape = {past_key_values.key_cache[0].shape}")

    prompt_tokens_list = tokenizer.encode(config.repeat_prompt, add_special_tokens=False)
    prompt_tensor = torch.tensor(prompt_tokens_list)

    print(f"  [调试] 完整序列: doc_o {len(target_tokens)} tokens + prompt {len(prompt_tokens_list)} tokens")
    print(f"  Prompt: '{config.repeat_prompt}'")

    # 生成
    reconstructed_text_raw = generate_with_cache(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_key_values,
        doc_tokens=torch.tensor(target_tokens),
        prompt_tokens=prompt_tensor,
        max_new_tokens=config.max_new_tokens,
        device=device
    )

    print(f"  原始生成文本长度: {len(reconstructed_text_raw)} 字符")
    print(f"  原始生成预览: {reconstructed_text_raw[:200]}...")

    # 使用 JSON 格式提取内容
    reconstructed_text = ""
    try:
        # 查找 JSON 对象
        json_start = reconstructed_text_raw.find('{')
        if json_start >= 0:
            # 找到匹配的结束 }
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(reconstructed_text_raw)):
                if reconstructed_text_raw[i] == '{':
                    brace_count += 1
                elif reconstructed_text_raw[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end > json_start:
                json_str = reconstructed_text_raw[json_start:json_end]
                print(f"  [提取] JSON 字符串: {json_str[:200]}...")

                data = json.loads(json_str)

                if 'repeated_text' in data:
                    reconstructed_text = data['repeated_text']
                    print(f"  [成功] 从 JSON 提取 'repeated_text'")
                else:
                    print(f"  [警告] JSON 中没有 'repeated_text'，可用字段: {list(data.keys())}")
                    # 尝试使用其他字段
                    if 'text' in data:
                        reconstructed_text = data['text']
                    elif 'content' in data:
                        reconstructed_text = data['content']
                    else:
                        reconstructed_text = str(data)
            else:
                print(f"  [错误] 未找到完整的 JSON 对象")
                reconstructed_text = reconstructed_text_raw.strip()
        else:
            print(f"  [错误] 未找到 JSON 对象开始")
            # 回退：尝试找 repeated_text 字段
            if '"repeated_text"' in reconstructed_text_raw:
                import re
                match = re.search(r'"repeated_text"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', reconstructed_text_raw)
                if match:
                    reconstructed_text = match.group(1).replace('\\"', '"').replace('\\n', '\n')
                    print(f"  [备用] 使用正则提取 repeated_text")
                else:
                    reconstructed_text = reconstructed_text_raw.strip()
            else:
                reconstructed_text = reconstructed_text_raw.strip()
                print(f"  [回退] 使用原始生成")

    except json.JSONDecodeError as e:
        print(f"  [错误] JSON 解析失败: {e}")
        reconstructed_text = reconstructed_text_raw.strip()

    # 如果提取的内容包含 <content> 标签，移除它们
    reconstructed_text = reconstructed_text.replace("<content>", "").replace("</content>", "").strip()

    print(f"  提取后的重建文本长度: {len(reconstructed_text)} 字符")
    print(f"  重建文本预览: {reconstructed_text[:200]}...")

    # 6. 计算指标
    f1_score = compute_f1(reconstructed_text, target_text, tokenizer)
    exact_match = (reconstructed_text.strip().lower() == target_text.strip().lower())

    # 7. 计算语义相似度
    semantic_sim = compute_semantic_similarity(target_text, reconstructed_text, config.bge_model)

    # 8. 分类匹配类型
    match_category = classify_match(
        f1_score, semantic_sim,
        config.f1_threshold,
        config.semantic_threshold
    )

    metrics = {
        'f1_score': f1_score,
        'exact_match': exact_match,
        'semantic_similarity': semantic_sim,
        'match_category': match_category,
        'generated_length': len(reconstructed_text),
        'target_length': len(target_text)
    }

    return reconstructed_text, metrics


# ============================================================
# 实验协调器
# ============================================================

def run_single_experiment(
    model,
    tokenizer,
    docs: List[Dict],
    config,
    experiment_idx: int,
    past_key_values
) -> Dict:
    """运行单个 KV cache 重建实验（复用传入的缓存对象）"""
    print(f"\n{'='*60}")
    print(f"运行实验 {experiment_idx + 1}/{config.num_experiments}")
    print(f"{'='*60}")

    # 重置缓存（而不是创建新的）
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    # 阶段1：选择文档并构建 KV cache
    target_doc = select_target_document(docs, seed=experiment_idx)
    random_docs = select_random_documents(docs, config.K, target_doc['id'], seed=experiment_idx)

    print(f"目标文档 ID: {target_doc['id']}")
    print(f"随机文档 IDs: {[d['id'] for d in random_docs]}")
    print(f"目标文本预览: {target_doc['text'][:100]}...")

    cache_dir, target_doc_id, system_len = build_and_save_target_kv_cache(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_key_values,
        target_doc=target_doc,
        random_docs=random_docs,
        config=config
    )

    # 阶段2：加载 KV cache 并测试重建
    # 注意：这里传入同一个 past_key_values 对象，函数内部会重置它
    reconstructed_text, metrics = load_kv_and_reconstruct(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_key_values,
        cache_dir=cache_dir,
        target_doc_id=target_doc_id,
        repeat_prompt=config.repeat_prompt,
        config=config
    )

    print(f"\n重建结果:")
    print(f"  F1 分数: {metrics['f1_score']:.4f}")
    print(f"  语义相似度: {metrics['semantic_similarity']:.4f}")
    print(f"  匹配类别: {metrics['match_category']}")
    print(f"  精确匹配: {metrics['exact_match']}")
    print(f"  原始文本: {target_doc['text']}...")
    print(f"  重建文本: {reconstructed_text}...")

    result = {
        'experiment_idx': experiment_idx,
        'target_doc_id': target_doc_id,
        'random_doc_ids': [d['id'] for d in random_docs],
        'K': config.K,
        'system_len': system_len,
        'target_len': len(target_doc['text']),
        'f1_score': metrics['f1_score'],
        'exact_match': metrics['exact_match'],
        'semantic_similarity': metrics['semantic_similarity'],
        'match_category': metrics['match_category'],
        'original_text': target_doc['text'],
        'reconstructed_text': reconstructed_text,
        'cache_dir': cache_dir
    }

    return result


# ============================================================
# 主函数
# ============================================================

@dataclass
class ExperimentConfig:
    model_type: str = 'qwen'
    model_path: str = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path: str = './data/2wiki_input_rebuilt.json'
    cache_save_path: str = './cache/rebuilt_exp/'
    K: int = 5
    num_experiments: int = 10
    max_new_tokens: int = 512
    # 使用 JSON 格式输出，简化内容提取
    repeat_prompt: str = (
        "\n\nInstruction: Repeat the above text word-for-word. "
        "Output your response as JSON with key 'repeated_text'."
    )
    device: str = "cuda:0"
    use_multi_gpu: bool = False
    revert_rope: bool = True
    bge_model_path: str = '/mnt/data/models/bge-m3-FP16'
    f1_threshold: float = 0.9
    semantic_threshold: float = 0.85
    bge_model = None  # 将在后续初始化


def main():
    parser = argparse.ArgumentParser(description='KV Cache 重建实验')
    parser.add_argument('--K', type=int, default=5, help='目标文档前的随机文档数量')
    parser.add_argument('--num_experiments', type=int, default=10, help='运行的实验次数')
    parser.add_argument('--model_type', type=str, default='qwen', help='模型类型')
    parser.add_argument('--model_path', type=str, default='/mnt/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--data_path', type=str, default='./data/2wiki_input_rebuilt.json')
    parser.add_argument('--cache_path', type=str, default='./cache/rebuilt_exp/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--repeat_prompt', type=str, default=None, help='Repeat prompt (默认: JSON格式)')
    parser.add_argument('--output_file', type=str, default='./results/kv_cache_rebuilt_results.json')
    parser.add_argument('--revert_rope', action='store_true', default=True)
    parser.add_argument('--bge_model_path', type=str, default='/mnt/data/models/bge-m3-FP16')
    parser.add_argument('--f1_threshold', type=float, default=0.9)
    parser.add_argument('--semantic_threshold', type=float, default=0.85)

    args = parser.parse_args()

    # 创建配置（移除路径末尾的斜杠避免双斜杠问题）
    cache_path = args.cache_path.rstrip('/')

    # 构建 repeat_prompt：如果命令行指定则使用，否则使用 ExperimentConfig 默认值
    repeat_prompt = args.repeat_prompt if args.repeat_prompt is not None else ExperimentConfig.repeat_prompt

    config = ExperimentConfig(
        model_type=args.model_type,
        model_path=args.model_path,
        data_path=args.data_path,
        cache_save_path=cache_path,
        K=args.K,
        num_experiments=args.num_experiments,
        max_new_tokens=args.max_new_tokens,
        repeat_prompt=repeat_prompt,
        device=args.device,
        use_multi_gpu=args.use_multi_gpu,
        revert_rope=args.revert_rope,
        bge_model_path=args.bge_model_path,
        f1_threshold=args.f1_threshold,
        semantic_threshold=args.semantic_threshold
    )

    print("="*80)
    print("KV Cache 重建实验")
    print("="*80)
    print(f"模型: {config.model_type} @ {config.model_path}")
    print(f"数据: {config.data_path}")
    print(f"K (目标文档前的随机文档数): {config.K}")
    print(f"实验次数: {config.num_experiments}")
    print(f"设备: {config.device}")
    print(f"RoPE 还原: {config.revert_rope}")
    print(f"BGE 模型: {config.bge_model_path}")
    print("="*80)

    # 加载数据
    print(f"\n从 {config.data_path} 加载文档...")
    with open(config.data_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    print(f"已加载 {len(docs)} 个文档")

    # 加载 BGE 模型用于语义相似度计算
    print(f"\n从 {config.bge_model_path} 加载 BGE 模型...")
    from FlagEmbedding import BGEM3FlagModel
    config.bge_model = BGEM3FlagModel(config.bge_model_path, use_fp16=True)
    print("BGE 模型加载成功")

    # 加载模型和分词器
    print(f"\n加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
    model_config.torch_dtype = torch.float16

    model, device_map = load_model(
        model_type=config.model_type,
        model_path=config.model_path,
        config=model_config,
        device=config.device,
        use_multi_gpu=config.use_multi_gpu
    )
    model.eval()
    print("模型加载成功")

    # 创建一个共享的 StaticCache 对象（所有实验复用）
    cache_device = config.device if not config.use_multi_gpu else 'auto'
    shared_cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=8192,
        device=cache_device,
        dtype=model.dtype
    )
    print("共享缓存对象创建成功")

    # 运行实验
    all_results = []
    success_count = 0
    match_counts = {
        'literal_match': 0,
        'semantic_match': 0,
        'partial_match': 0,
        'no_match': 0
    }

    for exp_idx in range(config.num_experiments):
        try:
            result = run_single_experiment(
                model=model,
                tokenizer=tokenizer,
                docs=docs,
                config=config,
                experiment_idx=exp_idx,
                past_key_values=shared_cache  # 传入共享的缓存对象
            )
            all_results.append(result)

            if result['f1_score'] >= config.f1_threshold:
                success_count += 1

            match_counts[result['match_category']] += 1

        except Exception as e:
            print(f"实验 {exp_idx} 出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 计算汇总统计
    print("\n" + "="*80)
    print("实验汇总")
    print("="*80)

    if all_results:
        f1_scores = [r['f1_score'] for r in all_results]
        semantic_sims = [r['semantic_similarity'] for r in all_results]
        exact_matches = [r['exact_match'] for r in all_results]

        print(f"总实验数: {len(all_results)}")
        print(f"\n成功 (F1 >= {config.f1_threshold}): {success_count} ({success_count/len(all_results)*100:.1f}%)")
        print(f"\nF1 分数统计:")
        print(f"  平均: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"  范围: [{np.min(f1_scores):.4f}, {np.max(f1_scores):.4f}]")
        print(f"\n语义相似度统计:")
        print(f"  平均: {np.mean(semantic_sims):.4f} ± {np.std(semantic_sims):.4f}")
        print(f"  范围: [{np.min(semantic_sims):.4f}, {np.max(semantic_sims):.4f}]")
        print(f"\n精确匹配: {sum(exact_matches)} ({sum(exact_matches)/len(exact_matches)*100:.1f}%)")
        print(f"\n匹配类别:")
        for category, count in match_counts.items():
            print(f"  {category}: {count} ({count/len(all_results)*100:.1f}%)")

        # 保存结果
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'K': config.K,
                    'num_experiments': config.num_experiments,
                    'model_type': config.model_type,
                    'model_path': config.model_path,
                    'repeat_prompt': config.repeat_prompt,
                    'f1_threshold': config.f1_threshold,
                    'semantic_threshold': config.semantic_threshold,
                    'revert_rope': config.revert_rope,
                    'bge_model_path': config.bge_model_path
                },
                'summary': {
                    'total_experiments': len(all_results),
                    'success_count': success_count,
                    'success_rate': success_count / len(all_results),
                    'avg_f1': float(np.mean(f1_scores)),
                    'std_f1': float(np.std(f1_scores)),
                    'min_f1': float(np.min(f1_scores)),
                    'max_f1': float(np.max(f1_scores)),
                    'avg_semantic_similarity': float(np.mean(semantic_sims)),
                    'std_semantic_similarity': float(np.std(semantic_sims)),
                    'exact_match_count': sum(exact_matches),
                    'exact_match_rate': sum(exact_matches) / len(exact_matches),
                    'match_counts': match_counts
                },
                'results': all_results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {args.output_file}")
    else:
        print("没有成功完成的实验!")


if __name__ == '__main__':
    main()
