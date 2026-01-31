#!/usr/bin/env python
# coding=utf-8
'''
Description  :
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
'''
import torch
from torch import nn
import itertools
import time
import enum
import re
import string
import json
import collections
import numpy as np
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
    TypicalLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    GenerationConfig,
    AutoTokenizer
)
from rouge import Rouge
from filelock import FileLock


# ============================================================
# Smart Query Selection 辅助函数
# ============================================================

def compute_dynamic_budget(attention_scores, doc_len, epsilon=0.1, delta=0.05,
                           base_sample_ratio=0.05, topk_ratio=0.5, min_rate=0.05, max_rate=0.5):
    """
    根据 attention 分布动态计算需要的 budget (参考 vAttention 论文)

    核心思想 (vAttention 论文)：
    1. Top-k 选择：找到覆盖 (1-ε) 累积权重需要的最少 token
    2. 随机采样：对残差部分使用 CLT 采样理论估计需要的样本数
       公式: n ≥ (z × CV / ε)²，其中 CV = σ/μ 是变异系数
    3. 总 budget = top-k + random

    - attention 集中（少数 token 占大部分权重）→ top-k 少，random 也少
    - attention 分散（权重分布均匀）→ top-k 多，random 也多

    Args:
        attention_scores: torch.Tensor [doc_len], 每个 token 的 attention 分数
        doc_len: 文档长度
        epsilon: 误差容忍度 (如 0.1 = 10% 相对误差)
        delta: 置信度 (如 0.05 = 95% 置信度)
        base_sample_ratio: 保留参数兼容性
        topk_ratio: 保留参数兼容性
        min_rate: 最小重算比例
        max_rate: 最大重算比例

    Returns:
        dict: 包含动态计算的 budget 信息
    """
    import scipy.stats as stats

    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.float().cpu()

    # 确保 attention 分数归一化
    attention_scores = attention_scores / attention_scores.sum()

    # ========================================
    # Step 1: 计算 top-k budget (基于累积覆盖率)
    # ========================================
    # 目标：找到覆盖 (1-ε) 权重需要的最少 token 数量

    sorted_scores, sorted_indices = torch.sort(attention_scores, descending=True)
    cumsum = torch.cumsum(sorted_scores, dim=0)

    # 找到累积权重首次超过 (1-ε) 的位置
    coverage_threshold = 1.0 - epsilon
    coverage_mask = cumsum >= coverage_threshold

    if coverage_mask.any():
        topk_budget = (coverage_mask.int().argmax().item() + 1)  # +1 因为 argmax 返回的是索引
    else:
        topk_budget = doc_len

    # 计算 top-k 覆盖的实际权重
    topk_coverage = cumsum[topk_budget - 1].item() if topk_budget > 0 else 0.0

    # ========================================
    # Step 2: 计算 random budget (基于残差方差)
    # ========================================
    # 对于残差部分（未被 top-k 选中的），使用采样来估计

    residual_scores = sorted_scores[topk_budget:]  # 残差部分的分数
    residual_weight = 1.0 - topk_coverage  # 残差部分的总权重

    if len(residual_scores) > 0 and residual_weight > 1e-10:
        # 残差部分的统计量
        residual_mean = residual_scores.mean().item()
        residual_std = residual_scores.std().item()

        if residual_mean > 1e-10:
            residual_cv = residual_std / residual_mean
        else:
            residual_cv = 0.0

        # 使用 CLT 计算需要的采样数量来估计残差
        # 目标：残差估计的相对误差 < ε (相对于残差权重)
        z = stats.norm.ppf(1 - delta / 2)

        if residual_cv > 0:
            # 需要的样本数 = (z * cv / ε_residual)²
            # 其中 ε_residual 是残差部分允许的相对误差
            # 由于残差权重本身就很小，我们允许更大的相对误差
            epsilon_residual = epsilon / residual_weight if residual_weight > epsilon else 1.0
            random_budget_theory = (z * residual_cv / epsilon_residual) ** 2
            random_budget = min(int(random_budget_theory), len(residual_scores))
        else:
            random_budget = 0
    else:
        residual_cv = 0.0
        residual_weight = 0.0
        random_budget = 0

    # 计算熵（衡量分布的均匀程度）
    # 熵越高 = 分布越均匀 = 需要更多 token
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10)).item()
    max_entropy = np.log(doc_len)  # 均匀分布的熵
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # ========================================
    # Step 3: 计算基于覆盖率的 rate
    # ========================================
    total_budget = topk_budget + random_budget
    coverage_based_rate = total_budget / doc_len

    # ========================================
    # Step 4: 直接使用 vAttention 的覆盖率+采样方法
    # ========================================
    # 纯粹按照 vAttention 论文：top-k (覆盖率) + random (CLT采样)
    # 不额外添加熵自适应，保持方法的纯粹性

    dynamic_rate = coverage_based_rate
    rate_source = f"vAttention: topk={topk_budget}({topk_budget/doc_len:.1%}) + random={random_budget}({random_budget/doc_len:.1%})"

    # 应用 min/max 限制
    dynamic_rate = max(min_rate, min(max_rate, dynamic_rate))

    # 重新计算 budget
    total_budget = int(dynamic_rate * doc_len)

    # 按比例分配 top-k 和 random
    if topk_budget + random_budget > 0:
        topk_ratio_actual = topk_budget / (topk_budget + random_budget)
    else:
        topk_ratio_actual = 1.0

    topk_budget_final = int(total_budget * topk_ratio_actual)
    random_budget_final = total_budget - topk_budget_final

    return {
        'total_budget': total_budget,
        'topk_budget': topk_budget_final,
        'random_budget': random_budget_final,
        'dynamic_rate': dynamic_rate,
        # 统计信息
        'topk_coverage': topk_coverage,
        'topk_count_for_coverage': topk_budget,  # 达到 (1-ε) 覆盖需要的原始 top-k 数量
        'residual_weight': residual_weight,
        'residual_cv': residual_cv,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'coverage_threshold': coverage_threshold,
        'coverage_based_rate': coverage_based_rate,
        'rate_source': rate_source,
    }


def compute_dynamic_ratio_comprehensive(
    attention_scores,
    doc_len,
    base_ratio=0.3,
    min_ratio=0.20,
    max_ratio=0.50
):
    """
    综合多特征的动态比例计算（新版本）

    基于 draft model attention 分布的多个特征来动态计算重算比例：
    1. Coverage-based ratio: 达到 85% attention 覆盖所需比例
    2. Connected components: 高 attention 位置的分散程度
    3. Spread factor: 高 attention 位置在文档中的跨度
    4. Gini coefficient: Attention 集中度

    核心思想：不依赖单一指标，综合多个分布特征来决定比例
    - 高度集中 + 少量连通分量 → 可降低比例（但不低于 min_ratio）
    - 分散 + 多个连通分量 → 需要提高比例
    - 始终保证 safety buffer 捕获 long-tail tokens

    Args:
        attention_scores: torch.Tensor [doc_len], 每个 token 的 attention 分数
        doc_len: 文档长度
        base_ratio: 参考基准比例
        min_ratio: 最小比例（默认 0.20，确保 long-tail）
        max_ratio: 最大比例

    Returns:
        dict: 包含动态计算的比例和详细分析
    """
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.float().cpu().numpy()
    else:
        attention_scores = np.array(attention_scores)

    # 归一化
    attention_scores = attention_scores / (attention_scores.sum() + 1e-10)

    # =========================================================================
    # 特征 1: Coverage-based Ratio
    # =========================================================================
    sorted_indices = np.argsort(attention_scores)[::-1]
    sorted_attn = attention_scores[sorted_indices]
    cumsum_attn = np.cumsum(sorted_attn)

    # 计算达到不同覆盖率所需的 token 比例
    coverage_thresholds = [0.80, 0.85, 0.90]
    coverage_ratios = {}

    for threshold in coverage_thresholds:
        tokens_needed = np.searchsorted(cumsum_attn, threshold) + 1
        ratio = tokens_needed / doc_len
        coverage_ratios[threshold] = ratio

    # =========================================================================
    # 特征 1b: Top-k Concentration（Top-k 集中度）
    # =========================================================================
    # 检查 top-1, top-3, top-5 tokens 的 attention 占比
    top1_ratio = sorted_attn[0]
    top3_ratio = sorted_attn[:min(3, len(sorted_attn))].sum()
    top5_ratio = sorted_attn[:min(5, len(sorted_attn))].sum()

    # =========================================================================
    # 特征 2: Connected Components（连通分量）
    # =========================================================================
    mean_attn = np.mean(attention_scores)
    std_attn = np.std(attention_scores)
    threshold = mean_attn + 0.5 * std_attn

    high_attn_positions = list(np.where(attention_scores > threshold)[0])
    components = find_connected_components(high_attn_positions, max_gap=2)
    num_components = len(components)

    # 归一化分量数量（假设 1-20 个分量）
    normalized_components = min(max(num_components - 1, 0) / 19.0, 1.0)

    # =========================================================================
    # 特征 3: Spread Factor（分散度）
    # =========================================================================
    if len(high_attn_positions) > 0:
        positions_array = np.array(high_attn_positions)
        position_span = positions_array.max() - positions_array.min() if len(positions_array) > 1 else 0
        spread_ratio = position_span / doc_len
    else:
        spread_ratio = 0.0
        position_span = 0

    # =========================================================================
    # 特征 4: Gini Coefficient（基尼系数）
    # =========================================================================
    sorted_attn_gini = np.sort(attention_scores)
    n = len(sorted_attn_gini)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_attn_gini)) / (n * np.sum(sorted_attn_gini)) - (n + 1) / n
    # Gini: 0 (平等) → 1 (不平等/集中)

    # =========================================================================
    # 综合计算动态比例（V2：真正连续、渐进的动态调整）
    # =========================================================================

    # ===== 核心思想 =====
    # 1. 使用连续的函数而不是离散的阈值
    # 2. 基于 Top-k concentration 直接决定主要比例
    # 3. 其他特征作为微调

    # ===== Step 1: 基于 Top-k Concentration 的核心比例 =====
    # 核心观察：如果 top-k 占的比例越高，需要的重算比例越低

    # 使用加权的 top-k 作为集中度指标
    # top1 权重最高，top3 次之，top5 再次之
    concentration_score = (
        top1_ratio * 3.0 +      # top-1 权重 3
        top3_ratio * 2.0 +      # top-3 权重 2
        top5_ratio * 1.0        # top-5 权重 1
    ) / 6.0  # 归一化

    # 集中度越高，需要的比例越低
    # 使用平滑的映射函数
    # concentration_score ∈ [0, 1]
    # - 0.7+: 高度集中 → 5-10%
    # - 0.5-0.7: 中度集中 → 10-20%
    # - 0.3-0.5: 低集中 → 20-30%
    # - <0.3: 很分散 → 25-30%

    if concentration_score > 0.6:
        # 高度集中：线性映射 [0.6, 1.0] → [15%, 5%]
        core_ratio = 0.15 - (concentration_score - 0.6) / 0.4 * 0.10
    elif concentration_score > 0.4:
        # 中度集中：线性映射 [0.4, 0.6] → [22%, 15%]
        core_ratio = 0.22 - (concentration_score - 0.4) / 0.2 * 0.07
    else:
        # 分散：线性映射 [0, 0.4] → [30%, 22%]
        core_ratio = 0.30 - concentration_score / 0.4 * 0.08

    ratio = core_ratio

    # ===== Step 2: 连通分量微调（渐进式）=====
    # 分量越多，信息越分散，需要更多 tokens
    # 使用平滑函数而不是阶梯

    if num_components <= 3:
        # 极少分量：降低 2-3%
        component_adjustment = -0.02 - (3 - num_components) * 0.005
    elif num_components <= 10:
        # 中等分量：微调 -2% 到 +2%
        component_adjustment = (num_components - 3) / 7.0 * 0.04 - 0.02
    else:
        # 很多分量：增加，但用平滑曲线
        # 10-20 分量：+2% 到 +5%
        # 20+ 分量：+5% 到 +8%
        excess_components = num_components - 10
        component_adjustment = 0.02 + min(excess_components / 10.0 * 0.03, 0.06)

    ratio += component_adjustment

    # ===== Step 3: Spread 微调（平滑）=====
    # spread_ratio ∈ [0, 1]
    # 0-0.3: -1%
    # 0.3-0.7: 0%
    # 0.7-1.0: +1% to +3%

    if spread_ratio < 0.3:
        spread_adjustment = -0.01
    elif spread_ratio < 0.7:
        spread_adjustment = 0.0
    else:
        # 线性映射 [0.7, 1.0] → [0%, 3%]
        spread_adjustment = (spread_ratio - 0.7) / 0.3 * 0.03

    ratio += spread_adjustment

    # ===== Step 4: Gini 微调（平滑）=====
    # gini ∈ [0, 1]
    # 0.9+: 极度集中 → -3%
    # 0.7-0.9: 中度集中 → -1% to -3%
    # 0.5-0.7: 中等 → 0%
    # <0.5: 均匀 → +2%

    if gini > 0.9:
        gini_adjustment = -0.03
    elif gini > 0.7:
        # 线性映射 [0.7, 0.9] → [-1%, -3%]
        gini_adjustment = -0.01 - (gini - 0.7) / 0.2 * 0.02
    elif gini > 0.5:
        # 线性映射 [0.5, 0.7] → [0%, -1%]
        gini_adjustment = -(gini - 0.5) / 0.2 * 0.01
    else:
        # 均匀分布，需要更多
        gini_adjustment = 0.02

    ratio += gini_adjustment

    # ===== Step 5: 应用 min/max 限制 =====
    ratio = max(ratio, min_ratio)
    ratio = min(ratio, max_ratio)

    # 四舍五入到 0.05 的倍数
    dynamic_ratio = round(ratio * 20) / 20

    return {
        'dynamic_ratio': dynamic_ratio,
        'base_coverage_ratio': float(base_coverage_ratio),
        'coverage_analysis': {
            f'{int(k*100)}%': float(v) for k, v in coverage_ratios.items()
        },
        'concentration': {
            'top1_ratio': float(top1_ratio),
            'top3_ratio': float(top3_ratio),
            'top5_ratio': float(top5_ratio),
            'concentration_factor': float(concentration_factor),
        },
        'num_components': num_components,
        'normalized_components': float(normalized_components),
        'position_span': int(position_span),
        'spread_ratio': float(spread_ratio),
        'gini_coefficient': float(gini),
        'adjustments': {
            'component': float(component_adjustment),
            'spread': float(spread_adjustment),
            'gini': float(gini_adjustment),
        },
        'raw_ratio': float(ratio),
        'base_ratio': float(base_ratio),
        'min_ratio': float(min_ratio),
        'max_ratio': float(max_ratio),
    }


def find_connected_components(positions, max_gap=2):
    """
    找到位置列表中的连通分量（相邻 token 群组）

    Args:
        positions: 位置列表
        max_gap: 最大允许的间隔，小于等于这个间隔的位置被认为是连通的

    Returns:
        List of lists, 每个子列表是一个连通分量
    """
    if len(positions) == 0:
        return []

    positions = sorted(positions)
    components = []
    current_component = [positions[0]]

    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= max_gap:
            current_component.append(positions[i])
        else:
            components.append(current_component)
            current_component = [positions[i]]

    components.append(current_component)
    return components


def smart_query_selection(attention_scores, doc_len, target_ratio, system_len, device='cpu', threshold_factor=0.5):
    """
    Smart Query Selection: 使用连通性分析确保相关 token 群组被完整选中

    Args:
        attention_scores: torch.Tensor, shape [doc_len], 每个位置的 attention 分数
        doc_len: 文档长度
        target_ratio: 目标选择比例
        system_len: system prompt 长度
        device: 计算设备
        threshold_factor: 阈值因子，用于确定高 attention 位置 (默认 0.5)
                         较小的值会选择更多位置进入连通分量分析

    Returns:
        List of selected positions (global indices, including system_len offset)
    """
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.float().cpu().numpy()

    target_count = int(doc_len * target_ratio)

    # Step 1: 找到高 attention 位置 (使用可配置的 threshold_factor)
    mean_attn = np.mean(attention_scores)
    std_attn = np.std(attention_scores)
    threshold = mean_attn + threshold_factor * std_attn

    high_attn_positions = list(np.where(attention_scores > threshold)[0])

    # Step 2: 连通分量分析
    components = find_connected_components(high_attn_positions, max_gap=2)

    # Step 3: 计算每个分量的总 attention
    component_scores = []
    for comp in components:
        total_score = sum(attention_scores[p] for p in comp)
        component_scores.append((comp, total_score))

    # Step 4: 按总 attention 排序
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 5: 贪心选择分量 + 上下文扩展 (±1)
    selected = set()

    for comp, total_score in component_scores:
        # 扩展分量边界 (±1)
        extended_comp = set()
        for p in comp:
            for offset in range(-1, 2):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)

        # 检查是否会超过目标 (允许 10% 余量)
        new_positions = extended_comp - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended_comp)

    # Step 6: 补充到目标数量
    if len(selected) < target_count:
        sorted_indices = np.argsort(attention_scores)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(int(pos))
                if len(selected) >= target_count:
                    break

    # Step 7: 如果超过目标，移除最低分的位置
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: attention_scores[p])
        selected.remove(min_pos)

    # 转换为全局索引 (加上 system_len 偏移)
    selected_global = [p + system_len for p in sorted(selected)]

    return selected_global


def entropy_layer_selection(layer_attentions, top_k=4, return_entropy=False):
    """
    基于熵动态选择层（熵越低的层，attention 越集中，信息量可能越大）

    Args:
        layer_attentions: dict {layer_idx: attention_tensor [doc_len]}
                          或 list of (layer_idx, attention_tensor)
        top_k: 选择熵最低的 top_k 层
        return_entropy: 是否返回各层的熵值

    Returns:
        selected_layers: 选中的层索引列表
        layer_entropy: (可选) 各层的熵值字典
    """
    layer_entropy = {}

    # 处理不同输入格式
    if isinstance(layer_attentions, dict):
        items = layer_attentions.items()
    else:
        items = layer_attentions

    for layer_idx, attn in items:
        # 确保是 numpy array
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().float().numpy()

        # 归一化为概率分布
        p = attn / (attn.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)

        # 计算熵 H = -sum(p * log(p))
        entropy = -np.sum(p * np.log(p))
        layer_entropy[layer_idx] = entropy

    # 按熵值排序，选择熵最低的 top_k 层
    sorted_layers = sorted(layer_entropy.items(), key=lambda x: x[1])
    selected_layers = [layer_idx for layer_idx, _ in sorted_layers[:top_k]]

    if return_entropy:
        return selected_layers, layer_entropy
    return selected_layers


def compute_draft_model_attention(draft_model, input_ids, query_start, device="cuda:0",
                                   extra_layers=None):
    """
    用 draft model 完整 prefill 获取 attention 分布（内存优化版本）

    使用 SDPA (Flash Attention) 进行前向传播，只对后 50% 的层计算 query→all 的 attention。
    这样内存从 O(seq_len²) 降到 O(query_len × seq_len)。

    Args:
        draft_model: 小模型
        input_ids: 输入 token ids [1, seq_len]
        query_start: query 的起始位置（只计算 query positions 的 attention）
        device: 设备
        extra_layers: 额外需要计算 attention 的层列表（用于固定层选择）

    Returns:
        layer_attention_scores: {layer_idx: attention_matrix [num_heads, query_len, seq_len]}
    """
    import torch.nn.functional as F

    seq_len = input_ids.shape[1]
    query_len = seq_len - query_start
    num_layers = draft_model.config.num_hidden_layers
    num_heads = draft_model.config.num_attention_heads
    num_kv_heads = draft_model.config.num_key_value_heads
    head_dim = draft_model.config.hidden_size // num_heads

    print(f"\n{'='*60}")
    print("Computing Draft Model Attention (Memory Optimized)")
    print(f"{'='*60}")
    print(f"  Layers: {num_layers}, Heads: {num_heads}")
    print(f"  Seq len: {seq_len}, Query len: {query_len} (from pos {query_start})")
    print(f"  Memory: ~{query_len * seq_len * num_heads * 4 / 1024 / 1024:.1f} MB per layer (vs {seq_len * seq_len * num_heads * 4 / 1024 / 1024:.1f} MB full)")

    layer_attention_scores = {}

    with torch.no_grad():
        inputs_embeds = draft_model.model.embed_tokens(input_ids.to(device))
        hidden_states = inputs_embeds
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # 获取 rotary_emb (兼容不同模型结构)
        if hasattr(draft_model.model, 'rotary_emb'):
            rotary_emb = draft_model.model.rotary_emb
            cos, sin = rotary_emb(hidden_states, position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            use_global_rope = True
        else:
            use_global_rope = False
            cos, sin = None, None

        for layer_idx in range(num_layers):
            layer = draft_model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            # Q, K, V projections
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)

            # Reshape
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            if not use_global_rope:
                cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)

            # Expand K, V for GQA
            n_rep = num_heads // num_kv_heads
            key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
            value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)

            # 对于后 50% 的层（以及 extra_layers 中指定的层），计算 query→all 的 attention
            should_compute_attn = layer_idx >= num_layers // 2
            if extra_layers is not None and layer_idx in extra_layers:
                should_compute_attn = True

            if should_compute_attn:
                # 只计算 query positions 的 attention: [bsz, num_heads, query_len, seq_len]
                query_states_subset = query_states[:, :, query_start:, :]  # [1, num_heads, query_len, head_dim]
                attn_weights_subset = torch.matmul(
                    query_states_subset.float(),
                    key_states_expanded.float().transpose(2, 3)
                ) / (head_dim ** 0.5)

                # Causal mask: 对于 query position i，只能看到 position <= query_start + i
                # 生成正确的 causal mask
                query_positions = torch.arange(query_start, seq_len, device=device)  # [query_len]
                key_positions = torch.arange(seq_len, device=device)  # [seq_len]
                causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)  # [query_len, seq_len]
                attn_weights_subset = attn_weights_subset.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                attn_weights_subset = F.softmax(attn_weights_subset, dim=-1)
                layer_attention_scores[layer_idx] = attn_weights_subset[0].cpu().float().numpy()

                # 计算 attention output（只对 query 部分）
                attn_output_subset = torch.matmul(
                    attn_weights_subset.to(value_states_expanded.dtype),
                    value_states_expanded
                )  # [1, num_heads, query_len, head_dim]

                # 对于前面的 positions，使用 SDPA
                if query_start > 0:
                    query_states_prefix = query_states[:, :, :query_start, :]
                    key_states_prefix = key_states_expanded[:, :, :query_start, :]
                    value_states_prefix = value_states_expanded[:, :, :query_start, :]
                    attn_output_prefix = F.scaled_dot_product_attention(
                        query_states_prefix,
                        key_states_prefix,
                        value_states_prefix,
                        is_causal=True
                    )
                    # 合并 prefix 和 subset
                    attn_output = torch.cat([attn_output_prefix, attn_output_subset], dim=2)
                else:
                    attn_output = attn_output_subset
            else:
                # 前 50% 的层：使用 SDPA（Flash Attention），不存储 attention
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states_expanded,
                    value_states_expanded,
                    is_causal=True
                )

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if layer_idx % 8 == 0 or layer_idx == num_layers - 1:
                print(f"  Layer {layer_idx} done")
    print(f"Draft model attention computed")
    return layer_attention_scores


def prefill_and_save_kv_cache(model, tokenizer, past_key_values, inputs,
                          save_path='', example_id = 0, chunk_id = 0, system_len = 0, passage_len = 0, reprocess_method=None, device="cuda", device_map=None
                          ):

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape

    # Token 统计：记录实际计算的 token 数量
    total_tokens_processed = seq_length  # 实际通过所有层的 token 数量

    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = "cuda:0" if device_map is not None else device
    inputs = inputs.to(input_device)

    tokens = []
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    with torch.no_grad():
        cache_position = torch.arange(seq_length, device=input_device)
        generated_ids = torch.zeros(
            batch_size, seq_length  + 1, dtype=torch.int, device=input_device
        )
        generated_ids[:, cache_position] = inputs.to(input_device).to(torch.int)
        if past_key_values != None:
            past_key_values.cur_idx=cache_position
        start_time = time.time()


        inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
        if reprocess_method == "Cache-Craft" and chunk_id != 0:
            passages_len = [system_len, passage_len]
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
            cachecraft_score = past_key_values.importance_cache[-1] # [num_head, passage_len]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{example_id}_{chunk_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
        past_len = past_key_values.past_tokens[0]
        key_cache = []
        value_cache = []
        if chunk_id == 0:
            # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
            key_cache = [past_key_values.key_cache[i][:,:,:past_len,:].cpu() for i in range(len(past_key_values.key_cache))]
            key_cache = torch.stack(key_cache)
            value_cache = [past_key_values.value_cache[i][:,:,:past_len,:].cpu() for i in range(len(past_key_values.value_cache))]
            value_cache = torch.stack(value_cache)
        else:
            # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
            key_cache = [past_key_values.key_cache[i][:,:,system_len:system_len + passage_len,:].cpu() for i in range(len(past_key_values.key_cache))]
            key_cache = torch.stack(key_cache)
            value_cache = [past_key_values.value_cache[i][:,:,system_len:system_len + passage_len,:].cpu() for i in range(len(past_key_values.value_cache))]
            value_cache = torch.stack(value_cache)
        # Use file lock to prevent concurrent writes from multiple processes
        key_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'
        value_path = f'{save_path}/{example_id}_{chunk_id}_value.pt'
        lock_path = f'{save_path}/{example_id}_{chunk_id}.lock'

        with FileLock(lock_path, timeout=60):
            # Double-check if file exists (another process might have created it)
            if not os.path.exists(key_path):
                torch.save(key_cache.clone(), key_path)
                torch.save(value_cache.clone(), value_path)
                print(f'example_id: {example_id}, chunk_id: {chunk_id} (saved by current process)')
            else:
                print(f'example_id: {example_id}, chunk_id: {chunk_id} (already exists, skipped)')

        return key_cache, value_cache, total_tokens_processed

def decode_one_tokens(model, cur_token, position_ids, cache_position, past_key_values, logits_warper, inputs):
    inputs_embeds = model.model.embed_tokens(cur_token)
    # with torch.cuda.stream(custom_stream):
    logits=model(inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False, use_cache=True)[0]
    if past_key_values != None:
        past_key_values.change_seq_length(1)
    #print(logits)
    next_token_scores = logits_warper(inputs, logits[:, -1, :])
    next_token = torch.argmax(next_token_scores, dim=-1)
    return next_token
# mistral 是这个函数，其他函数得考虑把这个函数换掉
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def prefill_with_cache_and_save_preprocess(model, tokenizer, past_key_values, passages,
                          save_path='', example_id = 0, chunk_id=0, system_len=0, revert_rope=False, reprocess_method=None, device="cuda", device_map=None):

    # load KV
    past_len = past_key_values.past_tokens[0]

    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = "cuda:0" if device_map is not None else device

    # prefill context
    inputs = passages[-1].unsqueeze(0).to(input_device)
    passage_len = passages[-1].shape[0]
    passages_len = [passages[i].shape[0] for i in range(len(passages))]
    batch_size, seq_length = inputs.shape
    cache_position = torch.arange(past_len,past_len + seq_length, device=input_device)
    # position_ids = torch.arange(system_len,system_len + seq_length, device='cuda').unsqueeze(0)
    generated_ids = torch.zeros(
        batch_size, past_len + seq_length + 1, dtype=torch.int, device=input_device
    )
    generated_ids[:, :past_len] = torch.cat(passages[:-1]).unsqueeze(0).to(input_device)
    generated_ids[:, cache_position] = inputs.to(input_device).to(torch.int)
    tokens = []
    with torch.no_grad():
        stream = TextStreamer(tokenizer)
        inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
        if reprocess_method == "Cache-Craft":
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position,
                past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
            cachecraft_score = past_key_values.importance_cache[-1] # [num_head, passage_len]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{example_id}_{chunk_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position,
                past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
    # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
    key_cache = torch.stack([cache.cpu() for cache in past_key_values.key_cache])[:,:,:,past_len:past_len + passage_len,:]
    # Compute RoPE on CPU to match key_cache device
    position_ids = torch.full((1, key_cache[0].shape[2]), system_len - past_len, device='cpu')
    key_cache_for_rope = key_cache[0].to(input_device)
    try:
        cos, sin = model.model.layers[0].self_attn.rotary_emb(key_cache_for_rope, position_ids.to(input_device))
    except:
        cos, sin = model.model.rotary_emb(key_cache_for_rope, position_ids.to(input_device))
    # mistral 限定
    cos = cos.unsqueeze(1).cpu()
    sin = sin.unsqueeze(1).cpu()
    key_cache = (key_cache * cos) + (rotate_half(key_cache) * sin)
    torch.save(key_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_key.pt')
    key_cache = None
    if "cuda" in input_device:
        torch.cuda.empty_cache()
    elif "npu" in input_device:
        torch.npu.empty_cache()
    # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
    value_cache = torch.stack([cache.cpu() for cache in past_key_values.value_cache])[:,:,:,past_len:past_len + passage_len,:]
    torch.save(value_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_value.pt')


def compute_query_doc_similarity(draft_model, input_ids, doc_start, doc_end, query_start, device="cuda:0"):
    """
    计算 query tokens 和 document tokens 的语义相似度

    使用 draft model 中间层的 hidden states 来计算余弦相似度

    Args:
        draft_model: draft model
        input_ids: 输入 token ids [1, seq_len]
        doc_start: 文档起始位置
        doc_end: 文档结束位置
        query_start: query 起始位置
        device: 计算设备

    Returns:
        similarity: (doc_len,) query-doc 相似度分数 [0, 1]
    """
    import torch.nn.functional as F

    draft_model.eval()

    with torch.no_grad():
        # 获取 hidden states
        outputs = draft_model(
            input_ids=input_ids.to(device),
            output_hidden_states=True,
            use_cache=False
        )

        # 使用中间层的 hidden states (比最后一层更通用)
        num_layers = len(outputs.hidden_states)
        mid_layer = num_layers // 2
        hidden_states = outputs.hidden_states[mid_layer][0]  # (seq_len, hidden_dim)

        # Query tokens embedding (平均)
        query_emb = hidden_states[query_start:].mean(dim=0)  # (hidden_dim,)

        # Document tokens embedding
        doc_emb = hidden_states[doc_start:doc_end]  # (doc_len, hidden_dim)

        # 计算余弦相似度
        query_emb = F.normalize(query_emb.unsqueeze(0), dim=-1)  # (1, hidden_dim)
        doc_emb = F.normalize(doc_emb, dim=-1)  # (doc_len, hidden_dim)
        similarity = torch.mm(doc_emb, query_emb.T).squeeze()  # (doc_len,)

        # 转换为正值 [0, 1]
        similarity = (similarity + 1) / 2

    return similarity.cpu()


def load_kv_and_generate(model, tokenizer, past_key_values, passages,
                          load_path='', example_id = 0, max_new_tokens=1, revert_rope=False,
                          reprocess_method='normal', rate=0, preprocess=False, draft_model=None,
                          draft_attention=None, use_entropy_selection=False, entropy_top_k=4,
                          draft_layer_selection='entropy',  # 'entropy', 'last', 'fixed', or 'middle'
                          draft_fixed_layer=3,  # 固定使用哪一层 (当 draft_layer_selection='fixed' 时生效)
                          draft_threshold_factor=0.5,  # smart_query_selection 阈值因子 (default 0.5)
                          use_similarity_rerank=False,  # 使用 query-doc 相似度重排序改进选择
                          rerank_multiplier=2.0,  # 重排序时先选择多少倍候选
                          group=False, device="cuda", chunk_ids=None, device_map=None,
                          vattention_topk_ratio=0.5,  # vAttention: top-k 占总 budget 的比例
                          # OracleDynamic 参数
                          epsilon=0.1,  # 误差容忍度 (如 0.1 = 10% 相对误差)
                          delta=0.05,   # 置信度 (如 0.05 = 95% 置信)
                          min_rate=0.05,  # 动态 budget 的最小比例
                          max_rate=0.5,  # 动态 budget 的最大比例
                          query_text='',  # 用于 DraftModelDynamic 的问题文本
                          # DraftModelLayerwise 参数
                          layerwise_decay='linear',  # 'linear', 'exponential', 'cosine', 'step'
                          layerwise_final_rate=0.05,  # 最后一层的 rate
                          # 文本块1用原始KV cache (prefix cache)
                          original_kv_path=None):  # 原始KV cache路径，用于文本块1 (chunk_id=0)
    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = "cuda:0" if device_map is not None else device

    passages_len = [passage.shape[0] for passage in passages]
    passages_start = [sum(passages_len[:i]) for i in range(1,len(passages_len))]
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
    inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(input_device)
    seq_length = passages[-1][query_prefix_len:].shape[0]

    # 用于存储额外信息（如 OracleDynamic 的动态 rate）
    extra_info = {
        'dynamic_rate': None,
        'topk_coverage': None,
        'topk_count_for_coverage': None,
        'normalized_entropy': None,
        'total_budget': None,
        'doc_len': None,
        'recompute_token_count': None,  # 新增：重算的 token 数量
        'decode_token_count': None      # 新增：解码生成的 token 数量
    }

    # load KV

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    past_len = 0
    system_len = passages[0].shape[0]

    key_cache = []
    value_cache = []
    all_position_ids = [torch.arange(0,system_len).unsqueeze(0).to(input_device)]

    # If chunk_ids is provided, use it; otherwise use sequential indices (backward compatible)
    if chunk_ids is None:
        chunk_ids = list(range(len(passages) - 1))

    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]

        # 对于 chunk_id=0 (system + 文本块1)，如果提供了 original_kv_path，则从原始路径加载
        # 这样文本块1可以使用没有 preprocess 的 KV cache (prefix cache hit)
        if chunk_id <= 1 and original_kv_path is not None:
            kv_path = original_kv_path
        else:
            kv_path = load_path

        chunk_key_cache = torch.load(f'{kv_path}/{example_id}_{chunk_id}_key.pt',weights_only=True).to('cpu')
        chunk_value_cache = torch.load(f'{kv_path}/{example_id}_{chunk_id}_value.pt',weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)
    start_time = time.time()
    
    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]
        key_cache[idx] = key_cache[idx].to(input_device)
        chunk_key_cache = key_cache[idx]
        chunk_value_cache = value_cache[idx].to(input_device)
        assert passage_len == chunk_key_cache.shape[3]
        if revert_rope and chunk_id > 0:
            all_position_ids = []
            # Get the device of the rotary embedding layer from inv_freq buffer
            rotary_emb = model.model.layers[0].self_attn.rotary_emb
            if hasattr(rotary_emb, 'inv_freq') and rotary_emb.inv_freq is not None:
                rotary_device = rotary_emb.inv_freq.device
            else:
                # Fallback: use the device of the first layer
                rotary_device = next(model.model.layers[0].parameters()).device

            position_ids = torch.full((1, chunk_key_cache[0].shape[2]), past_len - system_len, device=rotary_device)
            chunk_key_for_rope = chunk_key_cache[0].to(rotary_device)
            cos, sin = rotary_emb(chunk_key_for_rope, position_ids)
            # mistral 限定
            cos = cos.unsqueeze(1).to(input_device)
            sin = sin.unsqueeze(1).to(input_device)
            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)
        elif chunk_id > 0:
            all_position_ids.append(torch.arange(system_len,system_len+passage_len).to(input_device).unsqueeze(0))

        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx].narrow(2,past_len,passage_len).copy_(chunk_key_cache[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2,past_len,passage_len).copy_(chunk_value_cache[layer_idx])
            past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len
    storage_time = time.time() - start_time 
    print(f'storage_time: {storage_time}')
    if rate != 0:
        if reprocess_method == 'cacheBlend':
            without_attn_key = past_key_values.key_cache[1].narrow(2,0,past_len).clone()
            without_attn_value = past_key_values.value_cache[1].narrow(2,0,past_len).clone()
            inputs = torch.cat(passages[:-1]).to(input_device).unsqueeze(0)
            # 这里会在终端上多输出一次
            _, tmp_past_key_value, _ = prefill_and_generate(model, tokenizer, inputs, max_new_tokens=1, device=input_device, early_exit_layer=2,device_map=device_map)
            with_attn_key = tmp_past_key_value.key_cache[1].narrow(2,0,past_len).clone()
            with_attn_value = tmp_past_key_value.value_cache[1].narrow(2,0,past_len).clone()
            v_sub_all = without_attn_value - with_attn_value
            v_sub_all = v_sub_all.squeeze(0)
            v_sub_all = v_sub_all.transpose(0, 1)
            v_sum = torch.sum(v_sub_all**2, dim=[1,2])
            v_sum = v_sum[system_len:]
            v_need_index = torch.topk(v_sum,int(rate*(past_len - system_len))).indices.to('cpu')
            v_need_index = v_need_index + system_len

            # k_sub_all = without_attn_key - with_attn_key
            # k_sub_all = torch.abs(k_sub_all)
            # k_sub_all = k_sub_all.squeeze(0)
            # k_sub_all = k_sub_all.transpose(0, 1)
            # k_sub_all = k_sub_all.reshape(past_len,-1)
            # k_sum = torch.sum(k_sub_all,dim=1)
            # k_sum = k_sum.tolist()
            # k_sum = k_sum[system_len:]
            # k_sum = torch.tensor(k_sum,device=k_sub_all.device)
            # k_need_index = torch.topk(k_sum,int(rate*(past_len - system_len))).indices.to('cpu')
            # k_need_index = k_need_index + system_len
            k_need_index = v_need_index
        elif reprocess_method == 'FusionRAG':
            select_time = time.time()
            query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
            if query_prefix_len >= len(passages[-1]):
                query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0]))+1
            inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(input_device)
            seq_length = passages[-1][query_prefix_len:].shape[0]

            cache_position = torch.arange(past_len, past_len+seq_length, device=input_device)
            with torch.no_grad():
                ss_time = time.perf_counter()
                inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
                model(
                    inputs_embeds = inputs_embeds, past_key_values=past_key_values,
                    cache_position=cache_position, reprocess_method=reprocess_method, 
                    return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache
                    )
                
                k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:cache_position[0]]
                end_time = time.perf_counter() - ss_time
                if group:
                    k_sum_relevant = k_sum[system_len:]  # 只看中间文本块的分数
                    k_sum_relevant = torch.tensor(k_sum_relevant, device=input_device)

                    # 计算需要选择的 token 数量
                    total_relevant_tokens = torch.cat(passages[1:-1]).shape[0]  # 中间文本块的总 token 数
                    k_lens = int(rate * total_relevant_tokens)

                    # === 新增：按组选择逻辑 ===
                    group_size = 16
                    num_groups = (total_relevant_tokens + group_size - 1) // group_size  # 向上取整

                    # 将分数按组重塑（最后一组可能不足 8 个）
                    # 先 pad 到能被 group_size 整除
                    padded_length = num_groups * group_size
                    if total_relevant_tokens < padded_length:
                        # 用很小的负数填充，确保不会被选中
                        padding = torch.full((padded_length - total_relevant_tokens,),
                                            -float('inf'), device=input_device)
                        k_sum_padded = torch.cat([k_sum_relevant, padding])
                    else:
                        k_sum_padded = k_sum_relevant

                    # 重塑为 [num_groups, group_size]
                    k_sum_grouped = k_sum_padded.view(num_groups, group_size)

                    # 计算每组的最大分数
                    group_max_scores, _ = torch.max(k_sum_grouped, dim=1)  # [num_groups]

                    # 根据组的最大分数选择 top-k 组
                    num_groups_to_select = (k_lens + group_size - 1) // group_size  # 向上取整
                    num_groups_to_select = min(num_groups_to_select, num_groups)  # 不超过总组数

                    top_group_indices = torch.topk(group_max_scores, num_groups_to_select).indices

                    # 将选中的组展开为 token 索引
                    selected_token_indices = []
                    for group_idx in top_group_indices.tolist():
                        start_idx = group_idx * group_size
                        end_idx = min(start_idx + group_size, total_relevant_tokens)
                        selected_token_indices.extend(range(start_idx, end_idx))

                    # 转换为 tensor 并加上 system_len 偏移
                    k_need_index = torch.tensor(selected_token_indices, device='cpu') + system_len

                    print(f"选择了 {len(selected_token_indices)} 个 tokens，"
                          f"来自 {len(top_group_indices)} 个组 (目标: {k_lens} tokens)")
                else:
                    k_sum = k_sum.tolist()
                    k_sum = k_sum[system_len:]
                    k_sum = torch.tensor(k_sum,device=input_device)
                    k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
                    k_need_index = torch.topk(k_sum, k_lens).indices.to('cpu')
                    k_need_index = k_need_index + system_len
                    print(f'select_time: {time.time() - select_time}')
        elif reprocess_method == 'frontRow':
            k_need_index = []
            for i in range(len(passages_start[:-1])):
                k_need_index.extend(range(passages_start[i], passages_start[i] + int(passages_len[i+1]*rate)))
            k_need_index = torch.tensor(k_need_index)
        elif reprocess_method == "Cache-Craft":
            import os
            save_prefix_path_list = [f"{load_path}/cachecraftattn_{example_id}_{i}.pt" for i in range(1,len(passages)-1)]
            chunk_score_list = []
            for file in save_prefix_path_list:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"未找到 cache-craft 文件: {file}")
                tensor = torch.load(file, weights_only=True, map_location="cpu")
                chunk_score_list.append(tensor)
            chunk_score = torch.cat(chunk_score_list, dim=0)
            assert chunk_score.shape[0] == sum(passages_len[1:-1])
            k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
            k_need_index = torch.topk(chunk_score, k_lens).indices.to('cpu')
            k_need_index = k_need_index + system_len
        
        elif reprocess_method == "speculative_prefill":
            inputs = torch.cat(passages).to(input_device).unsqueeze(0)
            cache_position = torch.arange(0, inputs.shape[1], device=input_device)
            # Pass device_map if multi-GPU, otherwise pass device
            cache_device = device_map if device_map is not None else input_device
            tmp_past_key_values = StaticCache(
                                    config = model.config, max_batch_size = 1,
                                    max_cache_len = inputs.shape[1], device = cache_device, dtype = model.dtype,
                                    passage_len=torch.cat(passages[:-1]).shape[0],
                                )
            with torch.no_grad():
                ss_time = time.perf_counter()
                inputs_embeds = draft_model.model.embed_tokens(inputs).to(input_device)
                draft_model(
                    inputs_embeds = inputs_embeds, past_key_values=tmp_past_key_values,
                    cache_position=cache_position, reprocess_method=reprocess_method, 
                    return_dict=False, use_cache=True, passages_len=passages_len
                    )

                # 获取重要性分数
                k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:past_len]
                if group:
                    k_sum_relevant = k_sum[system_len:]  # 只看中间文本块的分数
                    k_sum_relevant = torch.tensor(k_sum_relevant, device=input_device)

                    # 计算需要选择的 token 数量
                    total_relevant_tokens = torch.cat(passages[1:-1]).shape[0]  # 中间文本块的总 token 数
                    k_lens = int(rate * total_relevant_tokens)

                    # === 新增：按组选择逻辑 ===
                    group_size = 16
                    num_groups = (total_relevant_tokens + group_size - 1) // group_size  # 向上取整

                    # 将分数按组重塑（最后一组可能不足 8 个）
                    # 先 pad 到能被 group_size 整除
                    padded_length = num_groups * group_size
                    if total_relevant_tokens < padded_length:
                        # 用很小的负数填充，确保不会被选中
                        padding = torch.full((padded_length - total_relevant_tokens,),
                                            -float('inf'), device=input_device)
                        k_sum_padded = torch.cat([k_sum_relevant, padding])
                    else:
                        k_sum_padded = k_sum_relevant

                    # 重塑为 [num_groups, group_size]
                    k_sum_grouped = k_sum_padded.view(num_groups, group_size)

                    # 计算每组的最大分数
                    group_max_scores, _ = torch.max(k_sum_grouped, dim=1)  # [num_groups]

                    # 根据组的最大分数选择 top-k 组
                    num_groups_to_select = (k_lens + group_size - 1) // group_size  # 向上取整
                    num_groups_to_select = min(num_groups_to_select, num_groups)  # 不超过总组数

                    top_group_indices = torch.topk(group_max_scores, num_groups_to_select).indices

                    # 将选中的组展开为 token 索引
                    selected_token_indices = []
                    for group_idx in top_group_indices.tolist():
                        start_idx = group_idx * group_size
                        end_idx = min(start_idx + group_size, total_relevant_tokens)
                        selected_token_indices.extend(range(start_idx, end_idx))

                    # 转换为 tensor 并加上 system_len 偏移
                    k_need_index = torch.tensor(selected_token_indices, device='cpu') + system_len

                    print(f"选择了 {len(selected_token_indices)} 个 tokens，"
                          f"来自 {len(top_group_indices)} 个组 (目标: {k_lens} tokens)")
                else:
                    k_sum = k_sum.tolist()
                    k_sum = k_sum[system_len:]
                    k_sum = torch.tensor(k_sum,device=input_device)
                    k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
                    k_need_index = torch.topk(k_sum, k_lens).indices.to('cpu')
                    k_need_index = k_need_index + system_len
        elif reprocess_method == 'QueryAttention':
            # Smart Query Selection: 使用 query attention + 连通分量分析
            select_time = time.time()

            # 获取 query tokens
            inputs = passages[-1][:].unsqueeze(0).to(input_device)
            seq_length = passages[-1][:].shape[0]

            cache_position = torch.arange(past_len, past_len + seq_length, device=input_device)

            with torch.no_grad():
                inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
                model(
                    inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                    cache_position=cache_position, reprocess_method='QueryAttention',
                    return_dict=False, use_cache=True, passages_len=passages_len
                )

                # 获取文档部分的 attention 分数
                num_layers = len(past_key_values.importance_cache)
                # 文本块1 长度 (用于 prefix cache，不参与重算)
                text_block1_len = passages_len[1]
                # 从文本块2开始选择 (文本块1直接用原始KV cache)
                doc_len = sum(passages_len[2:-1])
                selection_start = system_len + text_block1_len

                # 收集所有候选层的 attention（后 1/2 的层，用于熵选层）
                # 注意: 从 selection_start 开始，长度为 doc_len（只包括文本块2到文本块n）
                candidate_start = num_layers // 2
                layer_attention_dict = {}
                for layer_idx in range(candidate_start, num_layers):
                    layer_attn = past_key_values.importance_cache[layer_idx][:, selection_start:selection_start + doc_len]
                    layer_attn_avg = layer_attn.mean(dim=0).to(input_device)  # [doc_len]
                    layer_attention_dict[layer_idx] = layer_attn_avg

                # 选择使用的层
                if use_entropy_selection:
                    # 基于熵动态选层
                    active_layers, layer_entropy = entropy_layer_selection(
                        layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                    )
                    print(f"  熵选层: 选择了 {active_layers} (熵最低的 {entropy_top_k} 层)")
                else:
                    # 默认使用后 1/4 的层
                    start_layer = num_layers * 3 // 4
                    active_layers = list(range(start_layer, num_layers))

                # 聚合选中层的 attention
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]

                # 使用 smart_query_selection 进行选择
                # 注意: selection_start 是选择区域的起始位置（跳过了 system 和 文本块1）
                selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=rate,
                    system_len=selection_start,  # 使用 selection_start 作为偏移量
                    device=input_device
                )
                # 转成 tensor 以与后续 torch.sort 兼容
                k_need_index = torch.tensor(selected_indices, device='cpu')

                print(f"QueryAttention 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
                print(f"使用了 {len(active_layers)} 个层: {active_layers}")
                print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'DraftModel':
            # DraftModel: 用小模型 prefill 获取 attention，指导 token 选择
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache，不参与重算选择)
            # doc_len = 文本块2 + 文本块3 + ... + 文本块n
            doc_len = sum(passages_len[1:-1])
            # selection_start 跳过 system_prompt 和 文本块1
            selection_start = system_len 

            # 如果没有传入 draft_attention，需要用 draft_model 计算
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided for DraftModel method")
                # 构建完整输入
                full_input = torch.cat(passages).unsqueeze(0).to(input_device)
                # 如果使用固定层且层号在前 50%，需要额外计算该层的 attention
                extra_layers = None
                if draft_layer_selection == 'fixed':
                    num_layers = draft_model.config.num_hidden_layers
                    if draft_fixed_layer < num_layers // 2:
                        extra_layers = [draft_fixed_layer]
                        print(f"  固定层 {draft_fixed_layer} 在前半部分，额外计算其 attention")
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, input_device, extra_layers=extra_layers)
                torch.cuda.empty_cache()

            # draft_attention 是 {layer_idx: attention [num_heads, query_len, seq_len]} 格式（内存优化版本）
            # 已经只包含 query positions 的 attention，不需要再切片 query 维度

            # 收集各层的 query→doc attention
            # 注意: 从 selection_start 开始，长度为 doc_len（包括文本块1到文本块n）
            layer_attention_dict = {}
            for layer_idx, layer_attn in draft_attention.items():
                # layer_attn: [num_heads, query_len, seq_len]
                # 提取 query→doc attention（只切片 key 维度，跳过 system 和 文本块1）
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                # 对 heads 和 query positions 平均
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层
            if draft_layer_selection == 'entropy':
                # 基于熵动态选层
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  DraftModel 熵选层: 选择了 {active_layers}")
                # 聚合选中层的 attention
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]
            elif draft_layer_selection == 'last':
                # 只使用最后一层
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  DraftModel 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            elif draft_layer_selection == 'fixed':
                # 使用固定层
                if draft_fixed_layer in layer_attention_dict:
                    active_layers = [draft_fixed_layer]
                    print(f"  DraftModel 使用固定层: Layer {draft_fixed_layer}")
                    multi_layer_attn = layer_attention_dict[draft_fixed_layer]
                else:
                    # 如果指定层不在 attention dict 中，回退到使用最接近的层
                    available_layers = sorted(layer_attention_dict.keys())
                    closest_layer = min(available_layers, key=lambda x: abs(x - draft_fixed_layer))
                    active_layers = [closest_layer]
                    print(f"  DraftModel 固定层 {draft_fixed_layer} 不可用, 使用最接近的层: Layer {closest_layer}")
                    multi_layer_attn = layer_attention_dict[closest_layer]
            elif draft_layer_selection == 'middle':
                # 使用中间层 (40%-60% 位置的层)
                # 实验发现中间层与 7B 模型选择更相似
                available_layers = sorted(layer_attention_dict.keys())
                num_layers = draft_model.config.num_hidden_layers if draft_model is not None else max(available_layers) + 1
                mid_start = int(0.4 * num_layers)
                mid_end = int(0.6 * num_layers)
                middle_layers = [l for l in range(mid_start, mid_end + 1) if l in available_layers]
                if len(middle_layers) >= entropy_top_k:
                    active_layers = middle_layers[:entropy_top_k]
                elif len(middle_layers) > 0:
                    active_layers = middle_layers
                else:
                    # fallback: 如果中间层不可用，使用可用层中最接近中间的
                    mid_point = num_layers // 2
                    active_layers = sorted(available_layers, key=lambda x: abs(x - mid_point))[:entropy_top_k]
                print(f"  DraftModel 使用中间层: {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}, expected 'entropy', 'last', 'fixed', or 'middle'")

            # 使用 smart_query_selection 进行选择
            # 注意: selection_start 是选择区域的起始位置（跳过了 system 和 文本块1）

            if use_similarity_rerank and draft_model is not None:
                # 使用相似度重排序改进选择
                # 关键改进: 先用 smart_query_selection 选候选，保留连通分量和边界扩展
                print(f"  使用相似度重排序 (multiplier={rerank_multiplier})...")

                # 计算 query-doc 相似度
                similarity_scores = compute_query_doc_similarity(
                    draft_model, full_input, selection_start, selection_start + doc_len,
                    query_start, input_device
                )

                target_count = int(doc_len * rate)

                # 先用 smart_query_selection 选择 rerank_multiplier 倍候选
                # 这样保留了连通分量分析和边界扩展的优势
                candidate_ratio = min(rate * rerank_multiplier, 1.0)
                candidates_global = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=candidate_ratio,
                    system_len=selection_start,
                    device=input_device,
                    threshold_factor=draft_threshold_factor
                )

                # 转换为相对于 doc 的位置
                candidates_local = [pos - selection_start for pos in candidates_global]

                # 在候选中按相似度排序
                candidate_sim = [(pos, similarity_scores[pos].item()) for pos in candidates_local]
                candidate_sim.sort(key=lambda x: x[1], reverse=True)

                # 选择相似度最高的 target_count 个
                selected_local = [pos for pos, _ in candidate_sim[:target_count]]
                selected_indices = [pos + selection_start for pos in sorted(selected_local)]

                print(f"  相似度重排序完成: {len(candidates_global)} 候选 -> {len(selected_indices)} 最终选择")
            else:
                # 原始方法 (使用可配置的 threshold_factor)
                selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=rate,
                    system_len=selection_start,  # 使用 selection_start 作为偏移量
                    device=input_device,
                    threshold_factor=draft_threshold_factor
                )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"DraftModel 选择了 {len(k_need_index)} 个 tokens (从文本块2-n中选{len(k_need_index)/doc_len*100:.1f}%), 文本块1({text_block1_len}tokens)用prefix cache, threshold={draft_threshold_factor}")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'DynamicDraftModel':
            # DynamicDraftModel: 基于 3B 模型 attention 分布特征的动态重算比例
            #
            # 核心发现（来自 Type A vs Type B 分析，effect size > 0.4）：
            # - Type B（需要高rate）: attention 更分散，peak 更低，concentration 更低
            # - Type A（不需要高rate）: attention 更集中，peak 更高
            #
            # 正确策略：
            # - 当 attention 分散（peak 低，concentration 低，entropy 高）→ 需要高 rate
            # - 当 attention 集中 → 可以用低 rate
            #
            # 关键指标（按 effect size 排序）：
            # 1. top10_concentration: Type A=0.1448, Type B=0.1284 (effect=0.615)
            # 2. coverage_50: Type A=0.0659, Type B=0.0760 (effect=0.590)
            # 3. peak_strength: Type A=0.0044, Type B=0.0038 (effect=0.526)

            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            # 如果没有传入 draft_attention，需要用 draft_model 计算
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided for DynamicDraftModel method")
                full_input = torch.cat(passages).unsqueeze(0).to(input_device)
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, input_device)
                torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            layer_attention_dict = {}
            all_layer_attentions = []

            for layer_idx, layer_attn in draft_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)
                all_layer_attentions.append(doc_attention_avg)

            # 基于熵选层
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  DynamicDraftModel 熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  DynamicDraftModel 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                # 默认使用熵选层
                active_layers, _ = entropy_layer_selection(layer_attention_dict, top_k=entropy_top_k)
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)

            # =========================================================================
            # 计算 3B 模型的不确定性特征
            # =========================================================================
            aggregated_attn = multi_layer_attn.cpu()
            uncertainty_features = {}

            # Feature 1: Peak strength (attention 峰值强度)
            # 低 peak = attention 分散 → 需要高 rate (Type B: 0.0038, Type A: 0.0044)
            peak_strength = aggregated_attn.max().item()
            uncertainty_features['peak_strength'] = peak_strength

            # Feature 2: Top-10 concentration (前10个token占总attention的比例)
            # 低 concentration = attention 分散 → 需要高 rate (Type B: 0.1284, Type A: 0.1448)
            sorted_attn, _ = torch.sort(aggregated_attn, descending=True)
            total = sorted_attn.sum()
            top10_concentration = sorted_attn[:10].sum().item() / total.item() if total > 0 else 0
            uncertainty_features['top10_concentration'] = top10_concentration

            # Feature 3: Coverage 50% ratio (覆盖50% attention所需token比例)
            # 高 coverage = attention 分散 → 需要高 rate (Type B: 0.0760, Type A: 0.0659)
            cumsum = torch.cumsum(sorted_attn, dim=0)
            coverage_50 = (cumsum >= 0.5 * total).nonzero(as_tuple=True)[0]
            if len(coverage_50) > 0:
                tokens_for_50 = coverage_50[0].item() + 1
            else:
                tokens_for_50 = len(aggregated_attn)
            coverage_50_ratio = tokens_for_50 / len(aggregated_attn)
            uncertainty_features['coverage_50_ratio'] = coverage_50_ratio

            # Feature 4: Normalized entropy
            # 高 entropy = attention 分散 → 需要高 rate (Type B: 0.8610, Type A: 0.8504)
            p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
            p = torch.clamp(p, min=1e-10)
            entropy = -(p * torch.log(p)).sum()
            max_entropy = np.log(len(aggregated_attn))
            normalized_entropy = (entropy / max_entropy).item()
            uncertainty_features['normalized_entropy'] = normalized_entropy

            # 保留 layer consistency 用于分析
            top_k_per_layer = min(50, doc_len)
            layer_top_tokens = []
            for idx in active_layers:
                top_indices = torch.topk(layer_attention_dict[idx], top_k_per_layer).indices
                layer_top_tokens.append(set(top_indices.tolist()))

            layer_consistency_scores = []
            for i in range(len(layer_top_tokens) - 1):
                intersection = len(layer_top_tokens[i] & layer_top_tokens[i+1])
                union = len(layer_top_tokens[i] | layer_top_tokens[i+1])
                if union > 0:
                    layer_consistency_scores.append(intersection / union)

            layer_consistency = np.mean(layer_consistency_scores) if layer_consistency_scores else 0.5
            uncertainty_features['layer_consistency'] = layer_consistency

            # =========================================================================
            # 基于 attention 分散程度计算动态 rate
            # =========================================================================
            # 阈值（来自 Type A vs Type B 分析，取建议阈值）
            # Type A (不需要高rate): peak=0.0044, top10=0.1448, coverage_50=0.0659, entropy=0.8504
            # Type B (需要高rate):   peak=0.0038, top10=0.1284, coverage_50=0.0760, entropy=0.8610

            peak_threshold = 0.0041  # 低于此值 → 需要高 rate
            top10_threshold = 0.1366  # 低于此值 → 需要高 rate
            coverage_50_threshold = 0.0710  # 高于此值 → 需要高 rate
            entropy_threshold = 0.8557  # 高于此值 → 需要高 rate

            # 计算 dispersion score (0-1)
            # 越高表示 attention 越分散，需要越高的 rate
            dispersion = 0.0
            reasons = []

            # Peak strength contribution (低 peak = 分散)
            # effect size: 0.526, weight: 0.30
            if peak_strength < peak_threshold:
                peak_factor = min((peak_threshold - peak_strength) / 0.001, 1.0)
                dispersion += 0.30 * peak_factor
                reasons.append(f"low_peak({peak_strength:.4f})")

            # Top-10 concentration contribution (低 concentration = 分散)
            # effect size: 0.615 (最高), weight: 0.35
            if top10_concentration < top10_threshold:
                conc_factor = min((top10_threshold - top10_concentration) / 0.02, 1.0)
                dispersion += 0.35 * conc_factor
                reasons.append(f"low_top10({top10_concentration:.4f})")

            # Coverage 50% contribution (高 coverage = 分散)
            # effect size: 0.590, weight: 0.25
            if coverage_50_ratio > coverage_50_threshold:
                cov_factor = min((coverage_50_ratio - coverage_50_threshold) / 0.02, 1.0)
                dispersion += 0.25 * cov_factor
                reasons.append(f"high_cov50({coverage_50_ratio:.4f})")

            # Normalized entropy contribution (高 entropy = 分散)
            # effect size: 0.470, weight: 0.10
            if normalized_entropy > entropy_threshold:
                ent_factor = min((normalized_entropy - entropy_threshold) / 0.02, 1.0)
                dispersion += 0.10 * ent_factor
                reasons.append(f"high_entropy({normalized_entropy:.4f})")

            # Map dispersion to rate
            # rate 作为 base_rate，min_rate 和 max_rate 作为范围
            base_rate = rate
            dynamic_rate = base_rate + dispersion * (max_rate - base_rate)
            dynamic_rate = max(min_rate, min(max_rate, dynamic_rate))

            reason_str = ", ".join(reasons) if reasons else "concentrated"

            print(f"\n  DynamicDraftModel Attention Dispersion Features:")
            print(f"    peak_strength: {peak_strength:.5f} (threshold: <{peak_threshold})")
            print(f"    top10_concentration: {top10_concentration:.4f} (threshold: <{top10_threshold})")
            print(f"    coverage_50_ratio: {coverage_50_ratio:.4f} (threshold: >{coverage_50_threshold})")
            print(f"    normalized_entropy: {normalized_entropy:.4f} (threshold: >{entropy_threshold})")
            print(f"    dispersion_score: {dispersion:.3f}")
            print(f"  Dynamic rate: {dynamic_rate:.3f} ({reason_str})")
            print(f"  Rate range: base={base_rate}, min={min_rate}, max={max_rate}")

            # 使用动态计算的 rate 进行 token 选择
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=dynamic_rate,
                system_len=selection_start,
                device=input_device,
                threshold_factor=draft_threshold_factor
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            # 保存动态 rate 信息
            extra_info['dynamic_rate'] = dynamic_rate
            extra_info['dispersion_score'] = dispersion
            extra_info['peak_strength'] = peak_strength
            extra_info['top10_concentration'] = top10_concentration
            extra_info['coverage_50_ratio'] = coverage_50_ratio
            extra_info['normalized_entropy'] = normalized_entropy
            extra_info['layer_consistency'] = layer_consistency
            extra_info['doc_len'] = doc_len
            extra_info['total_budget'] = len(k_need_index)

            print(f"DynamicDraftModel 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'DraftModelDynamic':
            # DraftModelDynamic: 用小模型 prefill 获取 attention，动态计算重算比例
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            # 如果没有传入 draft_attention，需要用 draft_model 计算
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided for DraftModelDynamic method")
                full_input = torch.cat(passages).unsqueeze(0).to(input_device)
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, input_device)
                torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            layer_attention_dict = {}
            all_layer_attentions = []  # 用于动态rate计算

            for layer_idx, layer_attn in draft_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)
                all_layer_attentions.append(doc_attention_avg)

            # 基于熵选层
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  DraftModelDynamic 熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  DraftModelDynamic 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            elif draft_layer_selection == 'middle':
                # 使用中间层 (40%-60% 位置的层)
                available_layers = sorted(layer_attention_dict.keys())
                num_layers = draft_model.config.num_hidden_layers if draft_model is not None else max(available_layers) + 1
                mid_start = int(0.4 * num_layers)
                mid_end = int(0.6 * num_layers)
                middle_layers = [l for l in range(mid_start, mid_end + 1) if l in available_layers]
                if len(middle_layers) >= entropy_top_k:
                    active_layers = middle_layers[:entropy_top_k]
                elif len(middle_layers) > 0:
                    active_layers = middle_layers
                else:
                    mid_point = num_layers // 2
                    active_layers = sorted(available_layers, key=lambda x: abs(x - mid_point))[:entropy_top_k]
                print(f"  DraftModelDynamic 使用中间层: {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'fixed':
                if draft_fixed_layer in layer_attention_dict:
                    active_layers = [draft_fixed_layer]
                    print(f"  DraftModelDynamic 使用固定层: Layer {draft_fixed_layer}")
                    multi_layer_attn = layer_attention_dict[draft_fixed_layer]
                else:
                    available_layers = sorted(layer_attention_dict.keys())
                    closest_layer = min(available_layers, key=lambda x: abs(x - draft_fixed_layer))
                    active_layers = [closest_layer]
                    print(f"  DraftModelDynamic 固定层 {draft_fixed_layer} 不可用, 使用最接近的层: Layer {closest_layer}")
                    multi_layer_attn = layer_attention_dict[closest_layer]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}")

            # =========================================================================
            # 动态计算 rate
            # =========================================================================
            aggregated_attn = multi_layer_attn.cpu().numpy()
            attn_features = {}

            # Feature 1: 归一化熵
            p_agg = aggregated_attn / (aggregated_attn.sum() + 1e-10)
            p_agg = np.clip(p_agg, 1e-10, 1.0)
            attn_entropy = -np.sum(p_agg * np.log(p_agg))
            max_entropy = np.log(doc_len) if doc_len > 0 else 1
            attn_features['normalized_entropy'] = attn_entropy / max_entropy

            # Feature 2: Coverage ratio
            sorted_indices = np.argsort(aggregated_attn)[::-1]
            sorted_attn = aggregated_attn[sorted_indices]
            cumsum = np.cumsum(sorted_attn) / (sorted_attn.sum() + 1e-10)
            coverage_count = np.searchsorted(cumsum, 0.85) + 1
            attn_features['coverage_85_ratio'] = coverage_count / doc_len

            # Feature 3: Gini coefficient
            sorted_attn_asc = np.sort(aggregated_attn)
            n = len(sorted_attn_asc)
            index = np.arange(1, n + 1)
            gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
            attn_features['gini_coefficient'] = gini

            # Feature 4: 连通分量数量
            mean_attn = np.mean(aggregated_attn)
            std_attn = np.std(aggregated_attn)
            threshold_positions = list(np.where(aggregated_attn > mean_attn + 0.5 * std_attn)[0])
            components = find_connected_components(threshold_positions, max_gap=2)
            attn_features['num_components'] = len(components)

            # Feature 5: 高attention位置跨度
            high_positions = np.where(aggregated_attn > mean_attn + std_attn)[0]
            if len(high_positions) > 1:
                attn_features['high_attn_span'] = (high_positions.max() - high_positions.min()) / doc_len
            else:
                attn_features['high_attn_span'] = 0

            # =========================================================================
            # 动态计算 rate（范围 min_rate ~ max_rate）
            # =========================================================================
            #
            # 经过大量分析发现：
            # - Attention 特征（gini, coverage, cross_doc_entropy 等）区分力都很弱
            # - Easy 和 Medium 问题的特征分布高度重叠（分离度 < 0.5）
            # - 问题的"难度"并不反映在 attention 分布中
            #
            # 因此采用简单优雅的公式，基于 coverage 的物理意义：
            #
            #   rate = coverage_90_ratio
            #
            # 物理意义：
            #   "达到 90% attention 覆盖需要多少比例的 token，就用多少比例去重算"
            #
            # 这个公式简单、有理论依据，且不依赖于复杂的特征工程

            # 计算 coverage_90
            coverage_90_count = np.searchsorted(cumsum, 0.90) + 1
            coverage_90_ratio = coverage_90_count / doc_len
            attn_features['coverage_90_ratio'] = coverage_90_ratio

            # 简洁公式：rate = coverage_90_ratio
            # 但由于 coverage 计算的范围和实际需要的范围有差异，需要缩放
            # 经验缩放系数：0.7（因为 coverage_90 平均约 40%，而最优 rate 约 20-30%）
            scale_factor = 0.7

            dynamic_rate = coverage_90_ratio * scale_factor

            # 确保在范围内
            dynamic_rate = np.clip(dynamic_rate, min_rate, max_rate)

            print(f"\n  DraftModelDynamic Features:")
            print(f"    normalized_entropy: {attn_features['normalized_entropy']:.3f}")
            print(f"    coverage_85_ratio: {attn_features['coverage_85_ratio']:.3f}")
            print(f"    coverage_90_ratio: {coverage_90_ratio:.3f}")
            print(f"    gini_coefficient: {attn_features['gini_coefficient']:.3f}")
            print(f"    num_components: {attn_features['num_components']}")
            print(f"    high_attn_span: {attn_features['high_attn_span']:.3f}")
            print(f"  Dynamic rate: {dynamic_rate:.3f} (min={min_rate}, max={max_rate})")

            # 使用动态计算的 rate 进行 token 选择
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=dynamic_rate,
                system_len=selection_start,
                device=input_device
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            # 保存动态 rate 信息
            extra_info['dynamic_rate'] = dynamic_rate
            extra_info['normalized_entropy'] = attn_features['normalized_entropy']
            extra_info['topk_coverage'] = attn_features['coverage_85_ratio']
            extra_info['topk_count_for_coverage'] = coverage_count
            extra_info['doc_len'] = doc_len
            extra_info['total_budget'] = len(k_need_index)

            print(f"DraftModelDynamic 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'DraftModelLayerwise':
            # DraftModelLayerwise: 用小模型 attention 指导选择，但每层使用不同的重算比例
            # 逐层递减策略: 第0层用 initial_rate，逐层递减到 final_rate
            select_time = time.time()

            # 逐层参数已通过函数参数传入: layerwise_decay, layerwise_final_rate
            initial_rate = rate  # rate 参数作为 initial_rate

            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            print(f"\n{'='*60}")
            print("DraftModelLayerwise: Layer-wise Dynamic Rate")
            print(f"{'='*60}")
            print(f"  Initial rate: {initial_rate:.1%}")
            print(f"  Final rate: {layerwise_final_rate:.1%}")
            print(f"  Decay type: {layerwise_decay}")

            # 计算 draft model attention
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided")
                full_input = torch.cat(passages).unsqueeze(0).to(input_device)
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, input_device)
                torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            layer_attention_dict = {}
            for layer_idx, layer_attn in draft_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 基于熵选层
            if draft_layer_selection == 'entropy':
                active_layers, _ = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  Draft 熵选层: {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            else:
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]

            # 计算每层的 rate (使用递减策略)
            num_layers = model.config.num_hidden_layers

            def compute_layer_rate(layer_idx, num_layers, initial, final, decay_type):
                progress = layer_idx / (num_layers - 1) if num_layers > 1 else 0
                if decay_type == "linear":
                    return initial - (initial - final) * progress
                elif decay_type == "exponential":
                    ratio = final / initial if initial > 0 else 1
                    return initial * (ratio ** progress)
                elif decay_type == "cosine":
                    return final + (initial - final) * (1 + np.cos(np.pi * progress)) / 2
                else:  # step
                    if progress < 0.25: return initial
                    elif progress < 0.5: return initial * 0.7
                    elif progress < 0.75: return initial * 0.4
                    else: return final

            # 计算每层的 token selections
            per_layer_selections = []
            layer_rates = []
            for layer_idx in range(num_layers):
                layer_rate = compute_layer_rate(
                    layer_idx, num_layers, initial_rate, layerwise_final_rate, layerwise_decay
                )
                layer_rates.append(layer_rate)

                # 使用 smart_query_selection 为该层选择 tokens
                selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=layer_rate,
                    system_len=selection_start,
                    device=input_device
                )
                per_layer_selections.append(selected_indices)

            # 计算统计信息
            avg_rate = np.mean(layer_rates)
            total_selected = sum(len(s) for s in per_layer_selections)
            total_possible = doc_len * num_layers

            print(f"\n  Per-layer rate statistics:")
            print(f"    Layer 0:  {layer_rates[0]:.1%} ({len(per_layer_selections[0])} tokens)")
            print(f"    Layer {num_layers//2}:  {layer_rates[num_layers//2]:.1%} ({len(per_layer_selections[num_layers//2])} tokens)")
            print(f"    Layer {num_layers-1}: {layer_rates[-1]:.1%} ({len(per_layer_selections[-1])} tokens)")
            print(f"    Average rate: {avg_rate:.1%}")
            print(f"    Total recomputation: {total_selected}/{total_possible} = {total_selected/total_possible*100:.1f}%")

            # 对于 DraftModelLayerwise，我们需要使用自定义的逐层 prefill
            # 但为了保持与现有框架兼容，我们用第 0 层的 selections 作为 k_need_index
            # 然后在 prefill 时使用自定义的逐层处理
            # 将 per_layer_selections 存储在 extra_info 中供后续使用
            k_need_index = torch.tensor(per_layer_selections[0], device='cpu')

            extra_info['layerwise_mode'] = True
            extra_info['per_layer_selections'] = per_layer_selections
            extra_info['layer_rates'] = layer_rates
            extra_info['dynamic_rate'] = avg_rate

            print(f"\nDraftModelLayerwise 完成逐层选择")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'Oracle':
            # Oracle: 用主模型本身 prefill 获取 attention，指导 token 选择
            # 与 DraftModel 方法相同，唯一区别是使用主模型而非小模型
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[1:-1])
            selection_start = system_len

            # 使用主模型计算 attention（复用 compute_draft_model_attention 函数）
            print(f"\n{'='*60}")
            print("Oracle: Using main model for attention computation")
            print(f"{'='*60}")
            full_input = torch.cat(passages).unsqueeze(0).to(input_device)
            oracle_attention = compute_draft_model_attention(model, full_input, query_start, input_device)
            torch.cuda.empty_cache()

            # oracle_attention 是 {layer_idx: attention [num_heads, query_len, seq_len]} 格式
            # 已经只包含 query positions 的 attention

            # 收集各层的 query→doc attention
            # 注意: 从 selection_start 开始，长度为 doc_len（只包括文本块2到文本块n）
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                # layer_attn: [num_heads, query_len, seq_len]
                # 提取 query→doc attention（只切片 key 维度，跳过 system 和 文本块1）
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                # 对 heads 和 query positions 平均
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层
            if draft_layer_selection == 'entropy':
                # 基于熵动态选层
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  Oracle 熵选层: 选择了 {active_layers}")
                # 聚合选中层的 attention
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]
            elif draft_layer_selection == 'last':
                # 只使用最后一层
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  Oracle 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}, expected 'entropy' or 'last'")

            # 使用 smart_query_selection 进行选择
            # 注意: selection_start 是选择区域的起始位置（跳过了 system 和 文本块1）
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=rate,
                system_len=selection_start,  # 使用 selection_start 作为偏移量
                device=input_device
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"Oracle 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'OracleAdaptive':
            # OracleAdaptive: Oracle 选择方式 + 动态比例计算
            # 与 Oracle 相同的选择逻辑 (smart_query_selection: 连通分量 + 边界扩展)
            # 但比例是动态计算的，而非固定值
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            # 使用主模型计算 attention
            print(f"\n{'='*60}")
            print("OracleAdaptive: Oracle selection + Dynamic rate")
            print(f"{'='*60}")
            full_input = torch.cat(passages).unsqueeze(0).to(input_device)
            oracle_attention = compute_draft_model_attention(model, full_input, query_start, input_device)
            torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}")

            # 动态计算比例 (使用综合多特征方法)
            budget_info = compute_dynamic_ratio_comprehensive(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                base_ratio=rate,  # 使用 rate 作为 base_ratio
                min_ratio=min_rate,
                max_ratio=max_rate
            )

            dynamic_rate = budget_info['dynamic_ratio']

            print(f"  Parameters: base_ratio={rate:.0%}")
            print(f"  Budget range: [{min_rate:.1%}, {max_rate:.1%}]")
            print(f"\n  Attention 分布特征分析:")
            print(f"    Coverage Analysis:")
            for cov, ratio_val in budget_info['coverage_analysis'].items():
                print(f"      {cov} coverage: {ratio_val:.2%}")
            print(f"    Concentration (Top-k):")
            print(f"      Top-1: {budget_info['concentration']['top1_ratio']:.2%}")
            print(f"      Top-3: {budget_info['concentration']['top3_ratio']:.2%}")
            print(f"      Top-5: {budget_info['concentration']['top5_ratio']:.2%}")
            print(f"      → Concentration Factor: {budget_info['concentration']['concentration_factor']:.2f}")
            print(f"    Connected Components: {budget_info['num_components']}")
            print(f"    Position Span: {budget_info['position_span']} tokens")
            print(f"    Spread Ratio: {budget_info['spread_ratio']:.4f}")
            print(f"    Gini Coefficient: {budget_info['gini_coefficient']:.4f}")
            print(f"\n  动态比例计算:")
            print(f"    Base coverage ratio (80%): {budget_info['base_coverage_ratio']:.2%}")
            print(f"    After concentration factor: {budget_info['base_coverage_ratio'] * budget_info['concentration']['concentration_factor']:.2%}")
            print(f"    Adjustments:")
            for adj_name, adj_value in budget_info['adjustments'].items():
                print(f"      {adj_name}: {adj_value:+.4f}")
            print(f"    Raw computed ratio: {budget_info['raw_ratio']:.2%}")
            print(f"    FINAL DYNAMIC RATIO: {dynamic_rate:.2%}")

            # 使用 smart_query_selection 进行选择 (与 Oracle 相同)
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=dynamic_rate,  # 使用动态计算的比例
                system_len=selection_start,
                device=input_device
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"\n  OracleAdaptive 选择结果:")
            print(f"    选中 {len(k_need_index)} tokens ({len(k_need_index)/doc_len*100:.1f}%)")
            print(f"    (smart_query_selection: 连通分量 + 边界扩展)")
            print(f"  select_time: {time.time() - select_time:.3f}s")

            # 保存动态信息用于后续统计
            extra_info['dynamic_rate'] = dynamic_rate
            extra_info['base_coverage_ratio'] = budget_info['base_coverage_ratio']
            extra_info['num_components'] = budget_info['num_components']
            extra_info['gini_coefficient'] = budget_info['gini_coefficient']
            extra_info['spread_ratio'] = budget_info['spread_ratio']
            extra_info['concentration_factor'] = budget_info['concentration']['concentration_factor']
            extra_info['top1_ratio'] = budget_info['concentration']['top1_ratio']
            extra_info['total_budget'] = len(k_need_index)
            extra_info['doc_len'] = doc_len

        elif reprocess_method == 'vAttention':
            # vAttention: 结合 top-k 选择和随机采样
            # 参考论文 "vAttention: Verified Sparse Attention" (arXiv:2510.05688)
            # 核心思想：一部分 budget 用于 top-k，一部分用于随机采样
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            # 计算总的需要选择的 token 数量
            total_budget = int(rate * doc_len)

            # vAttention 参数：topk_ratio 控制 top-k 和随机采样的比例
            # 默认各占 50%（论文中推荐的配置）
            topk_ratio = vattention_topk_ratio

            topk_budget = int(total_budget * topk_ratio)
            random_budget = total_budget - topk_budget

            print(f"\n{'='*60}")
            print("vAttention: Combining Top-k and Random Sampling")
            print(f"{'='*60}")
            print(f"  Total budget: {total_budget} tokens (rate={rate:.2%})")
            print(f"  Top-k budget: {topk_budget} tokens ({topk_ratio:.0%})")
            print(f"  Random budget: {random_budget} tokens ({1-topk_ratio:.0%})")

            # 使用主模型计算 attention（复用 Oracle 的计算方式）
            full_input = torch.cat(passages).unsqueeze(0).to(input_device)
            oracle_attention = compute_draft_model_attention(model, full_input, query_start, input_device)
            torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            # 注意: 从 selection_start 开始，长度为 doc_len（只包括文本块2到文本块n）
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层（使用熵选层）
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  vAttention 熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  vAttention 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}")

            # Step 1: Top-k 选择
            # 获取 attention 分数最高的 topk_budget 个 token
            if topk_budget > 0:
                topk_values, topk_indices = torch.topk(multi_layer_attn, min(topk_budget, doc_len))
                topk_selected = set(topk_indices.cpu().tolist())
            else:
                topk_selected = set()

            # Step 2: 随机采样
            # 从剩余的 token 中随机采样 random_budget 个
            all_indices = set(range(doc_len))
            remaining_indices = list(all_indices - topk_selected)

            if random_budget > 0 and len(remaining_indices) > 0:
                # 设置随机种子以确保可复现性
                torch.manual_seed(42)
                random_sample_size = min(random_budget, len(remaining_indices))
                random_indices = torch.randperm(len(remaining_indices))[:random_sample_size]
                random_selected = set([remaining_indices[i] for i in random_indices.tolist()])
            else:
                random_selected = set()

            # Step 3: 合并两者
            combined_selected = topk_selected | random_selected

            # 确保不超过总 budget
            if len(combined_selected) > total_budget:
                # 如果超过了，按 attention 分数排序，保留分数最高的
                combined_list = list(combined_selected)
                combined_scores = [(idx, multi_layer_attn[idx].item()) for idx in combined_list]
                combined_scores.sort(key=lambda x: x[1], reverse=True)
                combined_selected = set([idx for idx, _ in combined_scores[:total_budget]])

            # 转换为全局索引（使用 selection_start 作为偏移量，跳过 system 和 文本块1）
            selected_indices = [idx + selection_start for idx in sorted(combined_selected)]
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"\n  vAttention 选择结果:")
            print(f"    Top-k 选中: {len(topk_selected)} tokens")
            print(f"    随机采样选中: {len(random_selected)} tokens")
            print(f"    总计选中: {len(combined_selected)} tokens ({len(combined_selected)/doc_len*100:.1f}%)")
            print(f'  select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'OracleDynamic':
            # OracleDynamic: 动态计算重算比例
            # 参考论文 "vAttention: Verified Sparse Attention" (arXiv:2510.05688)
            # 核心思想：根据 attention 分布的方差动态决定需要重算多少 token
            # 用户指定 epsilon (误差容忍度) 和 delta (置信度)，系统自动计算 budget
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            print(f"\n{'='*60}")
            print("OracleDynamic: Adaptive Budget Computation")
            print(f"{'='*60}")
            print(f"  Parameters: epsilon={epsilon}, delta={delta}")
            print(f"  Budget range: [{min_rate:.1%}, {max_rate:.1%}]")

            # Step 1: 使用主模型计算 attention
            full_input = torch.cat(passages).unsqueeze(0).to(input_device)
            oracle_attention = compute_draft_model_attention(model, full_input, query_start, input_device)
            torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            # 注意: 从 selection_start 开始，长度为 doc_len（只包括文本块2到文本块n）
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}")

            # Step 2: 动态计算 budget
            budget_info = compute_dynamic_budget(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                epsilon=epsilon,
                delta=delta,
                base_sample_ratio=0.05,
                topk_ratio=vattention_topk_ratio,
                min_rate=min_rate,
                max_rate=max_rate
            )

            total_budget = budget_info['total_budget']
            topk_budget = budget_info['topk_budget']
            random_budget = budget_info['random_budget']
            dynamic_rate = budget_info['dynamic_rate']

            print(f"\n  Attention 分布分析:")
            print(f"    覆盖阈值: {budget_info['coverage_threshold']:.1%}")
            print(f"    达到覆盖需要的 top-k 数量: {budget_info['topk_count_for_coverage']} tokens ({budget_info['topk_count_for_coverage']/doc_len*100:.2f}%)")
            print(f"    Top-k 实际覆盖权重: {budget_info['topk_coverage']:.2%}")
            print(f"    残差权重: {budget_info['residual_weight']:.2%}")
            print(f"    残差变异系数 (CV): {budget_info['residual_cv']:.4f}")
            print(f"    归一化熵: {budget_info['normalized_entropy']:.4f} (0=极度集中, 1=均匀分布)")
            print(f"\n  动态 Budget 计算结果:")
            print(f"    原始覆盖率: {budget_info['coverage_based_rate']:.2%}")
            print(f"    限制后比例: {dynamic_rate:.2%} (min={min_rate:.0%}, max={max_rate:.0%})")
            print(f"    总 budget: {total_budget} tokens")

            # Step 3: 纯 Top-k 选择 (和 Oracle 一样，只选 attention 最高的)
            # 注意：不再使用随机采样，因为随机采样的低 attention token 会影响生成质量
            if total_budget > 0:
                topk_values, topk_indices = torch.topk(multi_layer_attn, min(total_budget, doc_len))
                combined_selected = set(topk_indices.cpu().tolist())
            else:
                combined_selected = set()

            # 记录选择信息（保持兼容性）
            topk_selected = combined_selected
            random_selected = set()  # 不再使用随机采样

            # 转换为全局索引（使用 selection_start 作为偏移量，跳过 system 和 文本块1）
            selected_indices = [idx + selection_start for idx in sorted(combined_selected)]
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"\n  OracleDynamic 选择结果:")
            print(f"    选中 Top-{len(combined_selected)} tokens ({len(combined_selected)/doc_len*100:.1f}%)")
            print(f"    (纯 top-k 选择，和 Oracle 相同策略，只是动态计算比例)")
            print(f"  select_time: {time.time() - select_time:.3f}s")

            # 保存动态信息用于后续统计
            extra_info['dynamic_rate'] = dynamic_rate
            extra_info['topk_coverage'] = budget_info['topk_coverage']
            extra_info['topk_count_for_coverage'] = budget_info['topk_count_for_coverage']
            extra_info['normalized_entropy'] = budget_info['normalized_entropy']
            extra_info['total_budget'] = len(combined_selected)
            extra_info['doc_len'] = doc_len

        else:
            raise NotImplementedError

        # reprocess kv cache and prefill question
        # 对于 DraftModelLayerwise，使用所有层选择的并集
        if extra_info.get('layerwise_mode') and extra_info.get('per_layer_selections'):
            per_layer_selections = extra_info['per_layer_selections']
            # 使用所有层选择的并集 (即第 0 层的选择，因为它包含最多 tokens)
            all_positions = set()
            for layer_sel in per_layer_selections:
                all_positions.update(layer_sel)
            k_need_index = sorted(list(all_positions))
            print(f"\n  Layerwise union positions: {len(all_positions)} unique doc positions")
            print(f"  (Note: Using union for forward pass; per-layer rates documented in extra_info)")
        else:
            k_need_index = torch.sort(k_need_index)[0].tolist()
        k_need_index.extend(range(sum(passages_len[:-1]),sum(passages_len)))
    else:
        k_need_index = range(sum(passages_len[:-1]),sum(passages_len))
    past_len = sum(passages_len)

    batch_size, seq_length = 1, len(k_need_index)

    generated_ids = torch.zeros(
        batch_size, past_len + max_new_tokens + 1, dtype=torch.int, device=input_device
    )
    generated_ids[:, :past_len] = torch.cat(passages).unsqueeze(0).to(input_device)
    tokens = []

    if reprocess_method != 'FusionRAG':
        use_sparse_attention = False
    else:
        use_sparse_attention = False
    reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(input_device)
    cache_position = torch.tensor(k_need_index, device=input_device)

    with torch.no_grad():
        without_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(input_device)

        # Don't force move to input_device - keep on the device where model output is
        # This avoids cross-GPU transfer deadlock in PP mode
        model_output = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position,
            past_key_values=past_key_values, return_dict=False, use_cache=True, use_sparse_attention=use_sparse_attention,
        )[0]

        logits = model_output[:,-1,:].unsqueeze(0).clone()
        with_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
        v_sub_all = without_attn_value - with_attn_value
        v_sub_all = v_sub_all.squeeze(0)
        v_sub_all = v_sub_all.transpose(0, 1)
        v_sum = torch.sum(v_sub_all**2, dim=[1,2])   
        
        first_token_time = time.time() - start_time
        stream = TextStreamer(tokenizer)
        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)
        next_token_scores = logits_warper(reprocess_inputs, logits[:, -1, :])
        next_token = torch.argmax(next_token_scores, dim=-1)
        prefill_count = seq_length
        prefill_time = first_token_time
        # print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, past_len+1] = next_token
        tokens.append(next_token)

        # Use the device where logits/next_token are (model output device in PP mode)
        output_device = next_token.device
        inputs = torch.cat((torch.cat(passages).unsqueeze(0).to(output_device), next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([past_len], device=output_device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
 
        
        decode_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, inputs)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            seq_length += 1
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>':
                break

            cache_position += 1
            position_ids = cache_position.unsqueeze(0)
        

    total_time = time.time() - decode_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    print("")

    # print(f"prompt eval count:    {prefill_count} token(s)")
    # print(f"prompt eval duration: {prefill_time}s")
    # print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    # print(f"eval count:           {tokens_generated} token(s)")
    # print(f"eval duration:        {total_time}s")
    # print(f"eval rate:            {tokens_per_second} tokens/s")

    # 记录 token 统计信息
    extra_info['recompute_token_count'] = prefill_count
    extra_info['decode_token_count'] = len(tokens)

    return tokens, prefill_time, extra_info

def tf_logits_warper(temperature, top_k):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
        min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        warpers.append(TemperatureLogitsWarper(temperature))
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep))
        return warpers

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = False,
                         mode = 'normal', early_exit_layer=None, device='cuda', device_map=None):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape

    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = "cuda:0" if device_map is not None else device
    inputs = inputs.to(input_device)

    tokens = []
    
    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, use_cuda_graph: bool = False):
        if use_cuda_graph:
            logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        else:
            # custom_stream = torch.cuda.Stream()
            inputs_embeds = model.model.embed_tokens(cur_token).to(input_device)
            # with torch.cuda.stream(custom_stream):
            logits=model(inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=False, use_cache=True)[0]
        if past_key_values != None:
            past_key_values.change_seq_length(1)
        #print(logits)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])

        next_token = torch.argmax(next_token_scores, dim=-1)
        return next_token
    
    with torch.no_grad():
        stream = TextStreamer(tokenizer)

        # Pass device_map if multi-GPU, otherwise pass device
        cache_device = device_map if device_map is not None else input_device
        past_key_values = StaticCache(
            config = model.config, max_batch_size=1, max_cache_len=seq_length+max_new_tokens, device=cache_device, dtype=model.dtype
        )

        cache_position = torch.arange(seq_length, device=input_device)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=input_device
        )
        generated_ids[:, cache_position] = inputs.to(input_device).to(torch.int)
        if past_key_values != None:
            past_key_values.cur_idx=cache_position
        start_time = time.time()

        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            inputs_embeds = model.model.embed_tokens(inputs).to(input_device)

        # Pass early_exit_layer if specified (for CacheBlend)
        model_kwargs = {
            'inputs_embeds': inputs_embeds,
            'cache_position': cache_position,
            'past_key_values': past_key_values,
            'return_dict': False,
            'use_cache': True
        }
        if early_exit_layer is not None:
            model_kwargs['early_exit_layer'] = early_exit_layer

        logits = model(**model_kwargs)[0][:,-1,:].unsqueeze(0).clone().to(input_device)
        # generation_config, model_kwargs = model._prepare_generation_config(None, do_sample=False, top_k=1,  temperature=0.01)

        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])

        next_token = torch.argmax(next_token_scores, dim=-1)
        first_token_time = time.time() - start_time

        prefill_count = seq_length
        prefill_time = first_token_time
        # print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(next_token)
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=input_device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
        if use_cuda_graph:
            cuda_graph_runner = CUDAGraphRunner()
            cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, device, return_dict=False, use_cache=True)
        else:
            cuda_graph_runner = None
            
        start_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, use_cuda_graph).to(input_device)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            seq_length += 1
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>' or tokenizer.decode(next_token) == '[unused10]':
                # print(stream.end(), end="", flush=True)
                break

            cache_position += 1
            position_ids = cache_position.unsqueeze(0)
        

    total_time = time.time() - start_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time



    return tokens, past_key_values, prefill_time



# ============================================================
# Utility functions for process_cache
# ============================================================

def remove_unused_tokens(text):
    """移除所有 [unusedXX] 格式的 token (PanGu specific)"""
    cleaned = re.sub(r'\[unused\d+\]', '', text)
    return cleaned.strip()

def parse_generation(s):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def find_group_and_index(sizes, idx):
    """
    找到list中的某个索引属于哪个组及该组中的索引
    :param sizes: 每个组的大小的列表
    :param idx: 要查找的索引
    :return: (组号, 组中的索引)
    """
    cumulative_size = 0
    for group_id, group_size in enumerate(sizes):
        if cumulative_size + group_size > idx:
            group_index = idx - cumulative_size
            return group_id, group_index
        cumulative_size += group_size
    return None, None  # 如果索引超出范围，返回None

def split_passages_by_title(text, title_marker):
    # 使用标题标记作为分割点，找到所有的位置
    titles = [i for i in range(len(text)) if text.startswith(title_marker, i)]
    # 根据标题位置分割文本为多个段落
    passages = [text[titles[i]:titles[i+1]].strip() for i in range(len(titles) - 1)]
    passages.append(text[titles[-1]:].strip())  # 添加最后一个段落
    return passages

def normalize_answer(s, model_type='default'):
    """
    Normalize answer text
    :param s: answer string
    :param model_type: 'llama' uses slightly different normalization
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        if model_type == 'llama':
            return ' '.join(text.replace('\n', ' ').split())
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _rouge1_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(normalize_answer(prediction), normalize_answer(ground_truth), avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def save_list_to_jsonl(data_list, file_path):
    """
    保存一个字典的列表为 jsonl 文件。
    :param data_list: 要保存的字典列表
    :param file_path: 保存的文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')

def prepare_data(model_name, data_path, data_name, cache_path, tokenizer: AutoTokenizer,
                 topk: int, revert_rope, preprocess, bge_model_path='/mnt/data/models/bge-m3-FP16'):
    """
    Prepare data for process cache experiments
    """
    import faiss
    from FlagEmbedding import FlagModel
    import random
    import os

    prompt_config = json.load(open('./config/dataset2prompt_few-shot.json'))
    data_path = data_path+data_name
    if data_name in ['2wikimqa.jsonl', 'samsum.jsonl', 'multi_news.jsonl', 'musique.jsonl', 'hotpotqa.jsonl', 'triviaqa.jsonl']:
        data_name_prefix = data_name.split('.')[0]
    else:
        data_name_prefix = data_name.split('-')[0]
    if data_name_prefix in ['hotpotqa','triviaqa','2wikimqa','musique']:
        rouge_metrics = _rouge1_score
        max_tokens_length = 50
    elif data_name_prefix in ['samsum','multi_news']:
        rouge_metrics = _rougel_score
        max_tokens_length = 512
    system_prompt = prompt_config['system_prompt'][model_name.split('-')[0]][data_name_prefix]
    system_tokens = torch.tensor(tokenizer.encode(system_prompt, add_special_tokens = False),dtype=torch.int)
    query_task = prompt_config['query_prompt'][model_name.split('-')[0]][data_name_prefix]
    local_model_config = json.load(open('./config/model_config.json'))
    stop_token_id = local_model_config[model_name.split('-')[0]]['stop_token_id']
    # 存报告
    if not os.path.exists(f"{cache_path}{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}{data_name.split('.')[0]}/{model_name}")
    # 存数据
    if not os.path.exists(f"{cache_path}data"):
        os.makedirs(f"{cache_path}data")
    # reprocess 数据
    if not os.path.exists(f"{cache_path}data/{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}data/{data_name.split('.')[0]}/{model_name}")
    if not os.path.exists(f"{cache_path}{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}{data_name.split('.')[0]}/{model_name}")
    # preprocesss 数据
    if not os.path.exists(f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}"):
        os.makedirs(f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}")

    csv_path = f"{cache_path}{data_name.split('.')[0]}/{model_name}"
    reprocess_path = f"{cache_path}data/{data_name.split('.')[0]}/{model_name}"
    preprocess_path = f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}"
    data_file = open(data_path, 'r', encoding='utf-8')
    data = []
    for line in data_file.readlines():
        data.append(json.loads(line))
    if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
        data = data[0]
        for i in range(len(data)):
            # 打乱顺序
            random.seed(1)
            random.shuffle(data[i]['output'][0]['document'])
            data[i]['passage'] = data[i]['output'][0]['document']
    else:
        if data_name == 'samsum.jsonl':
            split_mark = 'Dialogue:'
        else:
            split_mark = 'Passage'
        for i in range(len(data)):
            data[i]['passage'] = re.findall(f'({split_mark} \\d+.*?)(?={split_mark} \\d+|$)', data[i]['context'], re.DOTALL)
        if data_name == 'musique-140.jsonl':
            for i in range(len(data)):
                data[i]['passage'] = re.findall(f'Passage \\d+:\\n(.*?)(?=Passage \\d+:|$)', data[i]['context'], re.DOTALL)
                data[i]['passage'] = ['\n\n' + text for text in data[i]['passage']]
                data[i]['passage'][-1] = data[i]['passage'][-1] + '\n'

    N = len(data)
    batch_data = []
    batch_tokens = []
    question_list = []
    real_answer_list = []

    for query_id,query in enumerate(data[:N]):
        query_prompt = query_task.format(input=data[query_id]['input'])
        query_tokens = torch.tensor(tokenizer.encode(query_prompt, add_special_tokens = False),dtype=torch.int)
        question_list.append(data[query_id]['input'])
        tmp_list = []
        if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
            for i in range(len(data[query_id]['output'])):
                if 'answer' in data[query_id]['output'][i] and \
                    data[query_id]['output'][i]['answer'] not in tmp_list:
                    tmp_list.append(data[query_id]['output'][i]['answer'])
        else:
            for i in range(len(data[query_id]['answers'])):
                if data[query_id]['answers'][i] not in tmp_list:
                    tmp_list.append(data[query_id]['answers'][i])
        real_answer_list.append(tmp_list)
        index = 0

        passage = [system_prompt]
        passage_tokens = [system_tokens]
        for bn in range(len(query['passage'])):
            if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
                passage.append(f'Passage {index+1}:\n' + query['passage'][index] + '\n')
                passage_tokens.append(torch.tensor(tokenizer.encode(f'Passage {index+1}:\n' + query['passage'][index] + '\n', add_special_tokens = False),dtype=torch.int))
            else:
                passage.append(query['passage'][index] + '\n')
                passage_tokens.append(torch.tensor(tokenizer.encode(query['passage'][index] + '\n', add_special_tokens = False),dtype=torch.int))
            index += 1
            if index >=len(query['passage']):
                break
        passage.append(query_prompt)
        passage_tokens.append(query_tokens)
        batch_tokens.append(passage_tokens)
        batch_data.append(passage)

    if preprocess == True:
        bgem3 = FlagModel(bge_model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=True)
        corpus = []
        corpus_lens = []
        for batch in batch_data:
            corpus.extend(batch[1:-1])
            corpus_lens.append(len(batch[1:-1]))
        path = f"{cache_path}data/{data_name.split('.')[0]}.bin"
        start_time = time.time()
        if os.path.exists(path):
            index = faiss.read_index(path)
        else:
            corpus_embeddings = bgem3.encode(corpus)
            print("shape of the corpus embeddings:", corpus_embeddings.shape)
            print("data type of the embeddings: ", corpus_embeddings.dtype)
            dim = corpus_embeddings.shape[-1]
            index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
            corpus_embeddings = corpus_embeddings.astype(np.float32)
            index.train(corpus_embeddings)
            index.add(corpus_embeddings)
            print(f"total number of vectors: {index.ntotal}")

            faiss.write_index(index, path)
        corpus = np.asarray(corpus)
        corpus_embeddings = bgem3.encode_queries(corpus)
        corpus_embeddings = corpus_embeddings[:].astype(np.float32)
        score, idx = index.search(corpus_embeddings, k=topk)
        context_rank = idx
        bgem3 = None
        duration_time = time.time() - start_time
        print(f"embedding time: {duration_time}")
    context_rank = context_rank if preprocess == True else []
    corpus_lens = corpus_lens if preprocess == True else []
    return batch_data, batch_tokens,question_list, real_answer_list, stop_token_id, \
        reprocess_path, preprocess_path, csv_path,\
            data_name_prefix, rouge_metrics, context_rank, corpus_lens
