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
# RoPE 辅助函数
# ============================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(x, cos, sin):
    """Apply Rotary Position Embedding to a single tensor (key or value).

    Args:
        x: Input tensor (key or value)
        cos: Cosine part of rotary embedding
        sin: Sine part of rotary embedding

    Returns:
        Tensor with rotary position embedding applied
    """
    cos = cos.unsqueeze(1)  # Add head dimension
    sin = sin.unsqueeze(1)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


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

    # DEBUG: 打印 threshold 计算信息
    print(f"\n[DEBUG] smart_query_selection:")
    print(f"  doc_len: {doc_len}")
    print(f"  target_ratio: {target_ratio}")
    print(f"  target_count: {target_count}")
    print(f"  threshold_factor: {threshold_factor}")

    # Step 1: 找到高 attention 位置 (使用可配置的 threshold_factor)
    mean_attn = np.mean(attention_scores[attention_scores > -1e8])
    std_attn = np.std(attention_scores)
    threshold = mean_attn + threshold_factor * std_attn

    print(f"  attention_scores - mean: {mean_attn:.6f}, std: {std_attn:.6f}")
    print(f"  threshold: {threshold:.6f}")

    high_attn_positions = list(np.where(attention_scores > threshold)[0])

    print(f"  high_attn_positions: {len(high_attn_positions)} positions ({len(high_attn_positions)/doc_len*100:.2f}%)")

    # Step 2: 连通分量分析
    components = find_connected_components(high_attn_positions, max_gap=2)

    print(f"  connected components: {len(components)}")

    # Step 3: 计算每个分量的总 attention
    component_scores = []
    for comp in components:
        total_score = sum(attention_scores[p] for p in comp)
        component_scores.append((comp, total_score))

    # Step 4: 按总 attention 排序
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 5: 贪心选择分量 + 上下文扩展 (±1)
    selected = set()
    component_idx = 0

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
            component_idx += 1
        else:
            # 超过余量，停止添加
            break

    print(f"  After component selection: {len(selected)} positions (from {component_idx} components)")

    before_fill = len(selected)
    if len(selected) < target_count:
        sorted_indices = np.argsort(attention_scores)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(int(pos))
                if len(selected) >= target_count:
                    break

    if len(selected) > before_fill:
        print(f"  After补充: {len(selected)} positions (added {len(selected) - before_fill})")

    # Step 7: 如果超过目标，移除最低分的位置
    before_prune = len(selected)
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: attention_scores[p])
        selected.remove(min_pos)

    if len(selected) != before_prune:
        print(f"  After修剪: {len(selected)} positions (removed {before_prune - len(selected)})")

    print(f"  Final selected: {len(selected)} positions (target: {target_count})")

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
                          save_path='', doc_id=None, system_len = 0, passage_len = 0, reprocess_method=None, device="cuda", device_map=None
                          ):

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape

    # Check doc_id parameter
    if doc_id is None:
        raise ValueError("doc_id parameter is required")

    # Define global system prompt ID
    SYSTEM_PROMPT_ID = -1

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
        if reprocess_method == "Cache-Craft" and doc_id != SYSTEM_PROMPT_ID:
            passages_len = [system_len, passage_len]
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
            cachecraft_score = past_key_values.importance_cache[-1] # [num_head, passage_len]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            torch.save(cachecraft_score, f'{save_path}/cachecraftattn_doc_{doc_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
        past_len = past_key_values.past_tokens[0]
        key_cache = []
        value_cache = []
        if doc_id == SYSTEM_PROMPT_ID:
            # System prompt: save entire cache
            # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
            key_cache = [past_key_values.key_cache[i][:,:,:past_len,:].cpu() for i in range(len(past_key_values.key_cache))]
            key_cache = torch.stack(key_cache)
            value_cache = [past_key_values.value_cache[i][:,:,:past_len,:].cpu() for i in range(len(past_key_values.value_cache))]
            value_cache = torch.stack(value_cache)
        else:
            # Document: save only document part (skip system tokens)
            # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
            key_cache = [past_key_values.key_cache[i][:,:,system_len:system_len + passage_len,:].cpu() for i in range(len(past_key_values.key_cache))]
            key_cache = torch.stack(key_cache)
            value_cache = [past_key_values.value_cache[i][:,:,system_len:system_len + passage_len,:].cpu() for i in range(len(past_key_values.value_cache))]
            value_cache = torch.stack(value_cache)
        # Use file lock to prevent concurrent writes from multiple processes
        key_path = f'{save_path}/doc_{doc_id}_key.pt'
        value_path = f'{save_path}/doc_{doc_id}_value.pt'
        lock_path = f'{save_path}/doc_{doc_id}.lock'

        with FileLock(lock_path, timeout=60):
            # Double-check if file exists (another process might have created it)
            if not os.path.exists(key_path):
                torch.save(key_cache.clone(), key_path)
                torch.save(value_cache.clone(), value_path)
                print(f'doc_id: {doc_id} (saved by current process)')
            else:
                print(f'doc_id: {doc_id} (already exists, skipped)')

        return key_cache, value_cache

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
                          save_path='', doc_id=None, system_len=0, revert_rope=False, reprocess_method=None, device="cuda", device_map=None):

    # Check doc_id parameter
    if doc_id is None:
        raise ValueError("doc_id parameter is required")

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
            torch.save(cachecraft_score, f'{save_path}/cachecraftattn_doc_{doc_id}.pt')
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
    torch.save(key_cache.clone(), f'{save_path}/doc_{doc_id}_key.pt')
    key_cache = None
    if "cuda" in input_device:
        torch.cuda.empty_cache()
    elif "npu" in input_device:
        torch.npu.empty_cache()
    # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
    value_cache = torch.stack([cache.cpu() for cache in past_key_values.value_cache])[:,:,:,past_len:past_len + passage_len,:]
    torch.save(value_cache.clone(), f'{save_path}/doc_{doc_id}_value.pt')


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
                          load_path='', doc_ids=None, max_new_tokens=1, revert_rope=False,
                          reprocess_method='normal', rate=0, preprocess=False, draft_model=None,
                          draft_attention=None, use_entropy_selection=False, entropy_top_k=4,
                          draft_layer_selection='entropy',  # 'entropy', 'last', 'fixed', or 'middle'
                          draft_fixed_layer=3,  # 固定使用哪一层 (当 draft_layer_selection='fixed' 时生效)
                          draft_threshold_factor=0.5,  # smart_query_selection 阈值因子 (default 0.5)
                          use_similarity_rerank=False,  # 使用 query-doc 相似度重排序改进选择
                          rerank_multiplier=2.0,  # 重排序时先选择多少倍候选
                          group=False, device="cuda", device_map=None,
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
                          original_kv_path=None):  # 原始KV cache路径，用于文本块1 (doc_id=first document)
    import os
    import pdb

    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = "cuda:0" if device_map is not None else device

    passages_len = [passage.shape[0] for passage in passages] # 列表，记录了每个召回文本块passage的 Token 数量，passage -1代表问题，问题部分也会有其他东西包装
    passages_start = [sum(passages_len[:i]) for i in range(1,len(passages_len))] # 列表，记录了每个文本块在整体输入中的起始位置
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0])) # 获得问题之前部分的长度 '<|im_end|>\n<|im_start|>user\nQuestion: Who is the director of the film Borunbabur Bondhu?<|im_end|>\n<|im_start|>assistant\nAnswer: '
    # inputs: Tensor[1, question_len] tokenizer.decode(passages[0].squeeze().tolist())
    inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(input_device) # 截取真正的用户问题,但是这里好像没考虑到后面部分的过滤？  '<|im_end|>\n<|im_start|>user\n
    # 'Question: Who is the director of the film Borunbabur Bondhu?<|im_end|>\n<|im_start|>assistant\nAnswer: '

    # seq_length: int, 问题的实际长度 (不包括前缀)
    seq_length = passages[-1][query_prefix_len:].shape[0]

    # extra_info: dict, 存储额外统计信息，用于分析和调试
    # 各字段说明:
    #   - dynamic_rate: float, OracleDynamic 方法动态计算的 rate 值
    #   - topk_coverage: float, top-k 选择的覆盖率 (0.0-1.0)
    #   - topk_count_for_coverage: int, 达到目标覆盖率需要的 token 数
    #   - normalized_entropy: float, 归一化的 attention 熵 (越大越分散)
    #   - total_budget: int, 总的 token budget
    #   - doc_len: int, 所有文档的总 token 数
    extra_info = {
        'dynamic_rate': None,
        'topk_coverage': None,
        'topk_count_for_coverage': None,
        'normalized_entropy': None,
        'total_budget': None,
        'doc_len': None
    }

    # past_key_values(StaticCache封装) 含有key_cache和value_cache两类属性,
    # 1)
    # past_key_values.key_cache = [
    #     (key_0),  # for layer 0 [batch_size, num_heads, seq_len, head_dim]
    #     (key_1),  # for layer 1 [batch_size, num_heads, seq_len, head_dim]
    #     ...                # for other layers
    # ]
    # 2) 具有写入指针.seen_tokens/也会记作.past_tokens 告诉模型当前显存里已经存了多少个 Token 

    # ========== 2. 初始化 KV Cache 加载状态 ==========
    # 重置所有层的 past_tokens 写入指针，更新每层已使用的 cache 长度)
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    past_len = 0  # 当前已加载到 past_key_values 中的 token 总数。

    # system prompt 的长度 (passages[0])，792，里面还带few examples了有点长
    system_len = passages[0].shape[0]
    # system prompt 的特殊 doc_id ，值为 -1,
    SYSTEM_PROMPT_ID = -1

    if doc_ids is None: # [-1, 2266, 183, 2078, 2467, 1152, 1141, 1910, 1444, 1490] 全局文档池子表示
        raise ValueError("doc_ids parameter is required")

    # key_cache/value_cache: List[List[Tensor] or None], 存储每个文档的 key cache
    # 结构: [[layer0_key, layer1_key, ...], [layer0_key, ...], None, ...]
    # 每层 key 维度: [1, num_heads, doc_seq_len, head_dim]
    # None 表示该文档的 KV cache 缺失，需要在线生成
    key_cache = []
    value_cache = []

    # all_position_ids: [1, seq_len], 记录各段的位置 ID (用于 RoPE)
    all_position_ids = [torch.arange(0,system_len).unsqueeze(0).to(input_device)]

    # ========== 3. online lazy: 检查哪些文档缺少 KV cache ==========
    # missing_chunks: List[Tuple(idx, doc_id, passage)], 记录缺失 KV cache 的文档
    # 结构: [(文档在passages中的索引, 文档全局ID, 文档token序列), ...] 这些文档将在后续 forward pass 时实时生成 KV cache 并保存到磁盘
    missing_chunks = []  # Track documents without KV cache: [(idx, doc_id, passage)]
    #pdb.set_trace()
    for idx, passage in enumerate(passages[:-1]):
        doc_id = doc_ids[idx]
        passage_len = passage.shape[0]

        kv_path = load_path

        # New path format using global doc_id
        key_path = f'{kv_path}/doc_{doc_id}_key.pt'
        value_path = f'{kv_path}/doc_{doc_id}_value.pt'

        if os.path.exists(key_path) and os.path.exists(value_path):
            # KV cache exists - try to load it
            # Note: Saved as list of layers, each layer is [1, num_heads, seq_len, head_dim]
            try:
                chunk_key_cache = torch.load(key_path, weights_only=True)
                chunk_value_cache = torch.load(value_path, weights_only=True)

                # Move to CPU if not already
                if isinstance(chunk_key_cache, list):
                    chunk_key_cache = [k.to('cpu') if k.device.type != 'cpu' else k for k in chunk_key_cache]
                    chunk_value_cache = [v.to('cpu') if v.device.type != 'cpu' else v for v in chunk_value_cache]
                else:
                    # Old format (single tensor) - shouldn't happen after clearing cache
                    chunk_key_cache = chunk_key_cache.to('cpu')
                    chunk_value_cache = chunk_value_cache.to('cpu')

                key_cache.append(chunk_key_cache)
                value_cache.append(chunk_value_cache)
            except Exception as e:
                # Corrupted cache file - delete and treat as missing
                print(f"  ⚠ Doc {doc_id}: KV cache corrupted ({e}), deleting and will regenerate")
                try:
                    if os.path.exists(key_path):
                        os.remove(key_path)
                    if os.path.exists(value_path):
                        os.remove(value_path)
                except:
                    pass
                key_cache.append(None) # 占位符，保持索引一致
                value_cache.append(None) # 占位符，保持索引一致
                missing_chunks.append((idx, doc_id, passage)) # 记录缺失的文档信息
        else:
            # KV cache missing - will generate during forward pass
            print(f"  ⚠ Doc {doc_id}: KV cache not found, will generate during answer generation")
            key_cache.append(None)  # Placeholder
            value_cache.append(None)
            missing_chunks.append((idx, doc_id, passage))

    #pdb.set_trace()
    # ========== 4. 将加载的 KV Cache 拷贝到 GPU 显存 ==========
    start_time = time.time()
    # 遍历所有文档 (不包括最后的 question)
    for idx, passage in enumerate(passages[:-1]):
        doc_id = doc_ids[idx]
        passage_len = passage.shape[0]  # 当前文档的 token 数

        # 跳过缺失的文档 - 在后续 forward pass 时生成
        if key_cache[idx] is None:  # 让加载成功的文档在显存里是紧凑排列的，miss chunk后面再加入
            continue
        # chunk_key_cache: List[Tensor], 当前文档的所有层 key cache
        # chunk_value_cache: List[Tensor], 当前文档的所有层 value cache
        # 每层维度: [1, num_heads, passage_len, head_dim]
        chunk_key_cache = key_cache[idx]
        chunk_value_cache = value_cache[idx]

        # 验证加载的 KV cache 格式是否正确
        if isinstance(chunk_key_cache, list):
            assert passage_len == chunk_key_cache[0].shape[2], f"passage_len={passage_len}, but KV shape={chunk_key_cache[0].shape}"

            # ========== 4.1 RoPE 位置调整  ==========
            # RoPE 调整目的: 将绝对位置编码转换为当前序列中的相对位置
            # 例如: 缓存时文档在位置 [0, doc_len), 现在要放到位置 [past_len, past_len+doc_len)
            if revert_rope and doc_id != SYSTEM_PROMPT_ID:
                for layer_idx in range(len(chunk_key_cache)):
                    # 获取当前层的 rotary embedding (RoPE) 模块
                    rotary_emb = model.model.layers[layer_idx].self_attn.rotary_emb

                    # 确定 RoPE 计算所在的设备
                    if hasattr(rotary_emb, 'inv_freq') and rotary_emb.inv_freq is not None:
                        rotary_device = rotary_emb.inv_freq.device
                    else:
                        rotary_device = next(model.model.layers[layer_idx].parameters()).device

                    # 将 KV 移动到 RoPE 计算设备
                    layer_chunk_key = chunk_key_cache[layer_idx].to(rotary_device)
                    layer_chunk_value = chunk_value_cache[layer_idx].to(rotary_device)

                    # ========== RoPE 平移调整 ==========
                    # 原理: RoPE 旋转矩阵满足复合性质 R_m * R_n = R_{m+n}
                    #
                    # 保存的 KV cache 使用相对位置: k_i 带有编码 R_i (i=0,1,2,...)
                    # 要移动到绝对位置 past_len + i，根据复合性质:
                    #   R_{past_len} * R_i = R_{past_len+i}
                    # 因此只需对所有 key 应用统一的旋转 R_{past_len - system_len}
                    # 这是一个整体平移操作，不需要先移除再重新编码

                    # position_ids: 所有位置都是 past_len - system_len (相对于 system prompt 后的偏移)
                    # 这会生成旋转矩阵 R_{past_len - system_len}
                    position_ids = torch.full(
                        (1, layer_chunk_key.shape[2]),
                        past_len - system_len,
                        device=rotary_device
                    )

                    # 计算旋转矩阵的 cos/sin
                    cos, sin = rotary_emb(layer_chunk_value, position_ids)
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)

                    # 应用平移旋转: 左乘 R_{past_len - system_len}
                    # 这会将 [R_0*k_0, R_1*k_1, ...] 转换为 [R_{past_len}*k_0, R_{past_len+1}*k_1, ...]
                    layer_chunk_key = (layer_chunk_key * cos) + (rotate_half(layer_chunk_key) * sin)

                    # 将更新后的 KV 移回输入设备并更新列表
                    chunk_key_cache[layer_idx] = layer_chunk_key.to(input_device)
                    chunk_value_cache[layer_idx] = layer_chunk_value.to(input_device)
            else:
                # 不需要 RoPE 调整 - 直接移动到输入设备
                chunk_key_cache = [k.to(input_device) for k in chunk_key_cache]
                chunk_value_cache = [v.to(input_device) for v in chunk_value_cache]

            # ========== 4.2 拷贝到 GPU 预先分配的内存 ==========
            # 使用 narrow + copy_ 实现原地复制，避免内存重新分配
            for layer_idx in range(len(past_key_values.key_cache)):
                # narrow(2, past_len, passage_len): 在第2维(序列维)切片
                # [past_len : past_len+passage_len] 区间
                # copy_: 原地复制数据到预分配的 past_key_values 中
                past_key_values.key_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_key_cache[layer_idx])
                past_key_values.value_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_value_cache[layer_idx])
                # 更新每层的已用长度计数
                past_key_values.past_tokens[layer_idx] += passage_len
        else:
            # 旧格式兼容代码 (单个 tensor 而非列表)
            # 正常情况下不应执行到这里
            chunk_key_cache = chunk_key_cache.to(input_device)
            chunk_value_cache = chunk_value_cache.to(input_device)
            assert passage_len == chunk_key_cache.shape[3]

            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_key_cache[layer_idx])
                past_key_values.value_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_value_cache[layer_idx])
                past_key_values.past_tokens[layer_idx] += passage_len

        # 累加 past_len: 记录当前已加载到 past_key_values 的总长度
        past_len += passage_len

    # storage_time: float, KV cache 从磁盘加载到 GPU 的总耗时
    storage_time = time.time() - start_time
    #pdb.set_trace()
    print(f'storage_time: {storage_time}')

    # ========== 5. 重要性计算与 Token 选择 ==========
    if rate != 0:
        # ========== 5.1 方法 A: FusionRAG - 使用大模型自身 attention ==========
        if reprocess_method == 'FusionRAG':
            select_time = time.time()

            # 重新计算 query_prefix_len (确保正确提取问题)
            # 通过字符串分割 'Question: ' 来定位真正的问题起始位置
            query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0])) # 通过字符串分割，只提取出 Question之后的部分

            # 兼容中文标点 'Question：' 的情况
            if query_prefix_len >= len(passages[-1]):
                query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0]))+1

            # inputs: Tensor[1, question_len], 纯问题部分的 token IDs
            # 示例: 'Who is the director of...' 的 token 序列
            inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(input_device)
            seq_length = passages[-1][query_prefix_len:].shape[0]

            # ========== 5.1.1 使用问题计算文档 token 的重要性 ==========
            # 原理: 让模型处理问题，观察哪些文档 tokens 的 attention 分数高
            # 注意: 当有 missing_chunks 时，使用临时 cache 避免在错误位置写入 question KV
            with torch.no_grad():
                ss_time = time.perf_counter()

                # inputs_embeds: Tensor[1, question_len, hidden_size], 问题的嵌入表示
                inputs_embeds = model.model.embed_tokens(inputs).to(input_device)

                # ========== 情况 1: 有缺失文档 - 使用临时 cache ==========
                # 原因: 缺失文档尚未加载，past_key_values 的位置映射会错位
                # 解决: 创建临时 cache，只包含已加载的文档，避免污染原始 cache
                if missing_chunks:
                    # 确定 cache 设备
                    if device_map is not None:
                        cache_device = device_map  # 模型并行: 使用 device_map
                    else:
                        cache_device = past_key_values.key_cache[0].device

                    # 提取 passage_len 参数 (用于创建 importance_cache)
                    # importance_cache: Tensor[num_heads, num_passages, max_seq_len]
                    # 用于记录每个 passage 对各 token 的 attention 分数
                    if hasattr(past_key_values, 'importance_cache') and len(past_key_values.importance_cache) > 0:
                        passage_len_for_cache = past_key_values.importance_cache[0].shape[1]
                    else:
                        passage_len_for_cache = past_key_values.key_cache[0].shape[2]

                    # 创建临时 StaticCache
                    # 维度与原始 past_key_values 相同，但是独立的对象
                    temp_past_key_values = StaticCache(
                        config=model.config,
                        max_batch_size=1,
                        max_cache_len=past_key_values.key_cache[0].shape[2],
                        device=cache_device,
                        dtype=past_key_values.key_cache[0].dtype,
                        passage_len=passage_len_for_cache  # 自动创建 importance_cache
                    )

                    # 将已加载的文档 KV 拷贝到临时 cache
                    # 只拷贝 [:past_len] 部分 (已加载的文档)
                    for layer_idx in range(len(past_key_values.key_cache)):
                        temp_past_key_values.key_cache[layer_idx][:, :, :past_len, :] = \
                            past_key_values.key_cache[layer_idx][:, :, :past_len, :].clone()
                        temp_past_key_values.value_cache[layer_idx][:, :, :past_len, :] = \
                            past_key_values.value_cache[layer_idx][:, :, :past_len, :].clone()

                    # cache_position: Tensor[question_len], 问题在序列中的位置
                    # 值: [past_len, past_len+1, ..., past_len+question_len-1]
                    # 用于告诉模型在 cache 的哪个位置写入 question 的 KV
                    cache_position = torch.arange(past_len, past_len + seq_length, device=input_device)

                    # 执行 forward pass: 问题 + 已加载的文档
                    # question 的 KV 会写入 temp_past_key_values (不影响原始 cache)
                    model(
                        inputs_embeds = inputs_embeds, past_key_values=temp_past_key_values,
                        cache_position=cache_position, reprocess_method=reprocess_method,
                        return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache
                    )

                    # 从临时 cache 提取重要性分数
                    # importance_cache[-1]: 最后一层的 importance 分数
                    # 维度: [num_heads, num_passages, max_seq_len]
                    # 对 num_heads 维求和, 得到 [num_passages, max_seq_len]
                    # 再对 num_passages 维求和, 得到每个 token 的总重要性 [max_seq_len]
                    if hasattr(temp_past_key_values, 'importance_cache') and len(temp_past_key_values.importance_cache) > 0:
                        k_sum = torch.sum(temp_past_key_values.importance_cache[-1], dim=0)[:past_len]
                    else:
                        # 降级策略: 如果临时 cache 没有 importance_cache, 使用原始 cache
                        print("  ⚠ Warning: temp_past_key_values has no importance_cache, using original past_key_values")
                        if hasattr(past_key_values, 'importance_cache') and len(past_key_values.importance_cache) > 0:
                            k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:past_len]
                        else:
                            raise RuntimeError("No importance_cache available for FusionRAG token selection")
                else:
                    # ========== 情况 2: 无缺失文档 - 直接使用原始 cache ==========
                    # cache_position: 问题在序列中的位置
                    cache_position = torch.arange(past_len, past_len + seq_length, device=input_device)

                    # 执行 forward pass: 问题 + 所有已加载的文档
                    model(
                        inputs_embeds = inputs_embeds, past_key_values=past_key_values,
                        cache_position=cache_position, reprocess_method=reprocess_method,
                        return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache
                    )

                    # 从原始 cache 提取重要性分数 (只取已加载部分)
                    # k_sum: Tensor[past_len], 每个 token 的重要性分数
                    k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:past_len]


            # ========== 5.1.2 基于重要性选择 top-k tokens ==========
            # k_sum: 转为 list 便于切片处理
            k_sum = k_sum.tolist()
            k_sum_original_len = len(k_sum)

            # 去除 system_prompt 部分, 只保留文档部分的重要性
            # k_sum: List[float], 长度应为 loaded_relevant_tokens
            k_sum = k_sum[system_len:]
            k_sum_doc_len = len(k_sum)
            k_sum = torch.tensor(k_sum,device=input_device)

            # DEBUG: 验证 k_sum 的长度是否正确
            if k_sum_doc_len != loaded_relevant_tokens:
                print(f"  ⚠️ WARNING: k_sum length mismatch!")
                print(f"    k_sum original length: {k_sum_original_len}")
                print(f"    k_sum after slicing system_len={system_len}: {k_sum_doc_len}")
                print(f"    Expected loaded_relevant_tokens: {loaded_relevant_tokens}")
                print(f"    Difference: {k_sum_doc_len - loaded_relevant_tokens}")

            # loaded_relevant_tokens: int, 已加载的文档 tokens 数量 (不包括 system 和 question)
            # 示例: past_len=1664, system_len=128 -> loaded_relevant_tokens=1536
            loaded_relevant_tokens = past_len - system_len  # 实际已加载的文档 tokens

            # DEBUG: 打印详细的中间计算值
            print(f"\n[DEBUG] Token selection calculation:")
            print(f"  past_len (loaded in KV): {past_len}")
            print(f"  system_len: {system_len}")
            print(f"  loaded_relevant_tokens (past_len - system_len): {loaded_relevant_tokens}")
            print(f"  rate: {rate}")
            print(f"  k_lens (rate * loaded_relevant_tokens): {k_lens}")

            # k_lens: int, 需要保留的 token 数量 = rate * loaded_relevant_tokens
            # 示例: rate=0.3, loaded_relevant_tokens=1536 -> k_lens=460
            k_lens = int(rate * loaded_relevant_tokens)

            # 打印在线懒加载模式的统计信息
            if missing_chunks:
                all_relevant_tokens = torch.cat(passages[:-1]).shape[0] - system_len
                missing_docs_len = sum([chunk[2].shape[0] for chunk in missing_chunks])
                print(f"    → ONLINE_LAZY mode: {loaded_relevant_tokens} tokens loaded, {missing_docs_len} tokens missing")
                print(f"    → Importance-based selection: {k_lens} tokens from loaded docs (rate={rate:.1f})")
                print(f"    → Missing docs will be added separately (100% coverage)")
            else:
                # 没有 missing_chunks, 所有文档都已加载
                pass

            # k_need_index: Tensor[k_lens], 选中的 token 的索引 (相对于完整序列)
            # topk 返回最大的 k_lens 个值的索引
            # 加上 system_len 转换为绝对位置
            # 示例: [128, 145, 200, ...] (包含 system_prompt 偏移)
            k_need_index = torch.topk(k_sum, k_lens).indices.to('cpu')
            k_need_index_original_len = len(k_need_index)

            k_need_index = k_need_index + system_len
            k_need_index_after_shift = len(k_need_index)

            # DEBUG: 验证选择结果
            print(f"  k_need_index length after topk: {k_need_index_original_len}")
            print(f"  k_need_index length after adding system_len: {k_need_index_after_shift}")
            print(f"  Expected k_lens: {k_lens}")

            print(f'select_time: {time.time() - select_time}')

        # ========== 5.2 方法 B: DraftModel - 使用小模型 attention 指导 ==========
        # 核心思想: 用小模型 (如 0.5B) 计算 attention, 指导大模型 (如 7B) 的 token 选择
        # 优势: 小模型计算快, 可以提前计算并复用 attention
        elif reprocess_method == 'DraftModel':
            select_time = time.time()

            # query_start: int, 问题在完整序列中的起始位置
            # Example: system(128) + doc1(512) + doc2(1024) + question -> query_start=1664
            query_start = sum(passages_len[:-1])

            # doc_len: int, **已加载文档**的总 token 数 (不包括 system, question, 和缺失文档)
            # BUG FIX: 原来是 sum(passages_len[1:-1])，包括了所有文档（已加载+缺失）
            #         这导致从所有文档中选择，然后又添加所有缺失文档，导致已加载文档的实际选择率过高
            # 正确做法: 只从已加载的文档中选择，因为缺失文档会 100% 添加
            if missing_chunks:
                # 只计算已加载文档的长度（排除缺失的）
                loaded_doc_len = 0
                missing_idx_set = {idx for idx, _, _ in missing_chunks}
                for idx in range(1, len(passages) - 1):  # 遍历所有文档（不包括 system 和 question）
                    if idx not in missing_idx_set:  # 只计算已加载的
                        loaded_doc_len += passages[idx].shape[0]
                doc_len = loaded_doc_len
                print(f"  DraftModel: 仅从已加载文档中选择，已加载文档总长度={loaded_doc_len} tokens")

                # 特殊情况：如果所有文档都缺失了，直接跳过 DraftModel 选择
                if doc_len == 0:
                    print(f"  DraftModel: 所有文档都缺失，跳过重要性选择，所有文档将 100% 重计算")
                    # 设置 k_need_index 为空（后续会添加所有缺失文档）
                    k_need_index = []
            else:
                # 没有缺失文档，doc_len 就是所有文档
                doc_len = sum(passages_len[1:-1])

            # selection_start: int, 选择区域的起始位置 (跳过 system_prompt), 从第一个文档开始选择
            selection_start = system_len

            # 如果没有传入 draft_attention，需要用 draft_model 计算
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided for DraftModel method")

                # *** 关键优化: 只向 draft model 传递已加载的文档 ***
                # 并且直接在 past_key_values 空间选择，避免后续的位置映射
                if missing_chunks:
                    # 有缺失文档：只拼接 system + 已加载文档
                    missing_idx_set = {idx for idx, _, _ in missing_chunks}
                    loaded_passages = [passages[0]]  # system
                    for idx in range(1, len(passages) - 1):  # 遍历所有文档
                        if idx not in missing_idx_set:
                            loaded_passages.append(passages[idx])  # 只添加已加载的
                    print(f"  DraftModel 输入: 只包含已加载文档 (跳过 {len(missing_chunks)} 个缺失文档)")
                else:
                    # 无缺失文档：所有文档都已加载
                    loaded_passages = list(passages[:-1])  # system + 所有文档（不包括 question）
                    print(f"  DraftModel 输入: 所有文档都已加载")

                # 添加 question 用于计算 attention
                # 此时 loaded_passages 的结构与 past_key_values 匹配：
                # - past_key_values[0:past_len]: [system, loaded_doc1, loaded_doc2, ...]
                # - loaded_passages:             [system, loaded_doc1, loaded_doc2, ...]
                loaded_passages_with_q = list(loaded_passages) + [passages[-1]]  # 添加 question
                full_input = torch.cat(loaded_passages_with_q).unsqueeze(0).to(input_device)

                # 计算 query_start（问题在 full_input 中的位置）
                query_start_in_full = full_input.shape[1] - passages[-1].shape[0]

                # 如果使用固定层且层号在前 50%，需要额外计算该层的 attention
                extra_layers = None
                if draft_layer_selection == 'fixed':
                    num_layers = draft_model.config.num_hidden_layers
                    if draft_fixed_layer < num_layers // 2:
                        extra_layers = [draft_fixed_layer]
                        print(f"  固定层 {draft_fixed_layer} 在前半部分，额外计算其 attention")

                print(f"  输入结构与 past_key_values 对齐，选择结果直接是 cache_position")

                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start_in_full, input_device, extra_layers=extra_layers)
                torch.cuda.empty_cache()

            # draft_attention 是 {layer_idx: attention [num_heads, query_len, seq_len]} 格式
            # *** 关键: full_input 只包含已加载文档，所以 attention 也只包含已加载文档 ***
            # 不需要 mask，因为缺失文档根本没有被传入 draft model

            # 收集各层的 query→doc attention（只针对已加载文档）
            layer_attention_dict = {}

            # full_input = [system, loaded_doc1, loaded_doc2, ..., question]
            # 提取文档部分的 attention（跳过 system 和 question）
            loaded_docs_start = system_len
            loaded_docs_end = full_input.shape[1] - passages[-1].shape[0]

            for layer_idx, layer_attn in draft_attention.items():
                # layer_attn: [num_heads, query_len, seq_len]
                # 提取 query→已加载文档的 attention（跳过 system 和 question）
                query_to_loaded_docs = layer_attn[:, :, loaded_docs_start:loaded_docs_end]
                # 对 heads 和 query positions 平均
                doc_attention_avg = query_to_loaded_docs.mean(axis=(0, 1))  # [loaded_doc_len]

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
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [loaded_doc_len]
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

            # *** 关键优化：选择结果直接是 cache_position，不需要映射 ***
            # 因为 full_input 的结构与 past_key_values 对齐：
            # - full_input:    [system, loaded_doc1, loaded_doc2, ..., question]
            # - past_kv[0:past_len]: [system, loaded_doc1, loaded_doc2, ...]
            # 对于已加载文档，在 full_input 中的位置 = 在 past_key_values 中的位置
            # smart_query_selection 返回的位置直接就是 cache_position！
            print(f"  选择结果直接是 cache_position，不需要 passages_to_kv_position 映射")

            # DEBUG: 打印 DraftModel 选择前的统计
            print(f"\n[DEBUG] DraftModel token selection:")
            print(f"  doc_len (loaded docs only): {doc_len}")
            print(f"  target_ratio (rate): {rate}")
            print(f"  target_count: {int(doc_len * rate)}")
            print(f"  selection_start: {selection_start}")
            print(f"  system_len: {system_len}")

            # 特殊情况：如果所有文档都缺失（doc_len=0），跳过 DraftModel 选择
            if doc_len == 0:
                print(f"  ⚠️ 所有文档都缺失，跳过 DraftModel 重要性选择")
                k_need_index = []
            else:
                # 正常情况：执行 DraftModel 选择
                # multi_layer_attn 是已加载文档的 attention（不需要 mask）

                # 使用 smart_query_selection 进行选择
                # 注意: selection_start 是选择区域的起始位置（跳过了 system）
                # 返回的 selected_positions 直接就是 cache_position（因为 full_input 与 past_kv 对齐）

                if use_similarity_rerank and draft_model is not None:
                    # 使用相似度重排序改进选择
                    # 关键改进: 先用 smart_query_selection 选候选，保留连通分量和边界扩展
                    print(f"  使用相似度重排序 (multiplier={rerank_multiplier})...")

                    # 计算 query-doc 相似度
                    similarity_scores = compute_query_doc_similarity(
                        draft_model, full_input, selection_start, selection_start + doc_len,
                        query_start_in_full, input_device
                    )

                    target_count = int(doc_len * rate)

                    # 先用 smart_query_selection 选择 rerank_multiplier 倍候选
                    # 这样保留了连通分量分析和边界扩展的优势
                    candidate_ratio = min(rate * rerank_multiplier, 1.0)
                    candidates_cache_pos = smart_query_selection(
                        attention_scores=multi_layer_attn,
                        doc_len=doc_len,
                        target_ratio=candidate_ratio,
                        system_len=selection_start,
                        device=input_device,
                        threshold_factor=draft_threshold_factor
                    )

                    # 转换为相对于 doc 的位置
                    candidates_local = [pos - selection_start for pos in candidates_cache_pos]

                    # 在候选中按相似度排序
                    candidate_sim = [(pos, similarity_scores[pos].item()) for pos in candidates_local]
                    candidate_sim.sort(key=lambda x: x[1], reverse=True)

                    # 选择相似度最高的 target_count 个
                    selected_local = [pos for pos, _ in candidate_sim[:target_count]]
                    selected_cache_pos = [pos + selection_start for pos in sorted(selected_local)]

                    print(f"  相似度重排序完成: {len(candidates_cache_pos)} 候选 -> {len(selected_cache_pos)} 最终选择")
                else:
                    # 原始方法 (使用可配置的 threshold_factor)
                    # 返回的位置直接是 cache_position
                    selected_cache_pos = smart_query_selection(
                        attention_scores=multi_layer_attn,
                        doc_len=doc_len,
                        target_ratio=rate,
                        system_len=selection_start,
                        device=input_device,
                        threshold_factor=draft_threshold_factor
                    )

                # *** 关键：selected_cache_pos 直接就是 cache_position，不需要任何映射 ***
                k_need_index = torch.tensor(selected_cache_pos, device='cpu')

                # DEBUG: 打印选择结果
                # print(f"\n[DEBUG] DraftModel selection result:")
                # print(f"  selected_indices length: {len(selected_indices)}")
                # print(f"  Expected (target_count): {int(doc_len * rate)}")
                # print(f"  Actual ratio: {len(selected_indices)/doc_len*100:.2f}%")
                # print(f"  Target ratio was: {rate*100:.2f}%")

                # print(f"DraftModel 选择了 {len(k_need_index)} 个 tokens (从已加载文档中选{len(k_need_index)/doc_len*100:.1f}%), threshold={draft_threshold_factor}")
                # print(f'select_time: {time.time() - select_time:.3f}s')

        else:
            raise NotImplementedError

        # ========== 5.3 合并所有选择的 tokens ==========
        # 对于 DraftModelLayerwise (逐层不同 rate), 使用所有层选择的并集
        if extra_info.get('layerwise_mode') and extra_info.get('per_layer_selections'):
            per_layer_selections = extra_info['per_layer_selections']
            # 使用所有层选择的并集 (第 0 层包含最多 tokens, 后续层逐渐减少)
            all_positions = set()
            for layer_sel in per_layer_selections:
                all_positions.update(layer_sel)
            k_need_index = sorted(list(all_positions))
            print(f"\n  Layerwise union positions: {len(all_positions)} unique doc positions")
            print(f"  (Note: Using union for forward pass; per-layer rates documented in extra_info)")
        else:
            # 对选中的索引排序 (保证顺序, 便于后续处理)
            k_need_index = torch.sort(torch.tensor(k_need_index))[0].tolist()

        # 添加问题部分的所有 tokens (问题总是 100% 保留)
        # range(sum(passages_len[:-1]), sum(passages_len)): 问题的索引范围
        k_need_index.extend(range(sum(passages_len[:-1]),sum(passages_len)))
    else:
        # rate == 0: 不进行压缩, 保留所有 tokens (除了 system_prompt 之外)
        # k_need_index 只包含问题部分
        k_need_index = range(sum(passages_len[:-1]),sum(passages_len))

    # ========== 6. 构建位置映射 (passages -> past_key_values) ==========
    # *** 关键优化：DraftModel 模式下，k_need_index 已经是 cache_position，不需要映射 ***
    # 但是，我们仍然需要 passages_to_kv_position 用于：
    # 1. 添加缺失文档时（这些文档的 passages 位置需要映射）
    # 2. Debug 可视化（需要知道每个 cache_position 属于哪个文档）

    # passages_len_cumsum: List[int], passages 的累计长度
    # 示例: [128, 640, 1664, 2688, 2720] 代表每个 passage 结束的位置
    passages_len_cumsum = [sum([p.shape[0] for p in passages[:i+1]]) for i in range(len(passages))]

    # passages_to_kv_position: Dict[int, int], 位置映射字典
    # key: passages 中的 token 位置 (全局位置)
    # value: past_key_values 中对应的位置
    passages_to_kv_position = {}  # Map: passages_pos -> past_key_values_pos

    # kv_to_passages_position: Dict[int, int], 反向映射字典
    # key: past_key_values 中的位置
    # value: passages 中的位置（用于 debug 可视化）
    kv_to_passages_position = {}  # Map: cache_pos -> passages_pos

    # ========== 6.1 情况 1: 有缺失文档 - 需要构建复杂映射 ==========
    if missing_chunks:
        # missing_chunk_set: Set[int], 缺失文档在 passages 中的索引集合
        missing_chunk_set = {idx for idx, _, _ in missing_chunks}

        # loaded_kv_pos: int, 已加载文档在 past_key_values 中的当前位置
        # 从 0 开始, 紧凑排列已加载的文档
        loaded_kv_pos = 0  # Track position of loaded docs in past_key_values (0 to past_len)

        # missing_kv_pos: int, 缺失文档在 past_key_values 中的起始位置
        # 从 past_len 开始, 缺失文档会被附加到已加载文档之后
        missing_kv_pos = past_len  # Track position for missing docs (starts from past_len)

        # 遍历所有文档 (不包括 question), 构建每个 token 的位置映射
        for doc_idx in range(len(passages) - 1):  # Exclude question
            # doc_start_in_passages: int, 当前文档在 passages 中的起始位置
            doc_start_in_passages = passages_len_cumsum[doc_idx-1] if doc_idx > 0 else 0
            doc_len = passages[doc_idx].shape[0]

            if doc_idx in missing_chunk_set:
                # 当前文档是缺失的: 映射到 past_len 之后的位置
                # 示例: doc2 缺失, 长度 1024
                #   passages 位置 640-1664 -> past_key_values 位置 past_len - past_len+1024
                for i in range(doc_len):
                    passages_to_kv_position[doc_start_in_passages + i] = missing_kv_pos
                    missing_kv_pos += 1
            else:
                # 当前文档已加载: 映射到紧凑排列的位置
                # 示例: doc1 已加载, 长度 512
                #   passages 位置 128-640 -> past_key_values 位置 0-512
                for i in range(doc_len):
                    passages_to_kv_position[doc_start_in_passages + i] = loaded_kv_pos
                    # 同时构建反向映射（用于 debug）
                    kv_to_passages_position[loaded_kv_pos] = doc_start_in_passages + i
                    loaded_kv_pos += 1

        # 映射问题的位置 (问题总是附加在最后)
        # question_start_in_passages: int, 问题在 passages 中的起始位置
        question_start_in_passages = passages_len_cumsum[-2]
        question_len = passages[-1].shape[0]
        for i in range(question_len):
            passages_to_kv_position[question_start_in_passages + i] = missing_kv_pos
            missing_kv_pos += 1
    else:
        # ========== 6.2 情况 2: 无缺失文档 - 恒等映射 ==========
        # passages 位置 = past_key_values 位置
        for i in range(passages_len_cumsum[-1]):
            passages_to_kv_position[i] = i
            kv_to_passages_position[i] = i

    # ========== 6.3 添加所有缺失文档的 tokens 到 k_need_index ==========
    # 重要: 缺失文档必须 100% 重计算 (因为它们没有预计算的 KV cache)
    # 这一步对 rate=0 和 rate!=0 都适用, 必须在位置映射之后执行
    if missing_chunks:
        # 确保 k_need_index 是 list 类型 (range 不支持 extend)
        if not isinstance(k_need_index, list):
            k_need_index = list(k_need_index)

        # 重新计算 passages_len_cumsum (只到倒数第二个, 不包括 question)
        passages_len_cumsum = [sum([p.shape[0] for p in passages[:i+1]]) for i in range(len(passages)-1)]

        # *** 关键：根据 reprocess_method 决定添加什么位置 ***
        if reprocess_method == 'DraftModel':
            # DraftModel: k_need_index 是 cache_position，需要添加缺失文档的 cache_position
            # 缺失文档在 past_key_values 中的位置从 past_len 开始
            missing_cache_pos = past_len
            for idx, doc_id, passage in missing_chunks:
                # 添加该文档的所有 cache_position
                doc_len = passage.shape[0]
                k_need_index.extend(range(missing_cache_pos, missing_cache_pos + doc_len))
                missing_cache_pos += doc_len
            print(f"    → Added {sum([chunk[2].shape[0] for chunk in missing_chunks])} missing document tokens (cache_position) to reprocess")
        else:
            # 其他方法: k_need_index 是 passages 位置，直接添加 passages 位置
            for idx, doc_id, passage in missing_chunks:
                # chunk_start: int, 缺失文档在 passages 中的起始位置
                chunk_start = passages_len_cumsum[idx-1] if idx > 0 else 0
                # chunk_end: int, 缺失文档在 passages 中的结束位置
                chunk_end = passages_len_cumsum[idx]
                # 添加该文档的所有位置 [chunk_start, chunk_end)
                k_need_index.extend(range(chunk_start, chunk_end))
            print(f"    → Added {sum([chunk[2].shape[0] for chunk in missing_chunks])} missing document tokens (passages_position) to reprocess")

        # 排序以保持顺序一致性
        k_need_index = sorted(k_need_index)
        
    # ========== 7. 准备重计算输入 ==========
    # 注意: past_len 保持为实际加载的长度, 不要重置!
    # 错误做法: past_len = sum(passages_len)  # 这是离线模式的做法
    # 原因:
    #   - 离线模式: 所有文档都已加载, past_len == sum(passages_len)
    #   - 在线懒加载: 只有部分文档加载, past_len < sum(passages_len)

    # final_len: int, 重计算后的最终序列长度 (包括所有 passages)
    # 示例: system(128) + doc1(512) + doc2(1024) + question(32) = 1696
    final_len = sum(passages_len)  # Total length after reprocess

    # batch_size: int, 批次大小 (固定为 1)
    # seq_length: int, 需要重计算的 token 数量
    # 示例: rate=0.3 时, seq_length 可能是 460 (文档) + 32 (问题) = 492
    batch_size, seq_length = 1, len(k_need_index)

    # generated_ids: Tensor[1, final_len + max_new_tokens + 1], 存储生成的完整序列
    # 用途: 记录输入 prompt 和生成的 tokens
    generated_ids = torch.zeros(
        batch_size, final_len + max_new_tokens + 1, dtype=torch.int, device=input_device
    )
    # 填充输入部分 (所有 passages 拼接)
    generated_ids[:, :final_len] = torch.cat(passages).unsqueeze(0).to(input_device)

    # tokens: List[Tensor], 存储生成的 tokens (不包括 prompt)
    tokens = []

    # use_sparse_attention: bool, 是否使用稀疏 attention (当前未启用)
    if reprocess_method != 'FusionRAG':
        use_sparse_attention = False
    else:
        use_sparse_attention = False

    # ========== 7.1 构建重计算输入 ==========
    # reprocess_inputs: Tensor[1, seq_length], 需要重计算的 tokens
    # 从完整序列中提取 k_need_index 指定的位置
    # 示例: 如果 k_need_index=[128, 145, 200, ...], 则提取这些位置的 token

    # *** 关键：根据 reprocess_method 决定如何提取 ***
    if reprocess_method == 'DraftModel':
        # DraftModel: k_need_index 已经是 cache_position
        # 但是 reprocess_inputs 需要 passages 中的 token，所以需要转换
        # 使用 kv_to_passages_position 将 cache_position 转换为 passages 位置
        k_need_passages_positions = [kv_to_passages_position.get(pos, pos) for pos in k_need_index]
        reprocess_inputs = torch.cat(passages)[k_need_passages_positions].unsqueeze(0).to(input_device)
        # k_need_index 已经是 cache_position，直接使用
        cache_position = torch.tensor(k_need_index, device=input_device)
    else:
        # 其他方法（FusionRAG）: k_need_index 是 passages 位置
        reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(input_device)
        # 需要将 passages 位置转换为 cache_position
        cache_position = torch.tensor([passages_to_kv_position[pos] for pos in k_need_index], device=input_device)

    # ========== DEBUG: 打印每个文档的重计算 token 统计 ==========
    print(f"\n{'='*80}")
    print(f"重计算 Token 统计 (rate={rate})")
    print(f"{'='*80}")

    # 将 k_need_index 转为集合便于查询
    if isinstance(k_need_index, list):
        k_need_set = set(k_need_index)
    else:
        k_need_set = set(k_need_index)

    # *** 关键：k_need_index 现在可能是 cache_position（DraftModel）或 passages 位置（其他方法） ***
    # 使用 kv_to_passages_position 将 cache_position 转换为 passages 位置用于统计
    if reprocess_method == 'DraftModel':
        # DraftModel: k_need_index 是 cache_position
        # 分离已加载文档和缺失文档的 cache_position
        loaded_cache_positions = [pos for pos in k_need_set if pos < past_len]
        missing_cache_positions = [pos for pos in k_need_set if pos >= past_len]

        # 转换已加载文档的 cache_position 为 passages 位置
        loaded_passages_positions = [kv_to_passages_position.get(pos, pos) for pos in loaded_cache_positions]

        # 缺失文档总是 100%，不需要转换，直接用 passages 位置范围计算
        # 我们知道缺失文档的 passages 位置范围，所以在统计时直接标记为 100%

        # 合并所有 passages 位置用于统计
        k_need_passages_positions = loaded_passages_positions  # 缺失文档在下面的循环中单独处理
    else:
        # 其他方法: k_need_index 已经是 passages 位置
        k_need_passages_positions = list(k_need_set)
        missing_cache_positions = []  # 没有单独的缺失文档 cache_position

    # passages_len_cumsum: 每个 passage 结束位置的累计
    passages_len_cumsum = [sum([p.shape[0] for p in passages[:i+1]]) for i in range(len(passages))]

    # 统计每个文档的重计算情况
    for doc_idx in range(len(passages)):
        # 计算当前文档的起始和结束位置
        doc_start = passages_len_cumsum[doc_idx-1] if doc_idx > 0 else 0
        doc_end = passages_len_cumsum[doc_idx]
        doc_total_len = passages[doc_idx].shape[0]

        # 确定文档类型标签
        if doc_idx == 0:
            doc_label = "System Prompt"
            # System 不参与重计算，应该总是 0
            doc_tokens_recompute = 0
            doc_coverage_pct = 0.0
        elif doc_idx == len(passages) - 1:
            doc_label = "Question"
            # Question 总是 100% 保留
            doc_tokens_recompute = doc_total_len
            doc_coverage_pct = 100.0
        else:
            # 检查是否是缺失文档
            is_missing = doc_idx in {idx for idx, _, _ in missing_chunks} if missing_chunks else False
            doc_label = f"Doc{doc_idx} (missing)" if is_missing else f"Doc{doc_idx}"

            if is_missing:
                # 缺失文档总是 100% 重计算
                doc_tokens_recompute = doc_total_len
                doc_coverage_pct = 100.0
            else:
                # 已加载文档：统计有多少 token 被选中（使用 passages 位置）
                doc_tokens_recompute = len([pos for pos in k_need_passages_positions if doc_start <= pos < doc_end])
                doc_coverage_pct = (doc_tokens_recompute / doc_total_len * 100) if doc_total_len > 0 else 0

        # 打印该文档的统计信息
        print(f"  {doc_label:20s}: {doc_tokens_recompute:5d} / {doc_total_len:5d} tokens ({doc_coverage_pct:6.2f}%)")

    # 打印总体统计
    total_tokens = sum([p.shape[0] for p in passages])
    total_recompute = len(k_need_set)
    overall_coverage = (total_recompute / total_tokens * 100) if total_tokens > 0 else 0
    print(f"  {'-'*20}")
    print(f"  {'Total':20s}: {total_recompute:5d} / {total_tokens:5d} tokens ({overall_coverage:6.2f}%)")
    print(f"{'='*80}\n")
    # pdb.set_trace()
    
    # ========== DETAILED DEBUG: Compare rate=0.0 vs rate>0 ==========
    debug = False
    if debug and missing_chunks or rate > 0:
        print(f"\n{'='*80}")
        print(f"RECOMPUTE TOKEN DETAILED DEBUG (rate={rate})")
        print(f"{'='*80}")
        print(f"\n[1] KV Cache State:")
        print(f"  past_len (loaded in past_key_values): {past_len}")
        print(f"  sum(passages_len): {sum(passages_len)}")
        print(f"  final_len (expected after recompute): {final_len}")

        print(f"\n[2] Passages Structure:")
        for i, p in enumerate(passages):
            start = passages_len_cumsum[i-1] if i > 0 else 0
            end = passages_len_cumsum[i] if i < len(passages_len_cumsum) else passages_len_cumsum[-1] + p.shape[0]
            doc_type = "Question" if i == len(passages)-1 else f"Doc{i}"
            is_missing = i in {idx for idx, _, _ in missing_chunks} if missing_chunks else False
            status = "MISSING" if is_missing else "Loaded" if i < len(passages)-1 else "To append"
            print(f"  {doc_type}: passages[{start}:{end}] ({p.shape[0]} tokens) - {status}")

        if missing_chunks:
            print(f"\n[3] Missing Chunks Info:")
            for idx, doc_id, passage in missing_chunks:
                chunk_start = passages_len_cumsum[idx-1] if idx > 0 else 0
                chunk_end = passages_len_cumsum[idx]
                print(f"  Doc{idx} (doc_id={doc_id}): passages[{chunk_start}:{chunk_end}] ({chunk_end-chunk_start} tokens)")

        print(f"\n[4] Position Mapping (sample):")
        # Show first 5 and last 5 mappings for each document
        prev_doc = -1
        for passages_pos in sorted(passages_to_kv_position.keys())[:20]:
            kv_pos = passages_to_kv_position[passages_pos]
            # Find which doc this position belongs to
            doc_idx = 0
            for i in range(len(passages_len_cumsum)):
                if passages_pos < passages_len_cumsum[i]:
                    doc_idx = i
                    break
            if doc_idx != prev_doc:
                print(f"  --- Doc{doc_idx} ---")
                prev_doc = doc_idx
            print(f"  passages[{passages_pos}] -> kv[{kv_pos}]")

        print(f"\n[5] k_need_index Analysis (tokens to recompute):")
        print(f"  Total tokens to recompute: {len(k_need_index)}")
        print(f"  k_need_index (passages positions): {k_need_index[:10]}...{k_need_index[-10:] if len(k_need_index) > 10 else []}")

        print(f"\n[6] cache_position Analysis (where to write in past_key_values):")
        print(f"  cache_position (kv positions): {cache_position[:10].tolist()}...{cache_position[-10:].tolist() if len(cache_position) > 10 else []}")
        print(f"  cache_position range: [{cache_position.min().item()}, {cache_position.max().item()}]")
        print(f"  cache_position is monotonic: {torch.all(cache_position[1:] >= cache_position[:-1]).item()}")

        print(f"\n[7] Token Distribution:")
        # Analyze which documents' tokens are being recomputed
        k_need_set = set(k_need_index)
        for i in range(len(passages)):
            start = passages_len_cumsum[i-1] if i > 0 else 0
            end = passages_len_cumsum[i] if i < len(passages_len_cumsum) else passages_len_cumsum[-1] + passages[i].shape[0]
            tokens_in_recompute = len(k_need_set & set(range(start, end)))
            total_tokens = passages[i].shape[0]
            pct = tokens_in_recompute / total_tokens * 100 if total_tokens > 0 else 0
            doc_type = "Question" if i == len(passages)-1 else f"Doc{i}"
            print(f"  {doc_type}: {tokens_in_recompute}/{total_tokens} tokens ({pct:.1f}%)")

        print(f"{'='*80}\n")
        print(f"  sum(passages_len) (total): {sum(passages_len)}")
        print(f"  k_need_index length: {len(k_need_index)}")
        print(f"  k_need_index range: [{min(k_need_index)}, {max(k_need_index)}]")
        print(f"  cache_position range: [{cache_position.min().item()}, {cache_position.max().item()}]")

        # 详细分析 k_need_index 的组成
        passages_len_cumsum = [sum([p.shape[0] for p in passages[:i+1]]) for i in range(len(passages))]
        question_start = passages_len_cumsum[-2]
        question_end = passages_len_cumsum[-1]

        k_need_set = set(k_need_index)
        question_tokens = set(range(question_start, question_end))
        question_in_recompute = len(k_need_set & question_tokens)

        print(f"  Question tokens: [{question_start}, {question_end}) = {question_end - question_start} tokens")
        print(f"  Question tokens in k_need_index: {question_in_recompute}")

        if missing_chunks:
            print(f"  Missing chunks: {len(missing_chunks)} documents")
            for idx, doc_id, passage in missing_chunks:
                chunk_start = passages_len_cumsum[idx-1] if idx > 0 else 0
                chunk_end = passages_len_cumsum[idx]
                missing_tokens = set(range(chunk_start, chunk_end))
                missing_in_recompute = len(k_need_set & missing_tokens)
                print(f"    Doc {doc_id}: [{chunk_start}, {chunk_end}) = {chunk_end - chunk_start} tokens, in k_need_index: {missing_in_recompute}")

        # 计算从 cached docs 选中的 tokens
        cached_tokens = k_need_set - question_tokens
        if missing_chunks:
            for idx, doc_id, passage in missing_chunks:
                chunk_start = passages_len_cumsum[idx-1] if idx > 0 else 0
                chunk_end = passages_len_cumsum[idx]
                cached_tokens -= set(range(chunk_start, chunk_end))

        print(f"  Tokens from cached docs (importance-based): {len(cached_tokens)}")
        print(f"  Total recompute tokens: {len(k_need_index)}")
        print("=" * 60)

    # ========== 8. 执行重计算 Forward Pass ==========
    with torch.no_grad():
        # ========== 8.1 保存重计算前的 value cache (用于分析) ==========
        # without_attn_value: 重计算前的最后一层 value cache
        # 只提取已加载的部分 (不包括即将生成的 missing chunks)
        if missing_chunks:
            # 只提取已加载的文档部分 [:past_len]
            without_attn_value = past_key_values.value_cache[-1].narrow(2, 0, past_len).clone()
        else:
            # 提取所有文档部分 (不包括 question)
            without_attn_value = past_key_values.value_cache[-1].narrow(2, 0, sum(passages_len[:-1])).clone()

        # ========== 8.2 将输入 token 转换为 embedding ==========
        # inputs_embeds: Tensor[1, seq_length, hidden_size], 输入 tokens 的嵌入向量
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(input_device)

        # 注意: 不强制移动到 input_device
        # 原因: 在模型并行 (Pipeline Parallelism) 模式下, 避免跨 GPU 传输死锁
        # 让模型输出保持在其自然的设备上

        # ========== 8.3 执行 forward pass: 重计算选中的 tokens ==========
        # 输入:
        #   - inputs_embeds: 需要重计算的 tokens 的嵌入 [1, seq_length, hidden_size]
        #   - cache_position: 这些 tokens 在 past_key_values 中的写入位置 [seq_length]
        #   - past_key_values: 包含已加载文档 KV 的 cache
        # 输出:
        #   - model_output: logits [1, seq_length, vocab_size]
        # 副作用:
        #   - past_key_values 被原地更新, 写入重计算的 KV
        model_output = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position,
            past_key_values=past_key_values, return_dict=False, use_cache=True, use_sparse_attention=use_sparse_attention,
        )[0]

        # ========== 9. 提取并保存新生成的 KV Cache (缺失文档) ==========
        # 如果有缺失文档, 它们的 KV 刚刚在 forward pass 中生成
        # 现在需要提取并保存到磁盘, 供后续查询复用
        if missing_chunks:
            print(f"    ⚡ Extracting and saving KV for {len(missing_chunks)} new document(s)...")
            passages_len_cumsum = [sum([p.shape[0] for p in passages[:i+1]]) for i in range(len(passages)-1)]

            # 遍历每个缺失的文档, 提取其 KV cache 并保存到磁盘
            for idx, doc_id, passage in missing_chunks:
                # ========== 9.1 计算文档在 past_key_values 中的位置 ==========
                # *** BUG 修复 ***: 使用映射后的 past_key_values 位置, 而非 passages 位置

                # chunk_start_in_passages: int, 文档在 passages 中的起始位置
                chunk_start_in_passages = passages_len_cumsum[idx-1] if idx > 0 else 0
                # chunk_end_in_passages: int, 文档在 passages 中的结束位置
                chunk_end_in_passages = passages_len_cumsum[idx]
                chunk_len = chunk_end_in_passages - chunk_start_in_passages

                # 使用位置映射将 passages 位置转换为 past_key_values 位置
                # chunk_start_in_kv: int, 文档在 past_key_values 中的起始位置
                chunk_start_in_kv = passages_to_kv_position[chunk_start_in_passages]
                # chunk_end_in_kv: int, 文档在 past_key_values 中的结束位置 (+1 因为切片是左闭右开)
                chunk_end_in_kv = passages_to_kv_position[chunk_end_in_passages - 1] + 1  # +1 because end is exclusive

                # ========== 9.2 提取所有层的 KV cache ==========
                num_layers = len(past_key_values.key_cache)
                chunk_key_all_layers = []    # List[Tensor], 存储所有层的 key
                chunk_value_all_layers = []  # List[Tensor], 存储所有层的 value

                for layer_idx in range(num_layers):
                    # 从 past_key_values 中切片提取该文档的 KV
                    # 维度: [1, num_heads, chunk_len, head_dim]
                    layer_chunk_key = past_key_values.key_cache[layer_idx][:, :, chunk_start_in_kv:chunk_end_in_kv, :].clone()
                    layer_chunk_value = past_key_values.value_cache[layer_idx][:, :, chunk_start_in_kv:chunk_end_in_kv, :].clone()

                    # ========== 9.3 应用 RoPE 逆变换 (还原到相对位置 0) ==========
                    # 目的: 将 KV cache 中的绝对位置编码还原为相对位置 (从 0 开始)
                    # 这样保存的 KV cache 可以在任何位置复用
                    if revert_rope and doc_id != SYSTEM_PROMPT_ID:
                        # 获取当前层的 rotary embedding 模块
                        rotary_emb = model.model.layers[layer_idx].self_attn.rotary_emb

                        # original_position_ids: 文档在当前序列中的绝对位置
                        # 示例: [512, 513, ..., 1536] (如果文档在位置 512 开始)
                        original_position_ids = torch.arange(chunk_start_in_kv, chunk_end_in_kv, device=layer_chunk_key.device).unsqueeze(0)

                        # target_position_ids: 目标相对位置 (从 0 开始)
                        # [0, 1, 2, ..., chunk_len-1]
                        target_position_ids = torch.arange(0, chunk_len, device=layer_chunk_key.device).unsqueeze(0)

                        # 计算两种位置的 cos/sin (用于 RoPE 变换)
                        original_cos, original_sin = rotary_emb(layer_chunk_value, original_position_ids)
                        target_cos, target_sin = rotary_emb(layer_chunk_value, target_position_ids)

                        # 逆变换: 移除原始位置的 RoPE 编码 (使用 -sin)
                        layer_chunk_key = apply_rotary_pos_emb_single(layer_chunk_key, original_cos, -original_sin)
                        # 正变换: 应用相对位置 0 的 RoPE 编码
                        layer_chunk_key = apply_rotary_pos_emb_single(layer_chunk_key, target_cos, target_sin)

                    chunk_key_all_layers.append(layer_chunk_key)
                    chunk_value_all_layers.append(layer_chunk_value)

                # ========== 9.4 原子保存到磁盘 ==========
                # 使用临时文件 + 重命名的方式保证原子性 (避免写入过程中崩溃导致文件损坏)
                key_save_path = f'{load_path}/doc_{doc_id}_key.pt'
                value_save_path = f'{load_path}/doc_{doc_id}_value.pt'
                key_temp_path = f'{key_save_path}.tmp'      # 临时文件路径
                value_temp_path = f'{value_save_path}.tmp'

                try:
                    # 步骤 1: 先写入临时文件 (移到 CPU 以节省显存)
                    torch.save([k.to('cpu') for k in chunk_key_all_layers], key_temp_path)
                    torch.save([v.to('cpu') for v in chunk_value_all_layers], value_temp_path)

                    # 步骤 2: 原子重命名 (覆盖旧文件, 如果存在)
                    # 重命名操作在大多数文件系统上是原子的
                    os.rename(key_temp_path, key_save_path)
                    os.rename(value_temp_path, value_save_path)

                    print(f"      ✓ Doc {doc_id} KV generated and saved ({chunk_len} tokens)")
                except Exception as e:
                    # 错误处理: 清理临时文件
                    if os.path.exists(key_temp_path):
                        os.remove(key_temp_path)
                    if os.path.exists(value_temp_path):
                        os.remove(value_temp_path)
                    print(f"      ✗ Failed to save Doc {doc_id} KV: {e}")
                    raise

        # ========== 10. 生成第一个 token (Prefill 阶段) ==========
        # logits: Tensor[1, 1, vocab_size], 最后一个位置的 logits (用于生成第一个 token)
        logits = model_output[:,-1,:].unsqueeze(0).clone()

        # ========== 10.1 提取重计算后的 value cache (用于分析差异) ==========
        # with_attn_value: 重计算后的最后一层 value cache
        # if missing_chunks:
        #     # 提取已加载的文档部分 (与 without_attn_value 长度相同)
        #     with_attn_value = past_key_values.value_cache[-1].narrow(2, 0, past_len).clone()
        # else:
        #     # 提取所有文档部分
        #     with_attn_value = past_key_values.value_cache[-1].narrow(2, 0, sum(passages_len[:-1])).clone()

        # # v_sub_all: 重计算前后 value cache 的差异
        # # 用于分析重计算对 cache 的影响 (调试用)
        # v_sub_all = without_attn_value - with_attn_value
        # v_sub_all = v_sub_all.squeeze(0)                 # [num_heads, seq_len, head_dim]
        # v_sub_all = v_sub_all.transpose(0, 1)            # [seq_len, num_heads, head_dim]
        # v_sum = torch.sum(v_sub_all**2, dim=[1,2])       # [seq_len], L2 范数

        # ========== 10.2 生成第一个 token ==========
        # first_token_time: float, prefill 阶段总耗时 (包括加载 KV + 重计算 + 第一个 token)
        first_token_time = time.time() - start_time

        # stream: TextStreamer, 用于流式输出生成的文本 (如果需要)
        stream = TextStreamer(tokenizer)

        # logits_warper: 对 logits 应用温度缩放和 top-k 过滤
        # temperature=0.01: 接近贪心解码 (取最大概率)
        # top_k=1: 只保留概率最高的 1 个 token
        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)

        # next_token_scores: Tensor[1, vocab_size], 经过 warper 处理的 logits
        next_token_scores = logits_warper(reprocess_inputs, logits[:, -1, :])

        # next_token: Tensor[1], 选中的下一个 token
        next_token = torch.argmax(next_token_scores, dim=-1)

        # 记录 prefill 阶段的统计信息
        prefill_count = seq_length   # 重计算的 token 数量
        prefill_time = first_token_time

        # 可选: 流式打印第一个生成的 token
        # print(stream.put(next_token.item()), end="", flush=True)

        # 将第一个生成的 token 写入 generated_ids
        # 注意: +1 是因为 final_len 位置已被占用
        generated_ids[:, final_len+1] = next_token

        # tokens: List[Tensor], 记录所有生成的 tokens
        tokens.append(next_token)

        # ========== 10.3 准备自回归解码 ==========
        # output_device: 模型输出所在的设备 (在模型并行模式下可能与输入设备不同)
        output_device = next_token.device

        # inputs: Tensor[1, final_len+1], 完整输入序列 (passages + 第一个生成的 token)
        # 用于后续的解码过程
        inputs = torch.cat((torch.cat(passages).unsqueeze(0).to(output_device), next_token.unsqueeze(0)), dim=-1)

        # cache_position: Tensor[1], 下一个 token 在 cache 中的位置
        # 初始值为 final_len (第一个生成的 token 的位置)
        cache_position = torch.tensor([final_len], device=output_device)

        # position_ids: Tensor[1, 1], 位置编码 ID
        position_ids = cache_position.unsqueeze(0)

        # seq_length: 当前序列总长度
        seq_length += 1

        # ========== 11. 自回归解码 (Decode 阶段) ==========
        decode_time = time.time()

        # 循环生成剩余的 tokens (最多 max_new_tokens-1 个, 因为已生成第一个)
        for _ in range(1, max_new_tokens):
            # decode_one_tokens: 生成下一个 token
            # 输入: 当前 token, 位置 ID, cache 位置, past_key_values
            # 输出: 下一个 token
            next_token = decode_one_tokens(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, inputs)

            # 将新生成的 token 拼接到 inputs
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)

            # 将新 token 写入 generated_ids
            generated_ids[:, cache_position] = next_token.int()

            # 记录生成的 token
            tokens.append(next_token.int())
            seq_length += 1

            # 检查是否达到结束条件
            # 1. 生成了 EOS token
            # 2. 生成了 '<|im_end|>' (对话结束标记)
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>':
                break

            # 更新位置信息，准备生成下一个 token
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)


    # ========== 12. 统计和返回结果 ==========
    # 计算解码阶段的性能统计
    total_time = time.time() - decode_time       # 解码总耗时
    tokens_generated = len(tokens)                # 生成的 token 数量
    tokens_per_second = tokens_generated / total_time  # 生成速度 (tokens/s)

    print("")  # 打印空行

    # 可选: 打印详细的性能统计 (已注释)
    # print(f"prompt eval count:    {prefill_count} token(s)")
    # print(f"prompt eval duration: {prefill_time}s")
    # print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    # print(f"eval count:           {tokens_generated} token(s)")
    # print(f"eval duration:        {total_time}s")
    # print(f"eval rate:            {tokens_per_second} tokens/s")

    # 返回结果
    # tokens: List[Tensor], 生成的所有 tokens
    # prefill_time: float, prefill 阶段耗时 (秒)
    # extra_info: dict, 额外统计信息 (dynamic_rate, coverage 等)
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
            config = past_key_values.config, max_batch_size=1, max_cache_len=seq_length+max_new_tokens, device=cache_device, dtype=model.dtype
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
