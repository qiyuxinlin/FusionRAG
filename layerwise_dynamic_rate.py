#!/usr/bin/env python3
"""
逐层动态调整重算比例

核心思想：
- 第 0 层用较高的 rate（如 30%）
- 在重算过程中，根据信号动态调整后续层的 rate
- 可能的信号：
  1. KV 变化幅度：重算后 KV 和原始 KV 的差异
  2. Attention 覆盖率：选中的 token 覆盖了多少 attention
  3. 逐层递减策略：简单的线性/指数递减
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from ktransformers.util.utils import rotate_half


def compute_kv_difference(
    original_k: torch.Tensor,
    original_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    positions: List[int]
) -> float:
    """
    计算重算后 KV 和原始 KV 的差异

    返回: 归一化的差异值 (0~1)，越小说明变化越小
    """
    if len(positions) == 0:
        return 0.0

    # 提取选中位置的 KV
    orig_k_selected = original_k[:, :, positions, :]
    orig_v_selected = original_v[:, :, positions, :]
    new_k_selected = new_k[:, :, positions, :]
    new_v_selected = new_v[:, :, positions, :]

    # 计算余弦相似度
    k_sim = F.cosine_similarity(
        orig_k_selected.reshape(-1, orig_k_selected.shape[-1]),
        new_k_selected.reshape(-1, new_k_selected.shape[-1]),
        dim=-1
    ).mean()

    v_sim = F.cosine_similarity(
        orig_v_selected.reshape(-1, orig_v_selected.shape[-1]),
        new_v_selected.reshape(-1, new_v_selected.shape[-1]),
        dim=-1
    ).mean()

    # 差异 = 1 - 相似度
    avg_diff = 1.0 - (k_sim + v_sim) / 2

    return avg_diff.item()


def compute_layer_rate_by_decay(
    layer_idx: int,
    num_layers: int,
    initial_rate: float = 0.30,
    final_rate: float = 0.05,
    decay_type: str = "linear"
) -> float:
    """
    简单的逐层递减策略

    decay_type:
    - "linear": 线性递减
    - "exponential": 指数递减
    - "cosine": 余弦递减
    - "step": 阶梯递减
    """
    progress = layer_idx / (num_layers - 1) if num_layers > 1 else 0

    if decay_type == "linear":
        rate = initial_rate - (initial_rate - final_rate) * progress

    elif decay_type == "exponential":
        # 指数衰减：rate = initial * (final/initial)^progress
        ratio = final_rate / initial_rate
        rate = initial_rate * (ratio ** progress)

    elif decay_type == "cosine":
        # 余弦衰减：前期慢，后期快
        rate = final_rate + (initial_rate - final_rate) * (1 + np.cos(np.pi * progress)) / 2

    elif decay_type == "step":
        # 阶梯递减：分成 4 段
        if progress < 0.25:
            rate = initial_rate
        elif progress < 0.5:
            rate = initial_rate * 0.7
        elif progress < 0.75:
            rate = initial_rate * 0.4
        else:
            rate = final_rate

    else:
        rate = initial_rate

    return max(final_rate, min(initial_rate, rate))


def compute_layer_rate_by_kv_diff(
    kv_diff: float,
    base_rate: float,
    min_rate: float = 0.05,
    sensitivity: float = 2.0
) -> float:
    """
    根据 KV 差异动态调整 rate

    - kv_diff 大（差异大）→ 保持较高 rate
    - kv_diff 小（差异小）→ 可以降低 rate
    """
    # kv_diff 通常在 0~0.5 之间
    # 当 kv_diff 接近 0 时，rate 可以降到 min_rate
    # 当 kv_diff 较大时，rate 保持较高

    # 使用 sigmoid 映射
    adjustment = 1.0 / (1.0 + np.exp(-sensitivity * (kv_diff - 0.1)))
    rate = min_rate + (base_rate - min_rate) * adjustment

    return rate


def select_tokens_for_layer(
    attention_scores: torch.Tensor,
    doc_len: int,
    target_ratio: float,
    system_len: int = 0,
    strategy: str = "top_k"
) -> List[int]:
    """
    为当前层选择需要重算的 tokens

    attention_scores: [doc_len] 的 attention 分数
    target_ratio: 目标选择比例
    """
    num_to_select = max(1, int(doc_len * target_ratio))

    if strategy == "top_k":
        # 简单的 top-k 选择
        _, indices = torch.topk(attention_scores, num_to_select)
        selected = indices.cpu().numpy().tolist()

    elif strategy == "threshold":
        # 基于阈值选择
        threshold = attention_scores.quantile(1 - target_ratio)
        selected = torch.where(attention_scores >= threshold)[0].cpu().numpy().tolist()

    else:
        selected = list(range(num_to_select))

    # 转换为绝对位置
    selected_positions = [system_len + pos for pos in selected]

    return sorted(selected_positions)


class LayerwiseDynamicRateRecomputer:
    """
    逐层动态重算比例管理器

    支持多种策略：
    1. decay: 简单的逐层递减
    2. kv_diff: 根据 KV 变化幅度调整
    3. attention_coverage: 根据 attention 覆盖率调整
    """

    def __init__(
        self,
        num_layers: int,
        initial_rate: float = 0.30,
        final_rate: float = 0.05,
        strategy: str = "decay",
        decay_type: str = "linear",
        verbose: bool = True
    ):
        self.num_layers = num_layers
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.strategy = strategy
        self.decay_type = decay_type
        self.verbose = verbose

        # 记录每层的实际 rate
        self.layer_rates = []
        self.layer_kv_diffs = []

    def get_rate_for_layer(
        self,
        layer_idx: int,
        prev_kv_diff: Optional[float] = None,
        prev_attention_coverage: Optional[float] = None
    ) -> float:
        """
        获取当前层的重算比例
        """
        if self.strategy == "decay":
            rate = compute_layer_rate_by_decay(
                layer_idx, self.num_layers,
                self.initial_rate, self.final_rate,
                self.decay_type
            )

        elif self.strategy == "kv_diff":
            if layer_idx == 0 or prev_kv_diff is None:
                rate = self.initial_rate
            else:
                # 根据上一层的 KV 差异调整
                base_rate = self.layer_rates[-1] if self.layer_rates else self.initial_rate
                rate = compute_layer_rate_by_kv_diff(
                    prev_kv_diff, base_rate, self.final_rate
                )

        elif self.strategy == "attention_coverage":
            if layer_idx == 0 or prev_attention_coverage is None:
                rate = self.initial_rate
            else:
                # 如果上一层的 attention 覆盖率高，可以降低 rate
                base_rate = self.layer_rates[-1] if self.layer_rates else self.initial_rate
                # 覆盖率高 → 降低 rate
                rate = base_rate * (1.0 - 0.5 * prev_attention_coverage)
                rate = max(self.final_rate, rate)

        else:
            rate = self.initial_rate

        self.layer_rates.append(rate)
        return rate

    def record_kv_diff(self, kv_diff: float):
        self.layer_kv_diffs.append(kv_diff)

    def get_summary(self) -> Dict:
        return {
            'strategy': self.strategy,
            'decay_type': self.decay_type if self.strategy == 'decay' else None,
            'initial_rate': self.initial_rate,
            'final_rate': self.final_rate,
            'layer_rates': self.layer_rates,
            'avg_rate': np.mean(self.layer_rates) if self.layer_rates else 0,
            'layer_kv_diffs': self.layer_kv_diffs,
        }

    def print_summary(self):
        print(f"\n{'='*80}")
        print("Layerwise Dynamic Rate Summary")
        print(f"{'='*80}")
        print(f"Strategy: {self.strategy}")
        if self.strategy == 'decay':
            print(f"Decay type: {self.decay_type}")
        print(f"Initial rate: {self.initial_rate:.2%}")
        print(f"Final rate: {self.final_rate:.2%}")
        print(f"Average rate: {np.mean(self.layer_rates):.2%}")
        print(f"\nPer-layer rates:")
        for i, rate in enumerate(self.layer_rates):
            if i % 4 == 0 or i == len(self.layer_rates) - 1:
                kv_diff_str = f", kv_diff={self.layer_kv_diffs[i]:.4f}" if i < len(self.layer_kv_diffs) else ""
                print(f"  Layer {i:2d}: {rate:.2%}{kv_diff_str}")


def sparse_prefill_with_dynamic_rate(
    model,
    past_key_values,
    base_attention_scores: torch.Tensor,  # [doc_len] 基础的 attention 分数（如来自 draft model）
    query_tensor: torch.Tensor,
    prefix_len: int,
    doc_len: int,
    position_to_token: Dict[int, int],
    initial_rate: float = 0.30,
    final_rate: float = 0.05,
    decay_strategy: str = "linear",  # "linear", "exponential", "cosine", "step", "kv_diff"
    device: str = "cuda:0"
):
    """
    使用逐层动态重算比例的 Sparse Prefill

    参数:
        model: 主模型
        past_key_values: 预计算的 KV cache
        base_attention_scores: 基础 attention 分数 [doc_len]
        query_tensor: 查询 tensor [1, query_len]
        prefix_len: prefix (system prompt) 的长度
        doc_len: 文档的总长度
        position_to_token: 位置到 token id 的映射
        initial_rate: 第 0 层的重算比例
        final_rate: 最后一层的重算比例
        decay_strategy: 递减策略
    """
    print(f"\n{'='*100}")
    print("Sparse Prefill with Dynamic Layerwise Rate")
    print(f"{'='*100}\n")

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    query_len = query_tensor.shape[1]

    print(f"Config:")
    print(f"  Layers: {num_layers}")
    print(f"  Doc length: {doc_len}")
    print(f"  Query length: {query_len}")
    print(f"  Initial rate: {initial_rate:.2%}")
    print(f"  Final rate: {final_rate:.2%}")
    print(f"  Decay strategy: {decay_strategy}")

    # 初始化动态 rate 管理器
    if decay_strategy == "kv_diff":
        rate_manager = LayerwiseDynamicRateRecomputer(
            num_layers, initial_rate, final_rate,
            strategy="kv_diff"
        )
    else:
        rate_manager = LayerwiseDynamicRateRecomputer(
            num_layers, initial_rate, final_rate,
            strategy="decay", decay_type=decay_strategy
        )

    # 准备 query 的 position ids
    query_positions = list(range(prefix_len + doc_len, prefix_len + doc_len + query_len))
    query_position_ids = torch.tensor([query_positions], dtype=torch.long, device=device)

    # Query embedding
    query_embeds = model.model.embed_tokens(query_tensor)  # [1, query_len, hidden_size]

    # 用于存储每层选中的 positions
    all_layer_selections = []

    # 上一层的 KV 差异（用于 kv_diff 策略）
    prev_kv_diff = None

    print(f"\n{'Layer':<8} {'Rate':<10} {'#Selected':<12} {'KV Diff':<12}")
    print("-" * 45)

    with torch.no_grad():
        # 逐层处理
        hidden_states = query_embeds

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            # 1. 获取当前层的 rate
            current_rate = rate_manager.get_rate_for_layer(layer_idx, prev_kv_diff)

            # 2. 选择当前层需要重算的 tokens
            num_to_select = max(1, int(doc_len * current_rate))
            _, top_indices = torch.topk(base_attention_scores, num_to_select)
            selected_doc_positions = sorted(top_indices.cpu().numpy().tolist())
            selected_abs_positions = [prefix_len + pos for pos in selected_doc_positions]

            all_layer_selections.append({
                'rate': current_rate,
                'positions': selected_abs_positions,
                'num_selected': len(selected_abs_positions)
            })

            # 3. 构建当前层的 sparse 输入
            # 包含：选中的 doc tokens + query tokens
            selected_token_ids = [position_to_token[pos] for pos in selected_doc_positions]
            selected_tensor = torch.tensor([selected_token_ids], dtype=torch.long, device=device)

            # 合并 selected + query
            sparse_input = torch.cat([selected_tensor, query_tensor], dim=1)
            sparse_positions = selected_abs_positions + query_positions
            sparse_position_ids = torch.tensor([sparse_positions], dtype=torch.long, device=device)

            # 4. 计算 sparse 输入的 embedding
            sparse_embeds = model.model.embed_tokens(sparse_input)

            # 5. 通过当前层
            # LayerNorm
            normed_hidden = layer.input_layernorm(sparse_embeds)

            # Q/K/V projections
            q_states = layer.self_attn.q_proj(normed_hidden)
            k_states = layer.self_attn.k_proj(normed_hidden)
            v_states = layer.self_attn.v_proj(normed_hidden)

            sparse_len = sparse_input.shape[1]
            num_selected = len(selected_abs_positions)

            # Reshape
            q_states = q_states.view(1, sparse_len, num_heads, head_dim).transpose(1, 2)
            k_states = k_states.view(1, sparse_len, num_kv_heads, head_dim).transpose(1, 2)
            v_states = v_states.view(1, sparse_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            cos, sin = layer.self_attn.rotary_emb(v_states, sparse_position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q_states = (q_states * cos) + (rotate_half(q_states) * sin)
            k_states_rope = (k_states * cos) + (rotate_half(k_states) * sin)

            # 6. 计算 KV 差异（用于下一层的 rate 调整）
            if decay_strategy == "kv_diff":
                # 获取原始 KV
                orig_k = past_key_values.key_cache[layer_idx][:, :, selected_abs_positions, :]
                orig_v = past_key_values.value_cache[layer_idx][:, :, selected_abs_positions, :]

                # 新的 KV（只取 selected 部分，不含 query）
                new_k = k_states_rope[:, :, :num_selected, :]
                new_v = v_states[:, :, :num_selected, :]

                # 计算差异
                k_sim = F.cosine_similarity(
                    orig_k.reshape(-1, head_dim),
                    new_k.reshape(-1, head_dim),
                    dim=-1
                ).mean()
                v_sim = F.cosine_similarity(
                    orig_v.reshape(-1, head_dim),
                    new_v.reshape(-1, head_dim),
                    dim=-1
                ).mean()
                prev_kv_diff = (1.0 - (k_sim + v_sim) / 2).item()
                rate_manager.record_kv_diff(prev_kv_diff)
            else:
                prev_kv_diff = None

            # 7. 更新 KV cache
            for i, abs_pos in enumerate(selected_abs_positions):
                past_key_values.key_cache[layer_idx][:, :, abs_pos, :] = k_states_rope[:, :, i, :]
                past_key_values.value_cache[layer_idx][:, :, abs_pos, :] = v_states[:, :, i, :]

            # 打印进度
            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                kv_diff_str = f"{prev_kv_diff:.4f}" if prev_kv_diff is not None else "N/A"
                print(f"{layer_idx:<8} {current_rate:<10.2%} {num_selected:<12} {kv_diff_str:<12}")

    # 打印总结
    rate_manager.print_summary()

    # 计算总体统计
    total_recomputed = sum(s['num_selected'] for s in all_layer_selections)
    total_possible = doc_len * num_layers
    overall_ratio = total_recomputed / total_possible

    print(f"\nOverall Statistics:")
    print(f"  Total tokens recomputed: {total_recomputed}")
    print(f"  Total possible: {total_possible}")
    print(f"  Overall ratio: {overall_ratio:.2%}")
    print(f"  vs fixed {initial_rate:.0%}: saved {(initial_rate - overall_ratio) / initial_rate * 100:.1f}%")

    return all_layer_selections, rate_manager.get_summary()


def demo_decay_strategies():
    """演示不同递减策略的效果"""
    import matplotlib.pyplot as plt

    num_layers = 28
    initial_rate = 0.30
    final_rate = 0.05

    strategies = ["linear", "exponential", "cosine", "step"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in strategies:
        rates = []
        for layer_idx in range(num_layers):
            rate = compute_layer_rate_by_decay(
                layer_idx, num_layers,
                initial_rate, final_rate, strategy
            )
            rates.append(rate)

        ax.plot(range(num_layers), [r * 100 for r in rates], 'o-', label=strategy, markersize=4)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Layerwise Rate Decay Strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 35)

    plt.tight_layout()
    plt.savefig('./layerwise_decay_strategies.png', dpi=150)
    print("Saved: layerwise_decay_strategies.png")
    plt.close()

    # 打印平均 rate
    print("\nAverage rates for each strategy:")
    for strategy in strategies:
        rates = []
        for layer_idx in range(num_layers):
            rate = compute_layer_rate_by_decay(
                layer_idx, num_layers,
                initial_rate, final_rate, strategy
            )
            rates.append(rate)
        print(f"  {strategy}: {np.mean(rates):.2%}")


if __name__ == "__main__":
    demo_decay_strategies()
