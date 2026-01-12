#!/usr/bin/env python3
"""
基于 3B Attention 不确定性的动态重算比例方法

核心发现：
- 当 3B 的 attention 更集中（peak 更高）时，反而更容易答错
- 这可能是"过度自信"的表现：3B 错误地集中关注某个位置
- 策略：当检测到这种模式时，提高重算比例

特征：
1. peak_strength: attention 峰值强度（越高越可能答错）
2. layer_consistency: 层间一致性（越低越可能答错）
3. coverage_80_ratio: 覆盖 80% attention 所需的 token 比例（越低越可能答错）
"""

import torch
import numpy as np
from typing import Tuple, Dict


def compute_uncertainty_features(
    attn_weights: torch.Tensor,
    doc_start: int,
    doc_end: int
) -> Dict[str, float]:
    """
    计算 3B 模型的 attention 不确定性特征

    Args:
        attn_weights: [num_layers, num_heads, seq_len, seq_len]
        doc_start: 文档开始位置
        doc_end: 文档结束位置

    Returns:
        dict: 不确定性特征
    """
    num_layers = attn_weights.shape[0]
    query_start = doc_end  # query 在 doc 之后

    # 提取 query -> doc 的 attention
    q2d_attn = attn_weights[:, :, query_start:, doc_start:doc_end]

    if q2d_attn.shape[2] == 0 or q2d_attn.shape[3] == 0:
        return {'peak_strength': 0.0, 'layer_consistency': 1.0, 'coverage_80_ratio': 1.0}

    # 对 heads 和 query positions 取平均
    layer_attn = q2d_attn.mean(dim=(1, 2))  # [num_layers, doc_len]
    avg_attn = layer_attn.mean(dim=0)  # [doc_len]

    # 1. Peak strength
    peak_strength = avg_attn.max().item()

    # 2. Coverage 80% ratio
    sorted_attn, _ = torch.sort(avg_attn, descending=True)
    cumsum = torch.cumsum(sorted_attn, dim=0)
    total = cumsum[-1]
    coverage_80 = (cumsum >= 0.8 * total).nonzero(as_tuple=True)[0]
    if len(coverage_80) > 0:
        tokens_for_80 = coverage_80[0].item() + 1
    else:
        tokens_for_80 = len(avg_attn)
    coverage_80_ratio = tokens_for_80 / len(avg_attn)

    # 3. Layer consistency
    top_k = min(50, layer_attn.shape[-1])
    layer_top_tokens = []
    for i in range(num_layers):
        top_indices = torch.topk(layer_attn[i], top_k).indices
        layer_top_tokens.append(set(top_indices.tolist()))

    layer_consistency = []
    for i in range(num_layers - 1):
        intersection = len(layer_top_tokens[i] & layer_top_tokens[i+1])
        union = len(layer_top_tokens[i] | layer_top_tokens[i+1])
        if union > 0:
            layer_consistency.append(intersection / union)

    layer_consistency_score = np.mean(layer_consistency) if layer_consistency else 0.5

    return {
        'peak_strength': peak_strength,
        'layer_consistency': layer_consistency_score,
        'coverage_80_ratio': coverage_80_ratio
    }


def compute_dynamic_rate(
    features: Dict[str, float],
    base_rate: float = 0.2,
    min_rate: float = 0.15,
    max_rate: float = 0.35
) -> Tuple[float, str]:
    """
    基于不确定性特征计算动态重算比例

    策略：
    - 当 peak_strength 较高 → 3B 可能"过度自信"在错误位置 → 提高 rate
    - 当 layer_consistency 较低 → 3B 不确定 → 提高 rate
    - 当 coverage_80_ratio 较低 → attention 过于集中 → 提高 rate

    Args:
        features: 不确定性特征
        base_rate: 基准重算比例
        min_rate: 最小重算比例
        max_rate: 最大重算比例

    Returns:
        rate: 动态计算的重算比例
        reason: 选择该比例的原因
    """
    peak = features['peak_strength']
    consistency = features['layer_consistency']
    coverage = features['coverage_80_ratio']

    # 阈值（基于分析结果）
    # Correct cases: peak=0.0060, consistency=0.392, coverage=0.399
    # Wrong cases:   peak=0.0063, consistency=0.382, coverage=0.386

    peak_threshold = 0.0061  # median
    consistency_threshold = 0.387
    coverage_threshold = 0.393

    # 计算 uncertainty score (0-1)
    # 越高表示越可能需要更多 tokens
    uncertainty = 0.0
    reasons = []

    # Peak strength contribution
    if peak > peak_threshold:
        peak_factor = min((peak - peak_threshold) / 0.001, 1.0)  # normalize
        uncertainty += 0.4 * peak_factor
        reasons.append(f"high_peak({peak:.4f})")

    # Layer consistency contribution (反向)
    if consistency < consistency_threshold:
        consistency_factor = min((consistency_threshold - consistency) / 0.05, 1.0)
        uncertainty += 0.3 * consistency_factor
        reasons.append(f"low_consistency({consistency:.3f})")

    # Coverage contribution (反向)
    if coverage < coverage_threshold:
        coverage_factor = min((coverage_threshold - coverage) / 0.05, 1.0)
        uncertainty += 0.3 * coverage_factor
        reasons.append(f"low_coverage({coverage:.3f})")

    # Map uncertainty to rate
    rate = base_rate + uncertainty * (max_rate - base_rate)
    rate = max(min_rate, min(max_rate, rate))

    reason = ", ".join(reasons) if reasons else "normal"

    return rate, reason


class DynamicRateSelector:
    """
    动态重算比例选择器

    使用 3B 模型的 attention 特征来决定每个问题的重算比例
    """

    def __init__(
        self,
        base_rate: float = 0.2,
        min_rate: float = 0.15,
        max_rate: float = 0.35,
        verbose: bool = False
    ):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.verbose = verbose

        # Statistics
        self.rate_history = []
        self.feature_history = []

    def get_rate(
        self,
        draft_attention: torch.Tensor,
        doc_start: int,
        doc_end: int
    ) -> float:
        """
        根据 3B draft model 的 attention 计算动态 rate

        Args:
            draft_attention: [num_layers, num_heads, seq_len, seq_len]
            doc_start: 文档开始位置
            doc_end: 文档结束位置

        Returns:
            rate: 动态计算的重算比例
        """
        features = compute_uncertainty_features(draft_attention, doc_start, doc_end)
        rate, reason = compute_dynamic_rate(
            features,
            self.base_rate,
            self.min_rate,
            self.max_rate
        )

        self.rate_history.append(rate)
        self.feature_history.append(features)

        if self.verbose:
            print(f"  Dynamic rate: {rate:.3f} ({reason})")

        return rate

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.rate_history:
            return {}

        rates = np.array(self.rate_history)
        return {
            'mean_rate': np.mean(rates),
            'std_rate': np.std(rates),
            'min_rate': np.min(rates),
            'max_rate': np.max(rates),
            'low_rate_count': np.sum(rates < 0.2),
            'high_rate_count': np.sum(rates > 0.25),
            'total_count': len(rates)
        }


def test_on_critical_cases():
    """在 critical cases 上测试动态 rate"""
    import json
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from test_fusionrag_reflect import prepare_reflect_data

    device = "cuda:0"

    # Load results
    with open('/mnt/data/wjh/FusionRAG/test_3b_direct_results.json', 'r') as f:
        test_results = json.load(f)

    print(f"Testing dynamic rate on {len(test_results)} critical cases\n")

    # Load 3B model
    print("Loading 3B model...")
    model = AutoModelForCausalLM.from_pretrained(
        '/mnt/data/models/Qwen2.5-3B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct', trust_remote_code=True)

    # Load dataset
    print("Loading dataset...")
    questions_data, system_tensor, _, _ = prepare_reflect_data(
        './data/result_reflect.json',
        tokenizer,
        '/mnt/data/models/bge-m3-FP16',
        'qwen',
        topk=10,
        max_main_questions=200,
        preprocess=False
    )
    system_len = system_tensor.shape[0]

    # Initialize dynamic rate selector
    selector = DynamicRateSelector(base_rate=0.2, min_rate=0.15, max_rate=0.35, verbose=True)

    print("\n" + "=" * 80)
    print("Testing Dynamic Rate Selection")
    print("=" * 80)

    for result in test_results:
        question = result['question']
        is_correct = result['is_correct_direct']

        # Find question in dataset
        for q_data in questions_data:
            for sub_q_info in q_data['sub_questions']:
                if sub_q_info['query'] == question:
                    # Build input
                    doc_chunk_ids = sub_q_info['chunk_ids']
                    doc_tensors = q_data['doc_tensors']
                    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

                    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
                    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

                    all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
                    full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

                    doc_start = system_len
                    doc_end = system_len + sum(t.shape[0] for t in sub_q_doc_tensors)

                    # Get attention
                    with torch.no_grad():
                        outputs = model(full_input, output_attentions=True, use_cache=False)
                        attentions = outputs.attentions

                    attn_weights = torch.stack([a.squeeze(0) for a in attentions])

                    # Get dynamic rate
                    print(f"\n{'✓' if is_correct else '✗'} {question[:60]}...")
                    rate = selector.get_rate(attn_weights, doc_start, doc_end)
                    break
            else:
                continue
            break

    # Print statistics
    print("\n" + "=" * 80)
    print("DYNAMIC RATE STATISTICS")
    print("=" * 80)

    stats = selector.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Analyze by correctness
    correct_rates = [r for r, f in zip(selector.rate_history, test_results) if f['is_correct_direct']]
    wrong_rates = [r for r, f in zip(selector.rate_history, test_results) if not f['is_correct_direct']]

    print(f"\n  Rate for correct cases: {np.mean(correct_rates):.4f} ± {np.std(correct_rates):.4f}")
    print(f"  Rate for wrong cases:   {np.mean(wrong_rates):.4f} ± {np.std(wrong_rates):.4f}")

    # Expected behavior: wrong cases should get higher rates
    if np.mean(wrong_rates) > np.mean(correct_rates):
        print("\n  ✓ Dynamic rate correctly assigns higher rates to harder cases!")
    else:
        print("\n  ✗ Dynamic rate needs adjustment - not assigning higher rates to harder cases")


if __name__ == '__main__':
    test_on_critical_cases()
