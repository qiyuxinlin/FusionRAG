#!/usr/bin/env python3
"""
Attention Distillation 方案分析

目标：训练 3B 模型，使其 attention 模式接近 7B

方案：
1. 收集训练数据：用 7B 跑一遍数据集，保存 attention 分布
2. 设计 loss：让 3B 的 attention 对齐 7B 的 attention
3. 微调 3B
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)


def analyze_layer_correspondence(model_3b, model_7b):
    """
    分析 3B 和 7B 层之间的对应关系

    3B: 36 layers
    7B: 28 layers

    按比例映射: 3B layer i <-> 7B layer i * 28/36
    """
    n_3b = model_3b.config.num_hidden_layers  # 36
    n_7b = model_7b.config.num_hidden_layers  # 28

    # 按相对深度映射
    mapping = {}
    for layer_3b in range(n_3b):
        # 3B 的相对位置
        rel_pos = layer_3b / (n_3b - 1)
        # 对应的 7B 层
        layer_7b = int(rel_pos * (n_7b - 1))
        mapping[layer_3b] = layer_7b

    print("Layer Mapping (3B -> 7B):")
    print("=" * 40)
    for layer_3b, layer_7b in mapping.items():
        print(f"  3B Layer {layer_3b:2d} -> 7B Layer {layer_7b:2d}")

    return mapping


def compute_attention_distillation_loss(
    student_attn: torch.Tensor,  # [batch, num_heads, seq_len, seq_len]
    teacher_attn: torch.Tensor,  # [batch, num_heads, seq_len, seq_len]
    loss_type: str = 'kl'
) -> torch.Tensor:
    """
    计算 attention distillation loss

    Args:
        student_attn: 学生模型的 attention
        teacher_attn: 教师模型的 attention
        loss_type: 'kl' (KL divergence), 'mse', 'cosine'
    """
    if loss_type == 'kl':
        # KL divergence: KL(teacher || student)
        # 注意：attention 已经是 softmax 后的概率分布
        teacher_attn = teacher_attn.clamp(min=1e-10)
        student_attn = student_attn.clamp(min=1e-10)
        loss = F.kl_div(
            student_attn.log(),
            teacher_attn,
            reduction='batchmean'
        )
    elif loss_type == 'mse':
        loss = F.mse_loss(student_attn, teacher_attn)
    elif loss_type == 'cosine':
        # Flatten and compute cosine similarity
        s_flat = student_attn.view(student_attn.size(0), -1)
        t_flat = teacher_attn.view(teacher_attn.size(0), -1)
        loss = 1 - F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


class AttentionDistillationTrainer:
    """
    Attention Distillation 训练器

    核心流程：
    1. Forward 7B (teacher) 获取 attention
    2. Forward 3B (student) 获取 attention
    3. 计算 attention alignment loss
    4. 反向传播更新 3B
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        layer_mapping: Dict[int, int],
        learning_rate: float = 1e-5,
        attention_loss_weight: float = 1.0,
        loss_type: str = 'kl',
        device: str = 'cuda:0'
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.layer_mapping = layer_mapping
        self.attention_loss_weight = attention_loss_weight
        self.loss_type = loss_type
        self.device = device

        # 冻结 teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 只训练 student
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate
        )

        # 选择要对齐的层（后半部分层更重要）
        n_student = self.student.config.num_hidden_layers
        self.layers_to_align = list(range(n_student // 2, n_student))

    def train_step(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """
        单步训练

        Args:
            input_ids: [batch, seq_len]
        """
        input_ids = input_ids.to(self.device)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                output_attentions=True,
                use_cache=False
            )
            teacher_attentions = teacher_outputs.attentions  # tuple of [batch, heads, seq, seq]

        # Student forward (with grad)
        student_outputs = self.student(
            input_ids,
            output_attentions=True,
            use_cache=False
        )
        student_attentions = student_outputs.attentions
        student_logits = student_outputs.logits

        # Language modeling loss
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Attention alignment loss
        attn_loss = 0.0
        num_aligned_layers = 0

        for student_layer in self.layers_to_align:
            teacher_layer = self.layer_mapping[student_layer]

            student_attn = student_attentions[student_layer]  # [batch, heads, seq, seq]
            teacher_attn = teacher_attentions[teacher_layer]  # [batch, heads, seq, seq]

            # 处理 head 数量不同的情况
            # 3B: 16 heads, 7B: 28 heads -> 平均 teacher heads
            n_student_heads = student_attn.size(1)
            n_teacher_heads = teacher_attn.size(1)

            if n_student_heads != n_teacher_heads:
                # 将 teacher heads 分组平均以匹配 student
                group_size = n_teacher_heads // n_student_heads
                teacher_attn = teacher_attn.view(
                    teacher_attn.size(0),
                    n_student_heads,
                    group_size,
                    teacher_attn.size(2),
                    teacher_attn.size(3)
                ).mean(dim=2)

            layer_loss = compute_attention_distillation_loss(
                student_attn, teacher_attn, self.loss_type
            )
            attn_loss += layer_loss
            num_aligned_layers += 1

        attn_loss = attn_loss / num_aligned_layers

        # Total loss
        total_loss = lm_loss + self.attention_loss_weight * attn_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'attn_loss': attn_loss.item(),
        }


def estimate_training_cost():
    """
    估算训练成本
    """
    print("\n" + "=" * 60)
    print("Attention Distillation Training Cost Estimation")
    print("=" * 60)

    # 假设参数
    n_samples = 1000  # 训练样本数
    seq_len = 2000    # 平均序列长度
    batch_size = 1    # 由于需要存 attention，batch size 受限
    n_epochs = 3

    # 内存估算
    # 3B model: ~6GB (BF16)
    # 7B model: ~14GB (BF16)
    # Attention matrices: 36 layers * 16 heads * 2000 * 2000 * 4 bytes ≈ 9GB (每层)
    # 只存后 18 层的 attention: ~4.5GB

    print(f"\nMemory Requirements:")
    print(f"  3B Model (BF16): ~6 GB")
    print(f"  7B Model (BF16): ~14 GB")
    print(f"  Attention matrices (18 layers, seq=2000): ~4.5 GB")
    print(f"  Total: ~25 GB GPU memory")

    # 时间估算
    # 每个 sample 需要：
    # - 7B forward: ~0.5s
    # - 3B forward + backward: ~0.3s
    time_per_sample = 0.8  # seconds
    total_time = n_samples * n_epochs * time_per_sample

    print(f"\nTime Requirements:")
    print(f"  Samples: {n_samples}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Time per sample: ~{time_per_sample}s")
    print(f"  Total time: ~{total_time/3600:.1f} hours")

    print(f"\nAlternative: Offline Attention Caching")
    print(f"  1. Pre-compute 7B attention for all samples")
    print(f"  2. Save to disk (~4.5GB per sample → use compression)")
    print(f"  3. Train 3B without loading 7B")
    print(f"  Benefit: Only need ~12GB GPU memory")


def propose_simplified_approach():
    """
    提出简化方案
    """
    print("\n" + "=" * 60)
    print("Simplified Approach: Query-Document Attention Distillation")
    print("=" * 60)

    print("""
核心观察：
-----------
我们不需要对齐完整的 attention 矩阵，只需要对齐 query->document 的部分。

简化方案：
-----------
1. 对于每个训练样本 (system + docs + query)：
   - 用 7B 计算 query tokens 对 doc tokens 的 attention 分布
   - 这是一个 [query_len, doc_len] 的矩阵，而不是 [seq_len, seq_len]

2. 训练目标：
   - 让 3B 的 query->doc attention 接近 7B

3. 优势：
   - 内存大幅减少：[query_len, doc_len] << [seq_len, seq_len]
   - 更聚焦于我们关心的 attention 模式
   - 可以用更大的 batch size

数据需求：
-----------
- 使用现有的 RAG 数据集
- 或者用 7B 在新数据上标注
- 估计 ~1000-5000 samples 足够

Loss 设计：
-----------
L = L_LM + α * KL(attn_3b_q2d || attn_7b_q2d)

其中 attn_q2d 是 query tokens 对 document tokens 的平均 attention。
""")


def main():
    print("=" * 60)
    print("Attention Distillation Analysis for 3B -> 7B Alignment")
    print("=" * 60)

    # 分析层对应关系
    from transformers import AutoConfig

    config_3b = AutoConfig.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct')
    config_7b = AutoConfig.from_pretrained('/mnt/data/models/Qwen2.5-7B-Instruct')

    print(f"\n3B Model: {config_3b.num_hidden_layers} layers, {config_3b.num_attention_heads} heads")
    print(f"7B Model: {config_7b.num_hidden_layers} layers, {config_7b.num_attention_heads} heads")

    # 层映射
    n_3b = config_3b.num_hidden_layers
    n_7b = config_7b.num_hidden_layers

    print(f"\nProposed Layer Mapping (by relative depth):")
    print("-" * 40)
    for layer_3b in range(n_3b):
        rel_pos = layer_3b / (n_3b - 1)
        layer_7b = int(rel_pos * (n_7b - 1))
        if layer_3b >= n_3b // 2:  # 只显示后半部分
            print(f"  3B Layer {layer_3b:2d} ({rel_pos*100:5.1f}%) -> 7B Layer {layer_7b:2d}")

    # 估算成本
    estimate_training_cost()

    # 简化方案
    propose_simplified_approach()

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("""
推荐方案：Query-Document Attention Distillation

步骤：
1. 准备数据：收集 1000-5000 个 RAG 样本
2. 离线标注：用 7B 计算每个样本的 query->doc attention
3. 训练 3B：添加 attention alignment loss
4. 评估：在测试集上比较 token selection IoU

预计效果：
- IoU 从 ~0.57 提升到 ~0.70-0.80
- 训练时间：~2-4 小时 (单卡)
- 额外存储：~1-2 GB (cached attention)
""")


if __name__ == '__main__':
    main()
