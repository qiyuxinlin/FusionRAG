#!/usr/bin/env python3
"""
注意力可视化工具

功能：
1. 对指定样本进行推理，hook attention weights
2. 分析模型在生成时关注了哪些输入token
3. 对比不同召回方法下的注意力分布差异
4. 生成热图可视化

使用方法：
    python visualize_attention.py --sample_id <id> --method <bge|random|no_prep>
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

#####################################################################
# 配置区域
#####################################################################

# 模型配置
MODEL_PATH = "/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_TYPE = "qwen"

# Cache 路径
BGE_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge"
RANDOM_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_random"
NO_PREP_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache"

# 数据路径
DATA_PATH = "./data/result_reflect.json"

# 输出目录
OUTPUT_DIR = "/home/shm/document/exp/FusionRAG/attention_analysis_output"

# 设备
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#####################################################################
# Attention Hook
#####################################################################

class AttentionHook:
    """
    Hook工具类，用于捕获attention weights
    """
    def __init__(self):
        self.attention_weights = {}
        self.hooks = []

    def hook_fn(self, module, input, output, layer_name):
        """
        Hook函数，捕获attention weights
        """
        # output的结构取决于模型
        # 通常: (hidden_states, present_key_value, attention_weights) 或类似
        if isinstance(output, tuple) and len(output) >= 3:
            # 第3个元素通常是 attention_weights
            attn_weights = output[2] if output[2] is not None else output[-1]
            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                # detach并移到CPU以节省GPU内存
                self.attention_weights[layer_name] = attn_weights.detach().cpu()

    def register_hooks(self, model):
        """
        在所有attention层注册hooks
        """
        for name, module in model.named_modules():
            # 根据模型类型调整layer名称匹配
            if 'attn' in name.lower() or 'attention' in name.lower():
                if 'self_attn' in name or 'self_attention' in name:
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: self.hook_fn(m, i, o, n)
                    )
                    self.hooks.append(hook)

    def remove_hooks(self):
        """
        移除所有hooks
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_attention_weights(self):
        """
        获取捕获的attention weights
        """
        return self.attention_weights


#####################################################################
# 加载模型和数据
#####################################################################

def load_model(model_path: str, device: str = "cuda:0"):
    """
    加载模型和tokenizer
    """
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "eager"  # 使用eager模式以便hook attention

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        output_attentions=True  # 重要：输出attention weights
    )
    model.eval()

    print(f"✓ 模型加载完成，设备: {device}")
    return model, tokenizer


def load_kv_cache(cache_dir: str, example_id: str, chunk_id: int, device: str = 'cpu'):
    """
    加载KV cache
    """
    key_path = os.path.join(cache_dir, f"{example_id}_{chunk_id}_key.pt")
    value_path = os.path.join(cache_dir, f"{example_id}_{chunk_id}_value.pt")

    if not os.path.exists(key_path):
        return None

    key_cache = torch.load(key_path, map_location=device)
    value_cache = torch.load(value_path, map_location=device)

    return (key_cache, value_cache)


#####################################################################
# 注意力分析
#####################################################################

def generate_with_attention_tracking(model, tokenizer, input_text: str,
                                    past_key_values=None, max_new_tokens: int = 50):
    """
    生成文本并跟踪attention weights
    """
    # Tokenize输入
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']

    # 设置attention hook
    hook_manager = AttentionHook()
    hook_manager.register_hooks(model)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True
        )

    # 获取attention weights
    attention_weights = hook_manager.get_attention_weights()

    # 移除hooks
    hook_manager.remove_hooks()

    # 解码生成的文本
    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        'generated_text': generated_text,
        'generated_ids': generated_ids,
        'input_ids': input_ids,
        'attention_weights': attention_weights,
    }


def analyze_attention_pattern(attention_weights: Dict[str, torch.Tensor],
                             input_tokens: List[str],
                             generated_tokens: List[str],
                             doc_ranges: List[Tuple[int, int, str]]) -> Dict:
    """
    分析注意力模式

    Args:
        attention_weights: {layer_name: attn_tensor} 字典
        input_tokens: 输入token列表
        generated_tokens: 生成token列表
        doc_ranges: [(start_idx, end_idx, doc_name), ...] 文档范围

    Returns:
        分析结果字典
    """
    results = {
        'layer_stats': {},
        'doc_attention_distribution': {},
        'token_importance': {},
    }

    for layer_name, attn_tensor in attention_weights.items():
        # attn_tensor shape: (batch, num_heads, seq_len, seq_len) 或类似
        # 我们关注最后几个生成token对输入的注意力

        # 平均所有head
        if len(attn_tensor.shape) == 4:
            attn_mean = attn_tensor.mean(dim=1)  # (batch, seq_len, seq_len)
        else:
            attn_mean = attn_tensor

        # 取最后生成的token（最后几行）
        num_generated = len(generated_tokens)
        if attn_mean.shape[-2] > num_generated:
            last_token_attn = attn_mean[0, -num_generated:, :]  # (num_generated, total_seq_len)
        else:
            last_token_attn = attn_mean[0]

        # 计算每个文档区域获得的平均注意力
        doc_attention = {}
        for start, end, doc_name in doc_ranges:
            doc_attn = last_token_attn[:, start:end].mean().item()
            doc_attention[doc_name] = doc_attn

        results['doc_attention_distribution'][layer_name] = doc_attention

        # 统计信息
        results['layer_stats'][layer_name] = {
            'mean_attention': last_token_attn.mean().item(),
            'std_attention': last_token_attn.std().item(),
            'max_attention': last_token_attn.max().item(),
        }

    return results


#####################################################################
# 可视化
#####################################################################

def visualize_attention_heatmap(attention_weights: Dict[str, torch.Tensor],
                               input_tokens: List[str],
                               generated_tokens: List[str],
                               output_path: str,
                               title: str = "Attention Heatmap"):
    """
    生成attention热图
    """
    # 选择一个代表性的层（例如中间层）
    layer_names = sorted(attention_weights.keys())
    if len(layer_names) == 0:
        print("⚠ 没有捕获到attention weights")
        return

    # 选择几个关键层进行可视化
    num_layers = len(layer_names)
    selected_layers = [
        layer_names[0],  # 第一层
        layer_names[num_layers // 2],  # 中间层
        layer_names[-1],  # 最后一层
    ] if num_layers >= 3 else layer_names

    fig, axes = plt.subplots(1, len(selected_layers), figsize=(6*len(selected_layers), 8))
    if len(selected_layers) == 1:
        axes = [axes]

    for idx, layer_name in enumerate(selected_layers):
        attn_tensor = attention_weights[layer_name]

        # 平均所有heads
        if len(attn_tensor.shape) == 4:
            attn_mean = attn_tensor.mean(dim=1)[0]  # (seq_len, seq_len)
        else:
            attn_mean = attn_tensor[0]

        # 截取最后几个token（生成的部分）对输入的注意力
        num_show = min(20, len(generated_tokens))
        attn_to_plot = attn_mean[-num_show:, :].cpu().numpy()

        # 绘制热图
        ax = axes[idx]
        im = ax.imshow(attn_to_plot, cmap='viridis', aspect='auto')

        # 设置标签
        ax.set_xlabel('Input Tokens')
        ax.set_ylabel('Generated Tokens')
        ax.set_title(f'{layer_name}\n(Layer {idx})')

        # 添加colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')

        # 设置x轴标签（输入tokens）
        if len(input_tokens) < 50:
            ax.set_xticks(range(len(input_tokens)))
            ax.set_xticklabels(input_tokens, rotation=90, fontsize=6)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 注意力热图已保存: {output_path}")


def visualize_doc_attention_distribution(doc_attention_results: Dict,
                                        output_path: str,
                                        title: str = "Document Attention Distribution"):
    """
    可视化不同文档区域获得的注意力分布
    """
    # doc_attention_results: {layer_name: {doc_name: attention_score}}

    layer_names = sorted(doc_attention_results.keys())
    if len(layer_names) == 0:
        return

    # 提取文档名称
    doc_names = list(doc_attention_results[layer_names[0]].keys())

    # 准备数据
    data = []
    for doc_name in doc_names:
        scores = [doc_attention_results[layer][doc_name] for layer in layer_names]
        data.append(scores)

    data = np.array(data)  # (num_docs, num_layers)

    # 绘制热图
    fig, ax = plt.subplots(figsize=(max(10, len(layer_names) * 0.5), max(6, len(doc_names) * 0.5)))

    im = sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd',
                    xticklabels=[f"L{i}" for i in range(len(layer_names))],
                    yticklabels=doc_names,
                    cbar_kws={'label': 'Attention Score'},
                    ax=ax)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Document/Region')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 文档注意力分布图已保存: {output_path}")


def compare_attention_across_methods(sample_id: int,
                                     methods: List[str] = ['bge', 'random', 'no_prep'],
                                     output_dir: str = OUTPUT_DIR):
    """
    对比不同方法在同一样本上的注意力分布

    这是一个简化的框架函数，实际使用需要完整的推理流程
    """
    print(f"\n对比样本 #{sample_id} 在不同方法下的注意力分布...")

    # TODO: 这里需要完整的推理流程
    # 1. 加载模型
    # 2. 加载数据样本
    # 3. 对每个方法加载对应的KV cache
    # 4. 使用KV cache进行生成并记录attention
    # 5. 对比分析

    print("⚠ 完整的推理流程需要在test_fusionrag_reflect.py中集成")
    print("  当前脚本提供了核心的hook和可视化功能")


#####################################################################
# 主函数
#####################################################################

def main():
    parser = argparse.ArgumentParser(description='Attention Visualization Tool')
    parser.add_argument('--sample_id', type=int, required=True,
                       help='样本ID')
    parser.add_argument('--method', type=str, default='bge',
                       choices=['bge', 'random', 'no_prep', 'repeat_self'],
                       help='使用的召回方法')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='输出目录')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                       help='模型路径')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='数据路径')

    args = parser.parse_args()

    print("="*80)
    print("注意力可视化工具")
    print("="*80)
    print(f"样本ID: {args.sample_id}")
    print(f"方法: {args.method}")
    print(f"输出目录: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    model, tokenizer = load_model(args.model_path, DEVICE)

    # 加载数据
    print(f"\n加载数据: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.sample_id >= len(data):
        print(f"✗ 样本ID {args.sample_id} 超出范围 (max: {len(data)-1})")
        return

    sample = data[args.sample_id]
    print(f"\n样本信息:")
    print(f"  问题: {sample.get('question', 'N/A')}")
    print(f"  答案: {sample.get('answer', 'N/A')}")

    # 构建输入文本（这里简化处理，实际需要根据你的prompt模板）
    question = sample.get('question', '')
    docs = sample.get('docs', [])

    input_text = f"Question: {question}\n\nDocuments:\n"
    for i, doc in enumerate(docs[:3]):  # 只取前3个文档
        input_text += f"[{i+1}] {doc}\n\n"
    input_text += "Answer:"

    print(f"\n生成答案并跟踪注意力...")

    # 生成并跟踪attention
    result = generate_with_attention_tracking(
        model, tokenizer, input_text,
        past_key_values=None,
        max_new_tokens=50
    )

    print(f"\n生成的答案: {result['generated_text']}")

    # Token化
    input_tokens = tokenizer.convert_ids_to_tokens(result['input_ids'][0])
    generated_tokens = tokenizer.convert_ids_to_tokens(result['generated_ids'])

    print(f"\n输入tokens: {len(input_tokens)}")
    print(f"生成tokens: {len(generated_tokens)}")
    print(f"捕获的attention层数: {len(result['attention_weights'])}")

    # 定义文档范围（简化处理）
    doc_ranges = [
        (0, 10, "Question"),
        (10, 50, "Doc1"),
        (50, 90, "Doc2"),
        (90, 130, "Doc3"),
    ]

    # 分析attention模式
    analysis = analyze_attention_pattern(
        result['attention_weights'],
        input_tokens,
        generated_tokens,
        doc_ranges
    )

    # 可视化
    heatmap_path = os.path.join(args.output_dir, f'sample_{args.sample_id}_{args.method}_attention_heatmap.png')
    visualize_attention_heatmap(
        result['attention_weights'],
        input_tokens,
        generated_tokens,
        heatmap_path,
        title=f'Attention Heatmap - Sample #{args.sample_id} ({args.method.upper()})'
    )

    doc_dist_path = os.path.join(args.output_dir, f'sample_{args.sample_id}_{args.method}_doc_attention.png')
    visualize_doc_attention_distribution(
        analysis['doc_attention_distribution'],
        doc_dist_path,
        title=f'Document Attention Distribution - Sample #{args.sample_id} ({args.method.upper()})'
    )

    # 保存分析结果
    report = {
        'sample_id': args.sample_id,
        'method': args.method,
        'question': question,
        'generated_answer': result['generated_text'],
        'analysis': {
            'layer_stats': analysis['layer_stats'],
            'doc_attention': analysis['doc_attention_distribution'],
        }
    }

    report_path = os.path.join(args.output_dir, f'sample_{args.sample_id}_{args.method}_attention_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 分析报告已保存: {report_path}")
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
