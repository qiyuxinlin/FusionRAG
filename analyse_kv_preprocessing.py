#!/usr/bin/env python3
"""
分析不同KV预处理方式的影响

用于对比和可视化不同预处理方法对KV cache的影响:
1. 统计KV cache的分布特征 (均值、方差、范数等)
2. 计算不同预处理方法之间的KV差异
3. 分析KV在不同层的变化模式
4. 可视化KV分布和差异

这可以帮助理解:
- 不同召回方法(BGE, Random, Repeat等)如何影响KV cache
- 预处理是否改变了KV的分布特性
- 哪些层受预处理影响最大
- KV的语义结构是否被保留
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd


class KVCacheAnalyzer:
    """KV Cache分析器"""

    def __init__(self, kv_cache_dir: str):
        """
        初始化分析器

        Args:
            kv_cache_dir: KV cache目录
        """
        self.kv_cache_dir = kv_cache_dir

    def load_kv_cache(self, cache_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载KV cache

        Args:
            cache_id: cache标识符 (如 "doc_0")

        Returns:
            (key_cache, value_cache): key和value的cache张量
        """
        key_path = os.path.join(self.kv_cache_dir, f"{cache_id}_key.pt")
        value_path = os.path.join(self.kv_cache_dir, f"{cache_id}_value.pt")

        if not os.path.exists(key_path) or not os.path.exists(value_path):
            raise FileNotFoundError(f"KV cache not found for {cache_id}")

        key_cache = torch.load(key_path, weights_only=True, map_location='cpu')
        value_cache = torch.load(value_path, weights_only=True, map_location='cpu')

        return key_cache, value_cache

    def compute_kv_statistics(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor
    ) -> Dict[str, Any]:
        """
        计算KV cache的统计特征

        Args:
            key_cache: key cache [num_layers, batch, num_heads, seq_len, head_dim]
            value_cache: value cache

        Returns:
            statistics: 统计信息字典
        """
        num_layers = len(key_cache)
        stats = {
            'num_layers': num_layers,
            'key_stats': {},
            'value_stats': {}
        }

        for layer_idx in range(num_layers):
            key = key_cache[layer_idx]
            value = value_cache[layer_idx]

            # 计算统计量
            key_stats = {
                'mean': key.mean().item(),
                'std': key.std().item(),
                'min': key.min().item(),
                'max': key.max().item(),
                'norm': torch.norm(key).item(),
                'l1_norm': torch.norm(key, p=1).item(),
                'l2_norm': torch.norm(key, p=2).item(),
                'shape': list(key.shape)
            }

            value_stats = {
                'mean': value.mean().item(),
                'std': value.std().item(),
                'min': value.min().item(),
                'max': value.max().item(),
                'norm': torch.norm(value).item(),
                'l1_norm': torch.norm(value, p=1).item(),
                'l2_norm': torch.norm(value, p=2).item(),
                'shape': list(value.shape)
            }

            stats['key_stats'][f'layer_{layer_idx}'] = key_stats
            stats['value_stats'][f'layer_{layer_idx}'] = value_stats

        return stats

    def compute_kv_difference(
        self,
        kv1: Tuple[torch.Tensor, torch.Tensor],
        kv2: Tuple[torch.Tensor, torch.Tensor],
        metric: str = 'l2'
    ) -> Dict[str, Any]:
        """
        计算两个KV cache之间的差异

        Args:
            kv1: (key1, value1)
            kv2: (key2, value2)
            metric: 差异度量 ('l2', 'l1', 'cosine', 'mse')

        Returns:
            difference: 差异统计
        """
        key1, value1 = kv1
        key2, value2 = kv2

        num_layers = len(key1)
        diff_stats = {
            'metric': metric,
            'key_diff': {},
            'value_diff': {}
        }

        for layer_idx in range(num_layers):
            k1 = key1[layer_idx].float()
            k2 = key2[layer_idx].float()
            v1 = value1[layer_idx].float()
            v2 = value2[layer_idx].float()

            # 计算差异
            if metric == 'l2':
                key_diff = torch.norm(k1 - k2, p=2).item()
                value_diff = torch.norm(v1 - v2, p=2).item()
                key_relative_diff = key_diff / (torch.norm(k1, p=2).item() + 1e-8)
                value_relative_diff = value_diff / (torch.norm(v1, p=2).item() + 1e-8)

            elif metric == 'l1':
                key_diff = torch.norm(k1 - k2, p=1).item()
                value_diff = torch.norm(v1 - v2, p=1).item()
                key_relative_diff = key_diff / (torch.norm(k1, p=1).item() + 1e-8)
                value_relative_diff = value_diff / (torch.norm(v1, p=1).item() + 1e-8)

            elif metric == 'cosine':
                k1_flat = k1.flatten()
                k2_flat = k2.flatten()
                v1_flat = v1.flatten()
                v2_flat = v2.flatten()

                key_diff = 1.0 - torch.nn.functional.cosine_similarity(
                    k1_flat.unsqueeze(0), k2_flat.unsqueeze(0)
                ).item()
                value_diff = 1.0 - torch.nn.functional.cosine_similarity(
                    v1_flat.unsqueeze(0), v2_flat.unsqueeze(0)
                ).item()
                key_relative_diff = key_diff
                value_relative_diff = value_diff

            elif metric == 'mse':
                key_diff = torch.nn.functional.mse_loss(k1, k2).item()
                value_diff = torch.nn.functional.mse_loss(v1, v2).item()
                key_relative_diff = key_diff / (k1.var().item() + 1e-8)
                value_relative_diff = value_diff / (v1.var().item() + 1e-8)

            else:
                raise ValueError(f"Unknown metric: {metric}")

            diff_stats['key_diff'][f'layer_{layer_idx}'] = {
                'absolute': key_diff,
                'relative': key_relative_diff
            }
            diff_stats['value_diff'][f'layer_{layer_idx}'] = {
                'absolute': value_diff,
                'relative': value_relative_diff
            }

        return diff_stats

    def analyze_layer_wise_changes(
        self,
        no_preprocess_kv: Tuple[torch.Tensor, torch.Tensor],
        preprocess_kv: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        分析每层的变化模式

        Args:
            no_preprocess_kv: 未预处理的KV
            preprocess_kv: 预处理后的KV

        Returns:
            analysis: 分层分析结果
        """
        key_no_prep, value_no_prep = no_preprocess_kv
        key_prep, value_prep = preprocess_kv

        num_layers = len(key_no_prep)
        analysis = {
            'layer_changes': []
        }

        for layer_idx in range(num_layers):
            k_before = key_no_prep[layer_idx].float()
            k_after = key_prep[layer_idx].float()
            v_before = value_no_prep[layer_idx].float()
            v_after = value_prep[layer_idx].float()

            # 计算多种变化指标
            layer_change = {
                'layer': layer_idx,
                'key': {
                    'mean_shift': (k_after.mean() - k_before.mean()).item(),
                    'std_change': (k_after.std() - k_before.std()).item(),
                    'norm_ratio': (torch.norm(k_after) / torch.norm(k_before)).item(),
                    'correlation': torch.corrcoef(
                        torch.stack([k_before.flatten(), k_after.flatten()])
                    )[0, 1].item()
                },
                'value': {
                    'mean_shift': (v_after.mean() - v_before.mean()).item(),
                    'std_change': (v_after.std() - v_before.std()).item(),
                    'norm_ratio': (torch.norm(v_after) / torch.norm(v_before)).item(),
                    'correlation': torch.corrcoef(
                        torch.stack([v_before.flatten(), v_after.flatten()])
                    )[0, 1].item()
                }
            }

            analysis['layer_changes'].append(layer_change)

        return analysis


class KVComparisonAnalyzer:
    """对比多种预处理方法的分析器"""

    def __init__(self, base_dir: str):
        """
        初始化对比分析器

        Args:
            base_dir: 包含多个预处理方法结果的基础目录
        """
        self.base_dir = base_dir
        self.analyzers = {}

    def add_method(self, method_name: str, kv_cache_dir: str):
        """添加一个预处理方法的结果"""
        self.analyzers[method_name] = KVCacheAnalyzer(kv_cache_dir)

    def compare_all_methods(
        self,
        cache_id: str,
        reference_method: str = "no_preprocess"
    ) -> Dict[str, Any]:
        """
        对比所有方法相对于参考方法的差异

        Args:
            cache_id: cache标识符
            reference_method: 参考方法名称

        Returns:
            comparison: 对比结果
        """
        if reference_method not in self.analyzers:
            raise ValueError(f"Reference method '{reference_method}' not found")

        # 加载参考方法的KV
        ref_analyzer = self.analyzers[reference_method]
        ref_kv = ref_analyzer.load_kv_cache(cache_id)

        comparison = {
            'cache_id': cache_id,
            'reference_method': reference_method,
            'methods': {}
        }

        # 对比每个方法
        for method_name, analyzer in self.analyzers.items():
            if method_name == reference_method:
                continue

            try:
                method_kv = analyzer.load_kv_cache(cache_id)

                # 计算统计信息
                stats = analyzer.compute_kv_statistics(method_kv[0], method_kv[1])

                # 计算与参考方法的差异
                diff_l2 = analyzer.compute_kv_difference(ref_kv, method_kv, metric='l2')
                diff_cosine = analyzer.compute_kv_difference(ref_kv, method_kv, metric='cosine')

                # 分层分析
                layer_analysis = analyzer.analyze_layer_wise_changes(ref_kv, method_kv)

                comparison['methods'][method_name] = {
                    'statistics': stats,
                    'difference_l2': diff_l2,
                    'difference_cosine': diff_cosine,
                    'layer_analysis': layer_analysis
                }

            except FileNotFoundError:
                print(f"Warning: Cache not found for method '{method_name}', cache_id '{cache_id}'")
                continue

        return comparison

    def visualize_comparison(
        self,
        comparison: Dict[str, Any],
        output_dir: str
    ):
        """
        可视化对比结果

        Args:
            comparison: 对比结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 绘制层级差异热图
        self._plot_layer_difference_heatmap(comparison, output_dir)

        # 2. 绘制统计量对比
        self._plot_statistics_comparison(comparison, output_dir)

        # 3. 绘制层级变化曲线
        self._plot_layer_changes(comparison, output_dir)

        # 4. 保存数值结果到CSV
        self._save_to_csv(comparison, output_dir)

    def _plot_layer_difference_heatmap(self, comparison: Dict, output_dir: str):
        """绘制层级差异热图"""
        methods = list(comparison['methods'].keys())
        if not methods:
            return

        # 提取每层的相对差异
        num_layers = comparison['methods'][methods[0]]['statistics']['num_layers']

        key_diffs = np.zeros((len(methods), num_layers))
        value_diffs = np.zeros((len(methods), num_layers))

        for i, method in enumerate(methods):
            for layer_idx in range(num_layers):
                layer_name = f'layer_{layer_idx}'
                key_diffs[i, layer_idx] = comparison['methods'][method]['difference_l2']['key_diff'][layer_name]['relative']
                value_diffs[i, layer_idx] = comparison['methods'][method]['difference_l2']['value_diff'][layer_name]['relative']

        # 绘制Key差异热图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        sns.heatmap(key_diffs, ax=ax1, cmap='YlOrRd', xticklabels=range(num_layers),
                   yticklabels=methods, cbar_kws={'label': 'Relative L2 Difference'})
        ax1.set_title('Key Cache: Layer-wise Relative Difference')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Method')

        sns.heatmap(value_diffs, ax=ax2, cmap='YlOrRd', xticklabels=range(num_layers),
                   yticklabels=methods, cbar_kws={'label': 'Relative L2 Difference'})
        ax2.set_title('Value Cache: Layer-wise Relative Difference')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Method')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_difference_heatmap.png'), dpi=300)
        plt.close()

    def _plot_statistics_comparison(self, comparison: Dict, output_dir: str):
        """绘制统计量对比"""
        methods = list(comparison['methods'].keys())
        if not methods:
            return

        num_layers = comparison['methods'][methods[0]]['statistics']['num_layers']

        # 提取统计量
        stats_to_plot = ['mean', 'std', 'l2_norm']
        fig, axes = plt.subplots(2, len(stats_to_plot), figsize=(15, 8))

        for stat_idx, stat_name in enumerate(stats_to_plot):
            # Key统计量
            ax_key = axes[0, stat_idx]
            for method in methods:
                values = []
                for layer_idx in range(num_layers):
                    layer_name = f'layer_{layer_idx}'
                    values.append(comparison['methods'][method]['statistics']['key_stats'][layer_name][stat_name])
                ax_key.plot(range(num_layers), values, marker='o', label=method, alpha=0.7)

            ax_key.set_title(f'Key: {stat_name}')
            ax_key.set_xlabel('Layer')
            ax_key.set_ylabel(stat_name)
            ax_key.legend()
            ax_key.grid(True, alpha=0.3)

            # Value统计量
            ax_value = axes[1, stat_idx]
            for method in methods:
                values = []
                for layer_idx in range(num_layers):
                    layer_name = f'layer_{layer_idx}'
                    values.append(comparison['methods'][method]['statistics']['value_stats'][layer_name][stat_name])
                ax_value.plot(range(num_layers), values, marker='o', label=method, alpha=0.7)

            ax_value.set_title(f'Value: {stat_name}')
            ax_value.set_xlabel('Layer')
            ax_value.set_ylabel(stat_name)
            ax_value.legend()
            ax_value.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'statistics_comparison.png'), dpi=300)
        plt.close()

    def _plot_layer_changes(self, comparison: Dict, output_dir: str):
        """绘制层级变化曲线"""
        methods = list(comparison['methods'].keys())
        if not methods:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ('mean_shift', 'Mean Shift'),
            ('std_change', 'Std Change'),
            ('norm_ratio', 'Norm Ratio'),
            ('correlation', 'Correlation')
        ]

        for metric_idx, (metric_name, metric_label) in enumerate(metrics):
            ax_key = axes[metric_idx // 2, (metric_idx % 2) * 2]
            ax_value = axes[metric_idx // 2, (metric_idx % 2) * 2 + 1]

            for method in methods:
                layer_changes = comparison['methods'][method]['layer_analysis']['layer_changes']

                key_values = [lc['key'][metric_name] for lc in layer_changes]
                value_values = [lc['value'][metric_name] for lc in layer_changes]

                layers = [lc['layer'] for lc in layer_changes]

                ax_key.plot(layers, key_values, marker='o', label=method, alpha=0.7)
                ax_value.plot(layers, value_values, marker='o', label=method, alpha=0.7)

            ax_key.set_title(f'Key: {metric_label}')
            ax_key.set_xlabel('Layer')
            ax_key.set_ylabel(metric_label)
            ax_key.legend()
            ax_key.grid(True, alpha=0.3)

            ax_value.set_title(f'Value: {metric_label}')
            ax_value.set_xlabel('Layer')
            ax_value.set_ylabel(metric_label)
            ax_value.legend()
            ax_value.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_changes.png'), dpi=300)
        plt.close()

    def _save_to_csv(self, comparison: Dict, output_dir: str):
        """保存数值结果到CSV"""
        methods = list(comparison['methods'].keys())
        if not methods:
            return

        # 1. 层级差异CSV
        diff_data = []
        for method in methods:
            for layer_name, diff_info in comparison['methods'][method]['difference_l2']['key_diff'].items():
                layer_idx = int(layer_name.split('_')[1])
                diff_data.append({
                    'method': method,
                    'layer': layer_idx,
                    'cache_type': 'key',
                    'absolute_diff': diff_info['absolute'],
                    'relative_diff': diff_info['relative']
                })

            for layer_name, diff_info in comparison['methods'][method]['difference_l2']['value_diff'].items():
                layer_idx = int(layer_name.split('_')[1])
                diff_data.append({
                    'method': method,
                    'layer': layer_idx,
                    'cache_type': 'value',
                    'absolute_diff': diff_info['absolute'],
                    'relative_diff': diff_info['relative']
                })

        df_diff = pd.DataFrame(diff_data)
        df_diff.to_csv(os.path.join(output_dir, 'layer_differences.csv'), index=False)

        # 2. 统计量CSV
        stats_data = []
        for method in methods:
            for layer_name, stats in comparison['methods'][method]['statistics']['key_stats'].items():
                layer_idx = int(layer_name.split('_')[1])
                stats_data.append({
                    'method': method,
                    'layer': layer_idx,
                    'cache_type': 'key',
                    **stats
                })

            for layer_name, stats in comparison['methods'][method]['statistics']['value_stats'].items():
                layer_idx = int(layer_name.split('_')[1])
                stats_data.append({
                    'method': method,
                    'layer': layer_idx,
                    'cache_type': 'value',
                    **stats
                })

        df_stats = pd.DataFrame(stats_data)
        df_stats.to_csv(os.path.join(output_dir, 'statistics.csv'), index=False)

        print(f"CSV files saved to {output_dir}")


def main():
    """主函数 - 使用示例"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze KV preprocessing methods")
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing preprocessing results')
    parser.add_argument('--cache_id', type=str, default='doc_0',
                       help='Cache ID to analyze')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['no_preprocess', 'preprocess_topk3_bge', 'preprocess_topk3_random'],
                       help='Methods to compare')
    parser.add_argument('--output_dir', type=str, default='./kv_analysis_results',
                       help='Output directory for analysis results')

    args = parser.parse_args()

    # 创建对比分析器
    analyzer = KVComparisonAnalyzer(args.base_dir)

    # 添加各个方法
    for method in args.methods:
        method_dir = os.path.join(args.base_dir, method)
        if os.path.exists(method_dir):
            analyzer.add_method(method, method_dir)
            print(f"Added method: {method}")
        else:
            print(f"Warning: Method directory not found: {method_dir}")

    # 执行对比分析
    print(f"\nAnalyzing cache: {args.cache_id}")
    comparison = analyzer.compare_all_methods(
        cache_id=args.cache_id,
        reference_method='no_preprocess'
    )

    # 可视化结果
    print(f"\nGenerating visualizations...")
    analyzer.visualize_comparison(comparison, args.output_dir)

    # 保存完整的JSON结果
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nAnalysis completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
