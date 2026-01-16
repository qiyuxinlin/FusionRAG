#!/usr/bin/env python3
"""
KV Cache Calibration Tool

类似BatchNorm的思路：
1. Offline阶段：统计 preprocess KV 和 no_preprocess KV 之间的偏移分布
2. Online阶段：对 no_preprocess KV 应用偏移，避免重新计算 preprocess

Author: FusionRAG Team
Date: 2026-01-16
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CalibrationConfig:
    """KV校准配置"""
    # 统计粒度
    granularity: str = "per_layer"  # per_layer | per_head | per_position

    # 聚合方式
    aggregation: str = "mean"  # mean | mean_std | weighted

    # 采样参数
    sample_ratio: float = 0.1  # 用于统计的样本比例

    # 参考方法（用于计算偏移的preprocess方法）
    reference_method: str = "bge"

    # 校准层配置
    key_layers: List[int] = None  # None表示所有层
    value_layers: List[int] = None

    # 自适应层选择
    auto_select_layers: bool = False
    threshold: float = 0.1  # L2范数阈值

    def __post_init__(self):
        if self.key_layers is None:
            self.key_layers = []
        if self.value_layers is None:
            self.value_layers = []


class KVCalibrationStats:
    """KV校准统计量"""

    def __init__(self, config: CalibrationConfig):
        self.config = config

        # 统计量存储
        # key_stats[layer_idx] = {'mean': tensor, 'std': tensor, 'count': int}
        self.key_stats: Dict[int, Dict[str, torch.Tensor]] = {}
        self.value_stats: Dict[int, Dict[str, torch.Tensor]] = {}

        # 自适应层选择的L2范数
        self.key_layer_norms: Dict[int, float] = {}
        self.value_layer_norms: Dict[int, float] = {}

    def save(self, save_path: str):
        """保存统计量到文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {
            'config': asdict(self.config),
            'key_stats': {},
            'value_stats': {},
            'key_layer_norms': self.key_layer_norms,
            'value_layer_norms': self.value_layer_norms,
        }

        # 转换tensor为可序列化格式
        for layer_idx, stats in self.key_stats.items():
            save_dict['key_stats'][str(layer_idx)] = {
                k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in stats.items()
            }

        for layer_idx, stats in self.value_stats.items():
            save_dict['value_stats'][str(layer_idx)] = {
                k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in stats.items()
            }

        # 保存为.pt文件（包含tensor）和.json（用于查看）
        torch.save(save_dict, save_path)
        json_path = save_path.replace('.pt', '.json')

        # JSON只保存配置和范数
        json_dict = {
            'config': save_dict['config'],
            'key_layer_norms': save_dict['key_layer_norms'],
            'value_layer_norms': save_dict['value_layer_norms'],
        }
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=2)

        print(f"✓ Saved calibration stats to {save_path}")
        print(f"✓ Saved config to {json_path}")

    @classmethod
    def load(cls, load_path: str) -> 'KVCalibrationStats':
        """从文件加载统计量"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Calibration stats not found: {load_path}")

        save_dict = torch.load(load_path, map_location='cpu', weights_only=False)

        # 重建配置
        config = CalibrationConfig(**save_dict['config'])
        stats_obj = cls(config)

        # 重建统计量
        for layer_idx_str, stats in save_dict['key_stats'].items():
            layer_idx = int(layer_idx_str)
            stats_obj.key_stats[layer_idx] = {
                k: torch.tensor(v) if isinstance(v, list) else v
                for k, v in stats.items()
            }

        for layer_idx_str, stats in save_dict['value_stats'].items():
            layer_idx = int(layer_idx_str)
            stats_obj.value_stats[layer_idx] = {
                k: torch.tensor(v) if isinstance(v, list) else v
                for k, v in stats.items()
            }

        stats_obj.key_layer_norms = save_dict.get('key_layer_norms', {})
        stats_obj.value_layer_norms = save_dict.get('value_layer_norms', {})

        print(f"✓ Loaded calibration stats from {load_path}")
        return stats_obj


class KVCalibrator:
    """KV Cache校准器"""

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.stats = KVCalibrationStats(config)

    def compute_offset_offline(
        self,
        cache_dir: str,
        dataset_name: str,
        model_name: str,
        example_ids: List[int],
        num_layers: int = 28,
    ):
        """
        Offline阶段：计算KV偏移统计量

        Args:
            cache_dir: KV cache根目录
            dataset_name: 数据集名称
            model_name: 模型名称
            example_ids: 要统计的样本ID列表
            num_layers: 模型层数
        """
        print(f"\n{'='*60}")
        print(f"KV Calibration - Offline Mode")
        print(f"{'='*60}")
        print(f"Granularity: {self.config.granularity}")
        print(f"Aggregation: {self.config.aggregation}")
        print(f"Reference method: {self.config.reference_method}")
        print(f"Samples: {len(example_ids)}")
        print(f"{'='*60}\n")

        # 路径构建
        no_preprocess_dir = os.path.join(
            cache_dir, dataset_name, model_name, "no_preprocess"
        )
        reference_dir = os.path.join(
            cache_dir, dataset_name, model_name, self.config.reference_method
        )

        # 初始化累加器
        if self.config.granularity == "per_layer":
            self._init_per_layer_accumulators(num_layers)
        elif self.config.granularity == "per_head":
            # 需要先读取一个KV cache确定head数量
            self._init_per_head_accumulators(num_layers, reference_dir, example_ids[0])
        elif self.config.granularity == "per_position":
            raise NotImplementedError("per_position granularity not yet implemented")

        # 遍历样本计算偏移
        total_processed = 0
        for example_id in example_ids:
            success = self._process_example(
                example_id, no_preprocess_dir, reference_dir, num_layers
            )
            if success:
                total_processed += 1

            if (total_processed + 1) % 10 == 0:
                print(f"  Processed {total_processed}/{len(example_ids)} examples")

        print(f"\n✓ Processed {total_processed}/{len(example_ids)} examples")

        # 计算最终统计量
        self._finalize_statistics(num_layers)

        # 自适应层选择
        if self.config.auto_select_layers:
            self._auto_select_layers()

    def _init_per_layer_accumulators(self, num_layers: int):
        """初始化per-layer累加器"""
        self.key_accumulators = {layer: [] for layer in range(num_layers)}
        self.value_accumulators = {layer: [] for layer in range(num_layers)}

    def _init_per_head_accumulators(self, num_layers: int, ref_dir: str, example_id: int):
        """初始化per-head累加器（需要知道head数量）"""
        # 读取第一个样本的chunk_id=0（system）来获取shape
        key_path = os.path.join(ref_dir, f"{example_id}_0_key.pt")
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Cannot init per_head: {key_path} not found")

        sample_kv = torch.load(key_path, map_location='cpu', weights_only=True)
        # Shape: [num_layers][batch, num_heads, seq_len, head_dim]
        self.num_heads = sample_kv[0].shape[1] if sample_kv[0].dim() == 4 else 1

        # 为每一层每个head创建累加器
        self.key_accumulators = {
            layer: {head: [] for head in range(self.num_heads)}
            for layer in range(num_layers)
        }
        self.value_accumulators = {
            layer: {head: [] for head in range(self.num_heads)}
            for layer in range(num_layers)
        }

        print(f"  Detected {self.num_heads} attention heads")

    def _process_example(
        self,
        example_id: int,
        no_preprocess_dir: str,
        reference_dir: str,
        num_layers: int
    ) -> bool:
        """处理单个样本，计算偏移"""
        # 只处理chunk_id=0（system）的KV cache
        # 因为不同文档的chunk数量不同，chunk_id>0的KV cache不一定存在
        chunk_id = 0

        no_prep_key_path = os.path.join(no_preprocess_dir, f"{example_id}_{chunk_id}_key.pt")
        no_prep_val_path = os.path.join(no_preprocess_dir, f"{example_id}_{chunk_id}_value.pt")
        ref_key_path = os.path.join(reference_dir, f"{example_id}_{chunk_id}_key.pt")
        ref_val_path = os.path.join(reference_dir, f"{example_id}_{chunk_id}_value.pt")

        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [no_prep_key_path, no_prep_val_path, ref_key_path, ref_val_path]):
            return False

        # 加载KV cache
        no_prep_key = torch.load(no_prep_key_path, map_location='cpu', weights_only=True)
        no_prep_val = torch.load(no_prep_val_path, map_location='cpu', weights_only=True)
        ref_key = torch.load(ref_key_path, map_location='cpu', weights_only=True)
        ref_val = torch.load(ref_val_path, map_location='cpu', weights_only=True)

        # 计算偏移并累加
        for layer_idx in range(num_layers):
            key_offset = ref_key[layer_idx] - no_prep_key[layer_idx]
            val_offset = ref_val[layer_idx] - no_prep_val[layer_idx]

            if self.config.granularity == "per_layer":
                # 对整层求平均：[batch, num_heads, seq_len, head_dim] -> [head_dim]
                # 或 [batch, seq_len, hidden_dim] -> [hidden_dim]
                key_offset_mean = key_offset.mean(dim=tuple(range(key_offset.dim() - 1)))
                val_offset_mean = val_offset.mean(dim=tuple(range(val_offset.dim() - 1)))

                self.key_accumulators[layer_idx].append(key_offset_mean)
                self.value_accumulators[layer_idx].append(val_offset_mean)

            elif self.config.granularity == "per_head":
                # 对每个head分别统计：[batch, num_heads, seq_len, head_dim] -> [num_heads, head_dim]
                if key_offset.dim() == 4:
                    # [batch, num_heads, seq_len, head_dim] -> [num_heads, head_dim]
                    key_offset_per_head = key_offset.mean(dim=(0, 2))  # 平均掉batch和seq_len
                    val_offset_per_head = val_offset.mean(dim=(0, 2))

                    for head_idx in range(self.num_heads):
                        self.key_accumulators[layer_idx][head_idx].append(
                            key_offset_per_head[head_idx]
                        )
                        self.value_accumulators[layer_idx][head_idx].append(
                            val_offset_per_head[head_idx]
                        )
                else:
                    # Fallback to per_layer if not 4D
                    key_offset_mean = key_offset.mean(dim=tuple(range(key_offset.dim() - 1)))
                    val_offset_mean = val_offset.mean(dim=tuple(range(val_offset.dim() - 1)))
                    self.key_accumulators[layer_idx][0].append(key_offset_mean)
                    self.value_accumulators[layer_idx][0].append(val_offset_mean)

        return True

    def _finalize_statistics(self, num_layers: int):
        """计算最终统计量（均值、标准差）"""
        print("\n  Computing final statistics...")

        for layer_idx in range(num_layers):
            if self.config.granularity == "per_layer":
                # Stack所有样本的偏移
                if len(self.key_accumulators[layer_idx]) == 0:
                    continue

                key_offsets = torch.stack(self.key_accumulators[layer_idx])  # [num_samples, head_dim]
                val_offsets = torch.stack(self.value_accumulators[layer_idx])

                # 计算均值和标准差
                key_mean = key_offsets.mean(dim=0)
                key_std = key_offsets.std(dim=0)
                val_mean = val_offsets.mean(dim=0)
                val_std = val_offsets.std(dim=0)

                self.stats.key_stats[layer_idx] = {
                    'mean': key_mean,
                    'std': key_std,
                    'count': len(self.key_accumulators[layer_idx])
                }
                self.stats.value_stats[layer_idx] = {
                    'mean': val_mean,
                    'std': val_std,
                    'count': len(self.value_accumulators[layer_idx])
                }

                # 记录L2范数（用于自适应层选择）
                self.stats.key_layer_norms[layer_idx] = key_mean.norm().item()
                self.stats.value_layer_norms[layer_idx] = val_mean.norm().item()

            elif self.config.granularity == "per_head":
                # 每个head单独统计
                self.stats.key_stats[layer_idx] = {}
                self.stats.value_stats[layer_idx] = {}

                for head_idx in range(self.num_heads):
                    if len(self.key_accumulators[layer_idx][head_idx]) == 0:
                        continue

                    key_offsets = torch.stack(self.key_accumulators[layer_idx][head_idx])
                    val_offsets = torch.stack(self.value_accumulators[layer_idx][head_idx])

                    key_mean = key_offsets.mean(dim=0)
                    key_std = key_offsets.std(dim=0)
                    val_mean = val_offsets.mean(dim=0)
                    val_std = val_offsets.std(dim=0)

                    self.stats.key_stats[layer_idx][head_idx] = {
                        'mean': key_mean,
                        'std': key_std,
                        'count': len(self.key_accumulators[layer_idx][head_idx])
                    }
                    self.stats.value_stats[layer_idx][head_idx] = {
                        'mean': val_mean,
                        'std': val_std,
                        'count': len(self.value_accumulators[layer_idx][head_idx])
                    }

                # 计算整层的平均L2范数
                layer_key_norms = [
                    self.stats.key_stats[layer_idx][h]['mean'].norm().item()
                    for h in range(self.num_heads)
                    if h in self.stats.key_stats[layer_idx]
                ]
                layer_val_norms = [
                    self.stats.value_stats[layer_idx][h]['mean'].norm().item()
                    for h in range(self.num_heads)
                    if h in self.stats.value_stats[layer_idx]
                ]

                if layer_key_norms:
                    self.stats.key_layer_norms[layer_idx] = np.mean(layer_key_norms)
                if layer_val_norms:
                    self.stats.value_layer_norms[layer_idx] = np.mean(layer_val_norms)

        print("  ✓ Statistics computed")

    def _auto_select_layers(self):
        """自适应选择偏移显著的层"""
        print(f"\n  Auto-selecting layers (threshold={self.config.threshold})...")

        # 选择L2范数大于阈值的层
        selected_key_layers = [
            layer for layer, norm in self.stats.key_layer_norms.items()
            if norm > self.config.threshold
        ]
        selected_val_layers = [
            layer for layer, norm in self.stats.value_layer_norms.items()
            if norm > self.config.threshold
        ]

        self.config.key_layers = selected_key_layers
        self.config.value_layers = selected_val_layers

        print(f"    Selected {len(selected_key_layers)} key layers: {selected_key_layers}")
        print(f"    Selected {len(selected_val_layers)} value layers: {selected_val_layers}")

    def apply_calibration_online(
        self,
        kv_cache_key: List[torch.Tensor],
        kv_cache_value: List[torch.Tensor],
        device: str = 'cpu'
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Online阶段：对no_preprocess的KV cache应用校准

        Args:
            kv_cache_key: 原始Key cache列表 [num_layers][batch, num_heads, seq_len, head_dim]
            kv_cache_value: 原始Value cache列表
            device: 目标设备

        Returns:
            calibrated_key, calibrated_value: 校准后的KV cache
        """
        calibrated_key = []
        calibrated_value = []

        # 确定要校准的层
        key_layers_to_calibrate = set(self.config.key_layers) if self.config.key_layers else set(self.stats.key_stats.keys())
        val_layers_to_calibrate = set(self.config.value_layers) if self.config.value_layers else set(self.stats.value_stats.keys())

        for layer_idx in range(len(kv_cache_key)):
            k = kv_cache_key[layer_idx].to(device)
            v = kv_cache_value[layer_idx].to(device)

            # Key校准
            if layer_idx in key_layers_to_calibrate and layer_idx in self.stats.key_stats:
                if self.config.granularity == "per_layer":
                    offset = self.stats.key_stats[layer_idx]['mean'].to(device)
                    # Broadcast offset to match KV shape
                    # offset: [head_dim] -> [1, 1, 1, head_dim]
                    offset = offset.view(*([1] * (k.dim() - 1)), -1)
                    k = k + offset

                elif self.config.granularity == "per_head":
                    # 每个head应用不同的偏移
                    if k.dim() == 4:
                        for head_idx in range(k.shape[1]):
                            if head_idx in self.stats.key_stats[layer_idx]:
                                offset = self.stats.key_stats[layer_idx][head_idx]['mean'].to(device)
                                offset = offset.view(1, 1, 1, -1)  # [1, 1, 1, head_dim]
                                k[:, head_idx:head_idx+1, :, :] = k[:, head_idx:head_idx+1, :, :] + offset

            # Value校准
            if layer_idx in val_layers_to_calibrate and layer_idx in self.stats.value_stats:
                if self.config.granularity == "per_layer":
                    offset = self.stats.value_stats[layer_idx]['mean'].to(device)
                    offset = offset.view(*([1] * (v.dim() - 1)), -1)
                    v = v + offset

                elif self.config.granularity == "per_head":
                    if v.dim() == 4:
                        for head_idx in range(v.shape[1]):
                            if head_idx in self.stats.value_stats[layer_idx]:
                                offset = self.stats.value_stats[layer_idx][head_idx]['mean'].to(device)
                                offset = offset.view(1, 1, 1, -1)
                                v[:, head_idx:head_idx+1, :, :] = v[:, head_idx:head_idx+1, :, :] + offset

            calibrated_key.append(k)
            calibrated_value.append(v)

        return calibrated_key, calibrated_value


def print_stats_summary(stats: KVCalibrationStats):
    """打印统计量摘要"""
    print(f"\n{'='*60}")
    print(f"Calibration Statistics Summary")
    print(f"{'='*60}")
    print(f"Granularity: {stats.config.granularity}")
    print(f"Reference method: {stats.config.reference_method}")
    print(f"\nKey Layer Norms (L2):")
    for layer, norm in sorted(stats.key_layer_norms.items()):
        print(f"  Layer {layer:2d}: {norm:.6f}")

    print(f"\nValue Layer Norms (L2):")
    for layer, norm in sorted(stats.value_layer_norms.items()):
        print(f"  Layer {layer:2d}: {norm:.6f}")

    if stats.config.key_layers:
        print(f"\nSelected Key Layers: {stats.config.key_layers}")
    if stats.config.value_layers:
        print(f"Selected Value Layers: {stats.config.value_layers}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='KV Cache Calibration Tool')
    parser.add_argument('--mode', type=str, required=True, choices=['offline', 'summary'],
                        help='offline: compute and save stats | summary: print stats summary')
    parser.add_argument('--cache_dir', type=str, default='/mnt/data3/tmp/fusionrag',
                        help='KV cache root directory')
    parser.add_argument('--dataset', type=str, default='musique',
                        help='Dataset name')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct',
                        help='Model name')
    parser.add_argument('--num_layers', type=int, default=28,
                        help='Number of model layers')

    # Offline mode parameters
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                        help='Sample ratio for offline statistics')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples (None=auto from ratio)')
    parser.add_argument('--reference_method', type=str, default='bge',
                        help='Reference preprocess method')
    parser.add_argument('--granularity', type=str, default='per_layer',
                        choices=['per_layer', 'per_head', 'per_position'],
                        help='Statistics granularity')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'mean_std', 'weighted'],
                        help='Aggregation method')
    parser.add_argument('--auto_select_layers', type=lambda x: x.lower() == 'true', default=False,
                        help='Auto select significant layers')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold for auto layer selection')

    # Output path
    parser.add_argument('--stats_path', type=str, default=None,
                        help='Path to save/load calibration stats')

    args = parser.parse_args()

    # 确定stats保存路径
    if args.stats_path is None:
        stats_filename = f"calibration_stats_{args.reference_method}_{args.granularity}.pt"
        args.stats_path = os.path.join(
            args.cache_dir, args.dataset, args.model_name, stats_filename
        )

    if args.mode == 'offline':
        # 创建配置
        config = CalibrationConfig(
            granularity=args.granularity,
            aggregation=args.aggregation,
            sample_ratio=args.sample_ratio,
            reference_method=args.reference_method,
            auto_select_layers=args.auto_select_layers,
            threshold=args.threshold,
        )

        # 确定要统计的样本ID
        # 这里简单起见，假设样本ID从0开始连续
        # 实际使用时可能需要从数据集读取
        if args.max_samples is None:
            # 根据ratio自动计算（假设总共500个样本）
            total_samples = 500
            num_samples = int(total_samples * args.sample_ratio)
        else:
            num_samples = args.max_samples

        example_ids = list(range(num_samples))

        # 创建calibrator并运行offline统计
        calibrator = KVCalibrator(config)
        calibrator.compute_offset_offline(
            cache_dir=args.cache_dir,
            dataset_name=args.dataset,
            model_name=args.model_name,
            example_ids=example_ids,
            num_layers=args.num_layers,
        )

        # 保存统计量
        calibrator.stats.save(args.stats_path)

        # 打印摘要
        print_stats_summary(calibrator.stats)

    elif args.mode == 'summary':
        # 加载并打印统计量摘要
        stats = KVCalibrationStats.load(args.stats_path)
        print_stats_summary(stats)
