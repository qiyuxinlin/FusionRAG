#!/usr/bin/env python3
"""
Multi-Method PCA Visualization of KV Cache Features

This script compares KV cache feature distributions across multiple methods
(no_preprocess, bge, random, repeat_self, fixed_doc, etc.) using PCA.

Usage:
    # Compare 2 methods
    python visualize_kv_pca_multi.py --sample_ids 0 1 2 \
        --methods no_preprocess bge

    # Compare 3 methods
    python visualize_kv_pca_multi.py --sample_ids 0 1 \
        --methods no_preprocess bge random

    # Compare all methods
    python visualize_kv_pca_multi.py --sample_ids 0 1 \
        --methods no_preprocess bge random repeat_self fixed_doc random_docs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json


# Method name to directory mapping
METHOD_DIR_MAP = {
    'no_preprocess': 'kv_cache',
    'bge': 'preprocess_kv_cache_global_topk10_bge',
    'random': 'preprocess_kv_cache_global_topk10_random',
    'repeat_self': 'preprocess_kv_cache_global_topk10_repeat_self',
    'fixed_doc': 'preprocess_kv_cache_global_topk10_fixed_doc',
    'random_docs': 'preprocess_kv_cache_global_topk10_random_docs',
    'random_text': 'preprocess_kv_cache_global_topk10_random_text',
    'bge_shuffled': 'preprocess_kv_cache_global_topk10_bge_shuffled',
    'repeat_self2': 'preprocess_kv_cache_global_topk1_repeat_self',
    # Note: 'steering' method is handled specially (no_preprocess + steering vectors)
}

# Color scheme for different methods (up to 10 methods)
METHOD_COLORS = {
    'no_preprocess': "#211fb4",  # Blue
    'bge': '#d62728',             # Red
    'random': '#2ca02c',          # Green
    'repeat_self': '#9467bd',     # Purple
    'fixed_doc': '#ff7f0e',       # Orange
    'random_docs': '#8c564b',     # Brown
    'random_text': '#e377c2',     # Pink
    'bge_shuffled': '#7f7f7f',    # Gray
    'steering': "#037678",        # Cyan (for steering method)
    'repeat_self2': '#bcbd22',    # Yellow-green
}


class MultiMethodKVCacheAnalyzer:
    """Analyzer for comparing KV Cache across multiple methods using PCA"""

    def __init__(self, cache_dir: str, dataset: str = "musique",
                 model_name: str = "Qwen2.5-7B-Instruct", methods: List[str] = None,
                 steering_path: str = None, steering_alpha: float = 1.0,
                 steering_key_layers: str = 'all', steering_value_layers: str = 'all',
                 use_per_head_steering: bool = False):
        self.cache_dir = Path(cache_dir)
        self.dataset = dataset
        self.model_name = model_name

        # Steering configuration
        self.steering_path = steering_path
        self.steering_alpha = steering_alpha
        self.steering_key_layers = steering_key_layers
        self.steering_value_layers = steering_value_layers
        self.use_per_head_steering = use_per_head_steering
        self.steering_stats = None

        # Default to comparing no_preprocess and bge
        if methods is None:
            methods = ['no_preprocess', 'bge']

        self.methods = methods
        self.method_paths = {}

        # Load steering vectors if 'steering' method is used
        if 'steering' in methods:
            if steering_path is None:
                raise ValueError("--steering_path must be provided when using 'steering' method")
            self.steering_stats = self._load_steering_vectors(steering_path)
            print(f"Loaded steering vectors from: {steering_path}")
            print(f"  Alpha: {steering_alpha}")
            print(f"  Key layers: {steering_key_layers}")
            print(f"  Value layers: {steering_value_layers}")
            print(f"  Per-head steering: {use_per_head_steering}")

        # Build paths for each method
        for method in methods:
            # Special handling for 'steering' method
            if method == 'steering':
                # Use no_preprocess path as base
                dir_name = METHOD_DIR_MAP['no_preprocess']
                path = self.cache_dir / model_name / dataset / dir_name
            else:
                if method not in METHOD_DIR_MAP:
                    raise ValueError(f"Unknown method: {method}. Valid methods: {list(METHOD_DIR_MAP.keys()) + ['steering']}")

                dir_name = METHOD_DIR_MAP[method]
                path = self.cache_dir / model_name / dataset / dir_name

            if not path.exists():
                raise FileNotFoundError(f"Method '{method}' KV cache not found at: {path}")

            self.method_paths[method] = path
            print(f"Method '{method}': {path}")

    def _load_steering_vectors(self, stats_path: str) -> Dict:
        """Load steering vectors from file"""
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Steering vectors file not found: {stats_path}")

        stats = torch.load(stats_path, map_location='cpu')

        # Verify format
        if 'key_steering' not in stats or 'value_steering' not in stats:
            raise ValueError("Invalid steering vectors file format")

        return stats

    def _parse_layer_selection(self, layer_spec: str, max_layers: int = 28) -> set:
        """Parse layer selection specification into a set of layer indices"""
        if layer_spec.lower() == 'all':
            return set(range(max_layers))

        layers = set()
        parts = layer_spec.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start_str, end_str = part.split('-')
                start_idx, end_idx = int(start_str), int(end_str)
                layers.update(range(start_idx, end_idx + 1))
            else:
                layers.add(int(part))
        return layers

    def _apply_steering_vector(self, kv_cache: torch.Tensor, steering_vector: torch.Tensor,
                               layer_idx: int, alpha: float = 1.0) -> torch.Tensor:
        """Apply steering vector to KV cache (layer-level)"""
        # kv_cache: [num_heads, seq_len, head_dim]
        # steering_vector: [num_heads, head_dim]
        steering_expanded = steering_vector.unsqueeze(1)  # [num_heads, 1, head_dim]
        kv_steered = kv_cache + alpha * steering_expanded
        return kv_steered

    def _apply_per_head_steering_vector(self, kv_cache: torch.Tensor, per_head_steering_dict: dict,
                                        layer_idx: int, alpha: float = 1.0) -> torch.Tensor:
        """Apply per-head steering vectors to KV cache"""
        original_shape = kv_cache.shape
        is_grouped = (kv_cache.dim() == 4)

        # Handle grouped attention format
        if is_grouped:
            num_groups, num_heads_per_group, seq_len, head_dim = kv_cache.shape
            kv_cache = kv_cache.reshape(num_groups * num_heads_per_group, seq_len, head_dim)

        num_heads, seq_len, head_dim = kv_cache.shape
        kv_steered = kv_cache.clone()

        # Apply steering vector to each head individually
        for head_idx in range(num_heads):
            head_key = f'head_{head_idx}'
            if head_key in per_head_steering_dict:
                steering_vec = per_head_steering_dict[head_key]['steering_vector']
                steering_vec = steering_vec.to(kv_cache.device).to(kv_cache.dtype)
                steering_expanded = steering_vec.unsqueeze(0)  # [1, head_dim]
                kv_steered[head_idx] = kv_cache[head_idx] + alpha * steering_expanded

        # Reshape back to original format if needed
        if is_grouped:
            kv_steered = kv_steered.reshape(original_shape)

        return kv_steered

    def load_kv_cache(self, method: str, example_id: int, chunk_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load key and value cache for a specific method, example and chunk"""
        path = self.method_paths[method]
        key_file = path / f"{example_id}_{chunk_id}_key.pt"
        value_file = path / f"{example_id}_{chunk_id}_value.pt"

        if not key_file.exists() or not value_file.exists():
            raise FileNotFoundError(f"KV cache not found for {method}: {key_file}")

        key_cache = torch.load(key_file, weights_only=True, map_location='cpu')
        value_cache = torch.load(value_file, weights_only=True, map_location='cpu')

        # Apply steering vectors if method is 'steering'
        if method == 'steering' and self.steering_stats is not None:
            key_cache = self._apply_steering_to_cache(
                key_cache, 'key', max_layers=key_cache.shape[0]
            )
            value_cache = self._apply_steering_to_cache(
                value_cache, 'value', max_layers=value_cache.shape[0]
            )

        return key_cache, value_cache

    def _apply_steering_to_cache(self, kv_cache: torch.Tensor, cache_type: str, max_layers: int) -> torch.Tensor:
        """Apply steering vectors to entire KV cache (all layers)"""
        # Determine which layers to apply steering
        if cache_type == 'key':
            layer_selection = self._parse_layer_selection(self.steering_key_layers, max_layers)
            steering_dict = self.steering_stats['key_steering']
        else:  # value
            layer_selection = self._parse_layer_selection(self.steering_value_layers, max_layers)
            steering_dict = self.steering_stats['value_steering']

        # Clone cache to avoid modifying original
        steered_cache = kv_cache.clone()

        # Apply steering per layer
        for layer_idx in range(max_layers):
            if layer_idx not in layer_selection:
                continue

            layer_name = f'layer_{layer_idx}'
            if layer_name not in steering_dict:
                continue

            layer_stats = steering_dict[layer_name]

            # Extract layer from cache
            layer_cache = steered_cache[layer_idx]

            # Handle grouped attention format
            original_shape = layer_cache.shape
            is_grouped = (layer_cache.dim() == 4)

            if is_grouped:
                # [num_groups, num_heads_per_group, seq_len, head_dim]
                num_groups, num_heads_per_group, seq_len, head_dim = layer_cache.shape
                layer_cache = layer_cache.reshape(num_groups * num_heads_per_group, seq_len, head_dim)

            # Apply steering (per-head or layer-level)
            if self.use_per_head_steering and 'per_head' in layer_stats:
                layer_cache = self._apply_per_head_steering_vector(
                    layer_cache.reshape(original_shape) if is_grouped else layer_cache,
                    layer_stats['per_head'],
                    layer_idx,
                    self.steering_alpha
                )
            else:
                # Layer-level steering
                steering_vec = layer_stats['steering_vector']
                layer_cache = self._apply_steering_vector(
                    layer_cache,
                    steering_vec,
                    layer_idx,
                    self.steering_alpha
                )

            # Reshape back if needed
            if is_grouped and not self.use_per_head_steering:
                layer_cache = layer_cache.reshape(original_shape)

            # Update steered cache
            steered_cache[layer_idx] = layer_cache

        return steered_cache

    def extract_layer_features(self, kv_cache: torch.Tensor, layer_idx: int,
                               max_tokens: int = None) -> np.ndarray:
        """Extract features from a specific layer"""
        layer_cache = kv_cache[layer_idx]
        layer_cache = layer_cache.squeeze(0)

        if layer_cache.dim() == 3:
            num_heads, seq_len, head_dim = layer_cache.shape
            layer_cache = layer_cache.permute(1, 0, 2).reshape(seq_len, num_heads * head_dim)
        elif layer_cache.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected layer_cache shape: {layer_cache.shape}")

        if max_tokens is not None and layer_cache.shape[0] > max_tokens:
            indices = torch.randperm(layer_cache.shape[0])[:max_tokens]
            layer_cache = layer_cache[indices]

        if layer_cache.dtype == torch.bfloat16:
            layer_cache = layer_cache.float()

        return layer_cache.numpy()

    def compute_pca(self, features: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """Compute PCA on features"""
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(features)
        return transformed, pca

    def analyze_single_sample(self, example_id: int, chunk_id: int,
                            selected_layers: List[int], max_tokens: int = 500):
        """Analyze KV cache for a single sample across multiple methods"""
        print(f"\n{'='*80}")
        print(f"Analyzing Example {example_id}, Chunk {chunk_id}")
        print(f"Methods: {', '.join(self.methods)}")
        print(f"{'='*80}\n")

        # Load KV caches for all methods
        method_kvs = {}
        for method in self.methods:
            try:
                key, value = self.load_kv_cache(method, example_id, chunk_id)
                method_kvs[method] = {'key': key, 'value': value}
                print(f"Loaded {method}: Key {key.shape}, Value {value.shape}")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

        if len(method_kvs) == 0:
            raise FileNotFoundError(f"No valid KV caches found for example {example_id}")

        if len(method_kvs) == 1:
            print(f"\n⚠️  Warning: Only 1 method loaded successfully for example {example_id}. Skipping comparison.")
            return None

        results = {
            'example_id': example_id,
            'chunk_id': chunk_id,
            'methods': list(method_kvs.keys()),
            'layers': {},
        }

        # Use first successfully loaded method as reference for L2 distance
        reference_method = list(method_kvs.keys())[0]
        print(f"\n✓ Using '{reference_method}' as reference method for L2 distance calculation")

        if len(method_kvs) < len(self.methods):
            missing = set(self.methods) - set(method_kvs.keys())
            print(f"⚠️  Note: {len(missing)} method(s) skipped due to missing KV cache: {', '.join(missing)}")

        for layer_idx in selected_layers:
            print(f"\n--- Processing Layer {layer_idx} ---")

            # Extract features for all methods
            layer_features = {}
            for method in method_kvs.keys():
                key_feat = self.extract_layer_features(method_kvs[method]['key'], layer_idx, max_tokens)
                val_feat = self.extract_layer_features(method_kvs[method]['value'], layer_idx, max_tokens)
                layer_features[method] = {'key': key_feat, 'val': val_feat}

            # Compute L2 distances (all methods vs reference method)
            l2_distances = {}
            for method in method_kvs.keys():
                if method == reference_method:
                    l2_distances[method] = {'key': 0.0, 'val': 0.0}
                else:
                    key_l2 = np.linalg.norm(
                        layer_features[reference_method]['key'] - layer_features[method]['key'],
                        axis=1
                    ).mean()
                    val_l2 = np.linalg.norm(
                        layer_features[reference_method]['val'] - layer_features[method]['val'],
                        axis=1
                    ).mean()
                    l2_distances[method] = {'key': float(key_l2), 'val': float(val_l2)}
                    print(f"L2 dist ({reference_method} vs {method}): Key={key_l2:.4f}, Value={val_l2:.4f}")

            # Combine all features for joint PCA
            all_key_features = np.vstack([layer_features[m]['key'] for m in method_kvs.keys()])
            all_val_features = np.vstack([layer_features[m]['val'] for m in method_kvs.keys()])

            # Compute PCA
            key_pca, key_pca_model = self.compute_pca(all_key_features, n_components=2)
            val_pca, val_pca_model = self.compute_pca(all_val_features, n_components=2)

            # Split back to individual methods
            n_samples = layer_features[list(method_kvs.keys())[0]]['key'].shape[0]
            key_pca_split = {}
            val_pca_split = {}

            start_idx = 0
            for method in method_kvs.keys():
                end_idx = start_idx + n_samples
                key_pca_split[method] = key_pca[start_idx:end_idx]
                val_pca_split[method] = val_pca[start_idx:end_idx]
                start_idx = end_idx

            # Store results
            results['layers'][layer_idx] = {
                'l2_distances': l2_distances,
                'key_pca': key_pca_split,
                'val_pca': val_pca_split,
                'key_variance_explained': key_pca_model.explained_variance_ratio_.tolist(),
                'val_variance_explained': val_pca_model.explained_variance_ratio_.tolist(),
            }

        return results

    def plot_layer_comparison(self, results: Dict, output_dir: str):
        """Plot PCA visualization for all layers and methods"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        example_id = results['example_id']
        chunk_id = results['chunk_id']
        methods = results['methods']
        layers = sorted(results['layers'].keys())

        n_layers = len(layers)
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        # Plot Keys
        fig_key, axes_key = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes_key = np.array([[axes_key]])
        elif n_rows == 1:
            axes_key = axes_key.reshape(1, -1)
        elif n_cols == 1:
            axes_key = axes_key.reshape(-1, 1)

        # Plot Values
        fig_val, axes_val = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes_val = np.array([[axes_val]])
        elif n_rows == 1:
            axes_val = axes_val.reshape(1, -1)
        elif n_cols == 1:
            axes_val = axes_val.reshape(-1, 1)

        for idx, layer_idx in enumerate(layers):
            row = idx // n_cols
            col = idx % n_cols

            layer_data = results['layers'][layer_idx]

            # Plot Keys
            ax_key = axes_key[row, col]
            for method in methods:
                pca_data = layer_data['key_pca'][method]
                color = METHOD_COLORS.get(method, 'gray')
                ax_key.scatter(pca_data[:, 0], pca_data[:, 1],
                              alpha=0.5, s=20, c=color, label=method)

            var_exp = layer_data['key_variance_explained']
            ax_key.set_xlabel(f'PC1 ({var_exp[0]:.2%})')
            ax_key.set_ylabel(f'PC2 ({var_exp[1]:.2%})')

            # Add L2 distances to title
            l2_info = []
            for method in methods:
                if method != methods[0]:  # Skip reference method
                    l2_val = layer_data['l2_distances'][method]['key']
                    l2_info.append(f"{method}:{l2_val:.2f}")
            l2_str = ", ".join(l2_info) if l2_info else "reference"

            ax_key.set_title(f'Layer {layer_idx} - Key\nL2 vs {methods[0]}: {l2_str}', fontsize=9)
            ax_key.legend(fontsize=7, loc='best')
            ax_key.grid(True, alpha=0.3)

            # Plot Values
            ax_val = axes_val[row, col]
            for method in methods:
                pca_data = layer_data['val_pca'][method]
                color = METHOD_COLORS.get(method, 'gray')
                ax_val.scatter(pca_data[:, 0], pca_data[:, 1],
                              alpha=0.5, s=20, c=color, label=method)

            var_exp = layer_data['val_variance_explained']
            ax_val.set_xlabel(f'PC1 ({var_exp[0]:.2%})')
            ax_val.set_ylabel(f'PC2 ({var_exp[1]:.2%})')

            # Add L2 distances to title
            l2_info = []
            for method in methods:
                if method != methods[0]:
                    l2_val = layer_data['l2_distances'][method]['val']
                    l2_info.append(f"{method}:{l2_val:.2f}")
            l2_str = ", ".join(l2_info) if l2_info else "reference"

            ax_val.set_title(f'Layer {layer_idx} - Value\nL2 vs {methods[0]}: {l2_str}', fontsize=9)
            ax_val.legend(fontsize=7, loc='best')
            ax_val.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_layers, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes_key[row, col].axis('off')
            axes_val[row, col].axis('off')

        methods_str = "_".join(methods)
        fig_key.suptitle(f'Key Cache PCA - Example {example_id}, Chunk {chunk_id}\nMethods: {", ".join(methods)}',
                         fontsize=14, y=0.998)
        fig_val.suptitle(f'Value Cache PCA - Example {example_id}, Chunk {chunk_id}\nMethods: {", ".join(methods)}',
                         fontsize=14, y=0.998)

        fig_key.tight_layout()
        fig_val.tight_layout()

        # Save figures
        key_path = output_path / f"pca_key_example{example_id}_chunk{chunk_id}_{methods_str}.png"
        val_path = output_path / f"pca_value_example{example_id}_chunk{chunk_id}_{methods_str}.png"

        fig_key.savefig(key_path, dpi=150, bbox_inches='tight')
        fig_val.savefig(val_path, dpi=150, bbox_inches='tight')

        print(f"\nSaved Key PCA plot to: {key_path}")
        print(f"Saved Value PCA plot to: {val_path}")

        plt.close(fig_key)
        plt.close(fig_val)

    def plot_l2_distance_trends(self, all_results: List[Dict], output_dir: str):
        """Plot L2 distance trends across layers for multiple samples and methods"""
        output_path = Path(output_dir)

        methods = all_results[0]['methods']
        reference_method = methods[0]

        # Create subplots for each non-reference method
        n_methods = len(methods) - 1  # Exclude reference
        if n_methods == 0:
            return

        fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(2, 1)

        for method_idx, method in enumerate([m for m in methods if m != reference_method]):
            ax_key = axes[0, method_idx]
            ax_val = axes[1, method_idx]

            for results in all_results:
                example_id = results['example_id']
                chunk_id = results['chunk_id']

                layers = sorted(results['layers'].keys())
                key_dists = [results['layers'][l]['l2_distances'][method]['key'] for l in layers]
                val_dists = [results['layers'][l]['l2_distances'][method]['val'] for l in layers]

                label = f"Ex{example_id}_Ch{chunk_id}"
                ax_key.plot(layers, key_dists, marker='o', label=label, linewidth=2)
                ax_val.plot(layers, val_dists, marker='s', label=label, linewidth=2)

            ax_key.set_xlabel('Layer Index', fontsize=12)
            ax_key.set_ylabel('Mean L2 Distance', fontsize=12)
            ax_key.set_title(f'Key: {reference_method} vs {method}', fontsize=13)
            ax_key.legend(fontsize=8)
            ax_key.grid(True, alpha=0.3)

            ax_val.set_xlabel('Layer Index', fontsize=12)
            ax_val.set_ylabel('Mean L2 Distance', fontsize=12)
            ax_val.set_title(f'Value: {reference_method} vs {method}', fontsize=13)
            ax_val.legend(fontsize=8)
            ax_val.grid(True, alpha=0.3)

        fig.tight_layout()

        methods_str = "_".join(methods)
        trend_path = output_path / f"l2_distance_trends_{methods_str}.png"
        fig.savefig(trend_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved L2 distance trends to: {trend_path}")

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Multi-Method KV Cache PCA Visualization')
    parser.add_argument('--cache_dir', type=str, default='/mnt/data3/tmp/fusionrag',
                       help='Base directory for KV cache')
    parser.add_argument('--dataset', type=str, default='musique',
                       help='Dataset name')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct',
                       help='Model name')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['no_preprocess', 'bge'],
                       help='Methods to compare. Valid: no_preprocess, bge, random, repeat_self, fixed_doc, random_docs, random_text')
    parser.add_argument('--sample_ids', type=int, nargs='+', default=[0],
                       help='Example IDs to analyze')
    parser.add_argument('--chunk_id', type=int, default=1,
                       help='Chunk ID to analyze (1 for first document, 0 for system)')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Specific layers to analyze (default: evenly spaced 6 layers)')
    parser.add_argument('--max_layers', type=int, default=28,
                       help='Total number of layers in the model')
    parser.add_argument('--max_tokens', type=int, default=500,
                       help='Maximum tokens to sample per layer')
    parser.add_argument('--output_dir', type=str, default='./kv_pca_multi_analysis',
                       help='Output directory for plots')

    # Steering-related arguments
    parser.add_argument('--steering_path', type=str, default=None,
                       help='Path to steering vectors file (.pt) - required when using "steering" method')
    parser.add_argument('--steering_alpha', type=float, default=1.0,
                       help='Steering vector strength coefficient (default: 1.0)')
    parser.add_argument('--steering_key_layers', type=str, default='all',
                       help='Layers to apply key steering (e.g., "all", "0-10", "0,5,10")')
    parser.add_argument('--steering_value_layers', type=str, default='all',
                       help='Layers to apply value steering (e.g., "all", "0-10", "0,5,10")')
    parser.add_argument('--use_per_head_steering', action='store_true',
                       help='Use per-head steering vectors (if available in steering file)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Multi-Method KV Cache PCA Analysis")
    print(f"{'='*80}")
    print(f"Methods to compare: {', '.join(args.methods)}")
    print(f"Available methods: {', '.join(list(METHOD_DIR_MAP.keys()) + ['steering'])}")
    print(f"{'='*80}\n")

    # Initialize analyzer
    analyzer = MultiMethodKVCacheAnalyzer(
        args.cache_dir, args.dataset, args.model_name, args.methods,
        steering_path=args.steering_path,
        steering_alpha=args.steering_alpha,
        steering_key_layers=args.steering_key_layers,
        steering_value_layers=args.steering_value_layers,
        use_per_head_steering=args.use_per_head_steering
    )

    # Select layers to analyze
    if args.layers is None:
        selected_layers = [0, args.max_layers // 5, 2 * args.max_layers // 5,
                          3 * args.max_layers // 5, 4 * args.max_layers // 5, args.max_layers - 1]
    else:
        selected_layers = args.layers

    print(f"Analyzing layers: {selected_layers}\n")

    # Analyze each sample
    all_results = []
    for example_id in args.sample_ids:
        try:
            results = analyzer.analyze_single_sample(
                example_id, args.chunk_id, selected_layers, args.max_tokens
            )

            # Skip if only 1 method was loaded (no comparison possible)
            if results is None:
                continue

            all_results.append(results)

            # Plot for this sample
            analyzer.plot_layer_comparison(results, args.output_dir)

        except FileNotFoundError as e:
            print(f"Warning: Skipping example {example_id} - {e}")
            continue

    if len(all_results) > 0:
        # Plot aggregated trends
        analyzer.plot_l2_distance_trends(all_results, args.output_dir)

        # Save summary statistics
        summary_path = Path(args.output_dir) / "summary_statistics.json"
        with open(summary_path, 'w') as f:
            json_results = []
            for r in all_results:
                json_r = {
                    'example_id': r['example_id'],
                    'chunk_id': r['chunk_id'],
                    'methods': r['methods'],
                    'layers': {}
                }
                for layer_idx, layer_data in r['layers'].items():
                    json_r['layers'][layer_idx] = {
                        'l2_distances': layer_data['l2_distances'],
                        'key_variance_explained': layer_data['key_variance_explained'],
                        'val_variance_explained': layer_data['val_variance_explained'],
                    }
                json_results.append(json_r)

            json.dump(json_results, f, indent=2)

        print(f"\nSaved summary statistics to: {summary_path}")
        print(f"\nAnalysis complete! All plots saved to: {args.output_dir}")
    else:
        print("\nNo valid samples found for analysis.")


if __name__ == "__main__":
    main()
