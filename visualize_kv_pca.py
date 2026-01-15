#!/usr/bin/env python3
"""
PCA Visualization of KV Cache Features Across Different Layers and Methods

This script analyzes and visualizes the feature distribution of Key and Value vectors
from KV cache across different layers, comparing no_preprocess and BGE modes.

Usage:
    python visualize_kv_pca.py --sample_ids 0 1 2 --max_layers 28
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json


class KVCacheAnalyzer:
    """Analyzer for KV Cache feature distribution using PCA"""

    def __init__(self, cache_dir: str, dataset: str = "musique", model_name: str = "Qwen2.5-7B-Instruct"):
        self.cache_dir = Path(cache_dir)
        self.dataset = dataset
        self.model_name = model_name

        # Paths
        self.no_preprocess_path = self.cache_dir / model_name / dataset / "kv_cache"
        self.bge_preprocess_path = self.cache_dir / model_name / dataset / "preprocess_kv_cache_global_topk10_bge"

        print(f"No Preprocess Path: {self.no_preprocess_path}")
        print(f"BGE Preprocess Path: {self.bge_preprocess_path}")

    def load_kv_cache(self, path: Path, example_id: int, chunk_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load key and value cache for a specific example and chunk"""
        key_file = path / f"{example_id}_{chunk_id}_key.pt"
        value_file = path / f"{example_id}_{chunk_id}_value.pt"

        if not key_file.exists() or not value_file.exists():
            raise FileNotFoundError(f"KV cache not found: {key_file}")

        key_cache = torch.load(key_file, weights_only=True, map_location='cpu')
        value_cache = torch.load(value_file, weights_only=True, map_location='cpu')

        print(f"Loaded KV cache shape - Key: {key_cache.shape}, Value: {value_cache.shape}")
        return key_cache, value_cache

    def extract_layer_features(self, kv_cache: torch.Tensor, layer_idx: int,
                               max_tokens: int = None) -> np.ndarray:
        """
        Extract features from a specific layer

        Args:
            kv_cache: Shape [num_layers, batch, num_heads, seq_len, head_dim] or
                           [num_layers, batch, seq_len, hidden_dim]
            layer_idx: Which layer to extract
            max_tokens: Maximum number of tokens to sample (for efficiency)

        Returns:
            features: Shape [num_tokens, hidden_dim]
        """
        # Extract layer
        layer_cache = kv_cache[layer_idx]

        # Remove batch dimension
        layer_cache = layer_cache.squeeze(0)

        # Handle different KV cache formats
        if layer_cache.dim() == 3:
            # Format: [num_heads, seq_len, head_dim]
            # Reshape to [seq_len, num_heads * head_dim]
            num_heads, seq_len, head_dim = layer_cache.shape
            layer_cache = layer_cache.permute(1, 0, 2).reshape(seq_len, num_heads * head_dim)
        elif layer_cache.dim() == 2:
            # Format: [seq_len, hidden_dim]
            pass
        else:
            raise ValueError(f"Unexpected layer_cache shape: {layer_cache.shape}")

        # Sample tokens if needed
        if max_tokens is not None and layer_cache.shape[0] > max_tokens:
            indices = torch.randperm(layer_cache.shape[0])[:max_tokens]
            layer_cache = layer_cache[indices]

        # Convert to float32 if needed (handle BFloat16)
        if layer_cache.dtype == torch.bfloat16:
            layer_cache = layer_cache.float()

        return layer_cache.numpy()

    def compute_pca(self, features: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """
        Compute PCA on features

        Args:
            features: Shape [num_samples, feature_dim]
            n_components: Number of PCA components

        Returns:
            transformed: Shape [num_samples, n_components]
            pca_model: Fitted PCA model
        """
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(features)

        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

        return transformed, pca

    def analyze_single_sample(self, example_id: int, chunk_id: int,
                            selected_layers: List[int], max_tokens: int = 500):
        """
        Analyze KV cache for a single sample

        Args:
            example_id: Example ID
            chunk_id: Chunk ID (1 for first document, 0 for system)
            selected_layers: List of layer indices to analyze
            max_tokens: Maximum tokens to sample per layer
        """
        print(f"\n{'='*80}")
        print(f"Analyzing Example {example_id}, Chunk {chunk_id}")
        print(f"{'='*80}\n")

        # Load KV caches
        no_prep_key, no_prep_value = self.load_kv_cache(self.no_preprocess_path, example_id, chunk_id)
        bge_key, bge_value = self.load_kv_cache(self.bge_preprocess_path, example_id, chunk_id)

        results = {
            'example_id': example_id,
            'chunk_id': chunk_id,
            'layers': {},
        }

        for layer_idx in selected_layers:
            print(f"\n--- Processing Layer {layer_idx} ---")

            # Extract features
            no_prep_key_feat = self.extract_layer_features(no_prep_key, layer_idx, max_tokens)
            no_prep_val_feat = self.extract_layer_features(no_prep_value, layer_idx, max_tokens)
            bge_key_feat = self.extract_layer_features(bge_key, layer_idx, max_tokens)
            bge_val_feat = self.extract_layer_features(bge_value, layer_idx, max_tokens)

            # Compute L2 distance between no_prep and bge
            key_l2_dist = np.linalg.norm(no_prep_key_feat - bge_key_feat, axis=1).mean()
            val_l2_dist = np.linalg.norm(no_prep_val_feat - bge_val_feat, axis=1).mean()

            print(f"Key L2 distance (no_prep vs BGE): {key_l2_dist:.4f}")
            print(f"Value L2 distance (no_prep vs BGE): {val_l2_dist:.4f}")

            # Combine features for PCA
            # Shape: [num_samples*4, hidden_dim]
            all_key_features = np.vstack([no_prep_key_feat, bge_key_feat])
            all_val_features = np.vstack([no_prep_val_feat, bge_val_feat])

            # Compute PCA
            key_pca, key_pca_model = self.compute_pca(all_key_features, n_components=2)
            val_pca, val_pca_model = self.compute_pca(all_val_features, n_components=2)

            # Split back
            n_samples = no_prep_key_feat.shape[0]
            no_prep_key_pca = key_pca[:n_samples]
            bge_key_pca = key_pca[n_samples:]

            no_prep_val_pca = val_pca[:n_samples]
            bge_val_pca = val_pca[n_samples:]

            # Store results
            results['layers'][layer_idx] = {
                'key_l2_dist': float(key_l2_dist),
                'val_l2_dist': float(val_l2_dist),
                'key_pca': {
                    'no_prep': no_prep_key_pca,
                    'bge': bge_key_pca,
                },
                'val_pca': {
                    'no_prep': no_prep_val_pca,
                    'bge': bge_val_pca,
                },
                'key_variance_explained': key_pca_model.explained_variance_ratio_.tolist(),
                'val_variance_explained': val_pca_model.explained_variance_ratio_.tolist(),
            }

        return results

    def plot_layer_comparison(self, results: Dict, output_dir: str):
        """
        Plot PCA visualization for all layers

        Creates a grid of subplots showing Key and Value distributions across layers
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        example_id = results['example_id']
        chunk_id = results['chunk_id']
        layers = sorted(results['layers'].keys())

        # Create figure with subplots
        n_layers = len(layers)
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        # Plot Keys
        fig_key, axes_key = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes_key = axes_key.reshape(1, -1)
        elif n_cols == 1:
            axes_key = axes_key.reshape(-1, 1)

        # Plot Values
        fig_val, axes_val = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes_val = axes_val.reshape(1, -1)
        elif n_cols == 1:
            axes_val = axes_val.reshape(-1, 1)

        for idx, layer_idx in enumerate(layers):
            row = idx // n_cols
            col = idx % n_cols

            layer_data = results['layers'][layer_idx]

            # Plot Keys
            ax_key = axes_key[row, col]
            no_prep_key = layer_data['key_pca']['no_prep']
            bge_key = layer_data['key_pca']['bge']

            ax_key.scatter(no_prep_key[:, 0], no_prep_key[:, 1],
                          alpha=0.5, s=20, c='blue', label='No Preprocess')
            ax_key.scatter(bge_key[:, 0], bge_key[:, 1],
                          alpha=0.5, s=20, c='red', label='BGE')

            var_exp = layer_data['key_variance_explained']
            ax_key.set_xlabel(f'PC1 ({var_exp[0]:.2%})')
            ax_key.set_ylabel(f'PC2 ({var_exp[1]:.2%})')
            ax_key.set_title(f'Layer {layer_idx} - Key\nL2 dist: {layer_data["key_l2_dist"]:.4f}')
            ax_key.legend(fontsize=8)
            ax_key.grid(True, alpha=0.3)

            # Plot Values
            ax_val = axes_val[row, col]
            no_prep_val = layer_data['val_pca']['no_prep']
            bge_val = layer_data['val_pca']['bge']

            ax_val.scatter(no_prep_val[:, 0], no_prep_val[:, 1],
                          alpha=0.5, s=20, c='blue', label='No Preprocess')
            ax_val.scatter(bge_val[:, 0], bge_val[:, 1],
                          alpha=0.5, s=20, c='red', label='BGE')

            var_exp = layer_data['val_variance_explained']
            ax_val.set_xlabel(f'PC1 ({var_exp[0]:.2%})')
            ax_val.set_ylabel(f'PC2 ({var_exp[1]:.2%})')
            ax_val.set_title(f'Layer {layer_idx} - Value\nL2 dist: {layer_data["val_l2_dist"]:.4f}')
            ax_val.legend(fontsize=8)
            ax_val.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_layers, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes_key[row, col].axis('off')
            axes_val[row, col].axis('off')

        fig_key.suptitle(f'Key Cache PCA - Example {example_id}, Chunk {chunk_id}', fontsize=16, y=0.995)
        fig_val.suptitle(f'Value Cache PCA - Example {example_id}, Chunk {chunk_id}', fontsize=16, y=0.995)

        fig_key.tight_layout()
        fig_val.tight_layout()

        # Save figures
        key_path = output_path / f"pca_key_example{example_id}_chunk{chunk_id}.png"
        val_path = output_path / f"pca_value_example{example_id}_chunk{chunk_id}.png"

        fig_key.savefig(key_path, dpi=150, bbox_inches='tight')
        fig_val.savefig(val_path, dpi=150, bbox_inches='tight')

        print(f"\nSaved Key PCA plot to: {key_path}")
        print(f"Saved Value PCA plot to: {val_path}")

        plt.close(fig_key)
        plt.close(fig_val)

    def plot_l2_distance_trends(self, all_results: List[Dict], output_dir: str):
        """
        Plot L2 distance trends across layers for multiple samples
        """
        output_path = Path(output_dir)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for results in all_results:
            example_id = results['example_id']
            chunk_id = results['chunk_id']

            layers = sorted(results['layers'].keys())
            key_dists = [results['layers'][l]['key_l2_dist'] for l in layers]
            val_dists = [results['layers'][l]['val_l2_dist'] for l in layers]

            label = f"Ex{example_id}_Ch{chunk_id}"
            ax1.plot(layers, key_dists, marker='o', label=label, linewidth=2)
            ax2.plot(layers, val_dists, marker='s', label=label, linewidth=2)

        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Mean L2 Distance', fontsize=12)
        ax1.set_title('Key Cache: No Preprocess vs BGE', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Mean L2 Distance', fontsize=12)
        ax2.set_title('Value Cache: No Preprocess vs BGE', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        trend_path = output_path / "l2_distance_trends.png"
        fig.savefig(trend_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved L2 distance trends to: {trend_path}")

        plt.close(fig)

    def plot_variance_explained(self, all_results: List[Dict], output_dir: str):
        """
        Plot variance explained by first 2 PCA components across layers
        """
        output_path = Path(output_dir)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for results in all_results:
            example_id = results['example_id']
            chunk_id = results['chunk_id']

            layers = sorted(results['layers'].keys())
            key_var = [sum(results['layers'][l]['key_variance_explained']) for l in layers]
            val_var = [sum(results['layers'][l]['val_variance_explained']) for l in layers]

            label = f"Ex{example_id}_Ch{chunk_id}"
            ax1.plot(layers, key_var, marker='o', label=label, linewidth=2)
            ax2.plot(layers, val_var, marker='s', label=label, linewidth=2)

        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Variance Explained (PC1+PC2)', fontsize=12)
        ax1.set_title('Key Cache: PCA Variance Explained', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Variance Explained (PC1+PC2)', fontsize=12)
        ax2.set_title('Value Cache: PCA Variance Explained', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        fig.tight_layout()

        var_path = output_path / "pca_variance_explained.png"
        fig.savefig(var_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved variance explained plot to: {var_path}")

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize KV Cache PCA across layers')
    parser.add_argument('--cache_dir', type=str, default='/mnt/data3/tmp/fusionrag',
                       help='Base directory for KV cache')
    parser.add_argument('--dataset', type=str, default='musique',
                       help='Dataset name')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct',
                       help='Model name')
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
    parser.add_argument('--output_dir', type=str, default='./kv_pca_analysis',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = KVCacheAnalyzer(args.cache_dir, args.dataset, args.model_name)

    # Select layers to analyze
    if args.layers is None:
        # Evenly space 6 layers across all layers
        selected_layers = [0, args.max_layers // 5, 2 * args.max_layers // 5,
                          3 * args.max_layers // 5, 4 * args.max_layers // 5, args.max_layers - 1]
    else:
        selected_layers = args.layers

    print(f"Analyzing layers: {selected_layers}")

    # Analyze each sample
    all_results = []
    for example_id in args.sample_ids:
        try:
            results = analyzer.analyze_single_sample(
                example_id, args.chunk_id, selected_layers, args.max_tokens
            )
            all_results.append(results)

            # Plot for this sample
            analyzer.plot_layer_comparison(results, args.output_dir)

        except FileNotFoundError as e:
            print(f"Warning: Skipping example {example_id} - {e}")
            continue

    if len(all_results) > 0:
        # Plot aggregated trends
        analyzer.plot_l2_distance_trends(all_results, args.output_dir)
        analyzer.plot_variance_explained(all_results, args.output_dir)

        # Save summary statistics
        summary_path = Path(args.output_dir) / "summary_statistics.json"
        with open(summary_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for r in all_results:
                json_r = {
                    'example_id': r['example_id'],
                    'chunk_id': r['chunk_id'],
                    'layers': {}
                }
                for layer_idx, layer_data in r['layers'].items():
                    json_r['layers'][layer_idx] = {
                        'key_l2_dist': layer_data['key_l2_dist'],
                        'val_l2_dist': layer_data['val_l2_dist'],
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
