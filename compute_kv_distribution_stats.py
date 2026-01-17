#!/usr/bin/env python3
"""
Compute Steering Vectors Between BGE and No_Preprocess KV Caches
Following "Mitigating Overthinking" Paper's Manifold Steering Approach

This script implements the manifold steering method from the paper to compute
steering vectors that align no_preprocess KV caches to BGE distribution.

Method (from paper):
1. Compute steering vector: r = mean(KV_bge) - mean(KV_no_preprocess)
2. PCA-based manifold projection: r_M = P_M @ r (eliminates interference noise)
3. Online intervention: KV_aligned = KV_no_preprocess + α * r_M

Analogy to paper:
- KV_no_preprocess ≈ overthinking activations (redundant, noisy)
- KV_bge ≈ concise activations (focused, after retrieval)
- r ≈ steering direction from overthinking to concise

Key Insights:
- Steering vectors reside on low-dimensional manifold M (d_eff << d)
- Projection eliminates interference noise in orthogonal complement M⊥
- Theorem 4.1: E[||r_noise||²] = tr((I - P_M)Σ_noise)
- Typically k=10 components capture >70% variance

Usage:
    python compute_kv_distribution_stats.py \
        --cache_dir /mnt/data3/tmp/fusionrag \
        --dataset musique \
        --model_name Qwen2.5-7B-Instruct \
        --sample_ids 0 1 2 3 4 5 6 7 8 9 \
        --output_path ./kv_steering_vectors.pt \
        --use_manifold_projection \
        --pca_variance_threshold 0.7

Output format:
    {
        'key_steering': {
            'layer_0': {
                'steering_vector': tensor,  # r_M (projected steering vector)
                'bge_mean': tensor,         # for reference
                'no_prep_mean': tensor      # for reference
            },
            'layer_1': {...},
            ...
        },
        'value_steering': {
            'layer_0': {...},
            ...
        },
        'metadata': {
            'num_samples': int,
            'num_layers': int,
            'sample_ids': list,
            'use_manifold_projection': bool,
            'pca_variance_threshold': float,
        }
    }
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
from sklearn.decomposition import PCA


# Method name to directory mapping
METHOD_DIR_MAP = {
    'no_preprocess': 'kv_cache',
    'bge': 'preprocess_kv_cache_global_topk10_bge',
}

class KVDistributionAnalyzer:
    """Analyzer for computing distribution statistics between BGE and no_preprocess"""

    def __init__(self, cache_dir: str, dataset: str = "musique",
                 model_name: str = "Qwen2.5-7B-Instruct"):
        self.cache_dir = Path(cache_dir)
        self.dataset = dataset
        self.model_name = model_name

        self.method_paths = {}
        for method in ['no_preprocess', 'bge']:
            dir_name = METHOD_DIR_MAP[method]
            path = self.cache_dir / model_name / dataset / dir_name

            if not path.exists():
                raise FileNotFoundError(f"Method '{method}' KV cache not found at: {path}")

            self.method_paths[method] = path
            print(f"Method '{method}': {path}")

    def load_kv_cache(self, method: str, example_id: int, chunk_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load key and value cache for a specific method, example and chunk"""
        path = self.method_paths[method]
        key_file = path / f"{example_id}_{chunk_id}_key.pt"
        value_file = path / f"{example_id}_{chunk_id}_value.pt"

        if not key_file.exists() or not value_file.exists():
            return None, None

        key_cache = torch.load(key_file, weights_only=True, map_location='cpu')
        value_cache = torch.load(value_file, weights_only=True, map_location='cpu')

        return key_cache, value_cache

    def compute_layer_mean(self, cache_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute mean activation for a cache tensor (averaged over seq_len)

        Args:
            cache_tensor: shape [num_heads, seq_len, head_dim]

        Returns:
            mean: [num_heads, head_dim]
        """
        if cache_tensor.dim() == 3:
            # [num_heads, seq_len, head_dim]
            mean = cache_tensor.mean(dim=1)  # [num_heads, head_dim]
        else:
            raise ValueError(f"Unexpected cache tensor shape: {cache_tensor.shape}")

        return mean

    def compute_manifold_projection(self, activations: torch.Tensor,
                                    variance_threshold: float = 0.7) -> torch.Tensor:
        """
        Compute PCA-based manifold projection matrix following the paper's approach

        Args:
            activations: shape [n_samples, n_features] - combined activations from all samples
            variance_threshold: cumulative variance threshold for selecting components

        Returns:
            projection_matrix: [n_features, n_features] projection matrix P_M
        """
        # Convert to numpy for sklearn PCA
        activations_np = activations.numpy()

        # Perform PCA
        pca = PCA()
        pca.fit(activations_np)

        # Find number of components that explain variance_threshold of variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(cumsum_variance, variance_threshold) + 1

        print(f"    PCA: {n_components} components capture {cumsum_variance[n_components-1]:.2%} variance")
        print(f"    Dimension reduction: {activations.shape[1]} -> {n_components}")

        # Get top-k principal components: U_eff [n_features, k]
        U_eff = torch.tensor(pca.components_[:n_components].T, dtype=torch.float32)

        # Compute projection matrix: P_M = U_eff @ U_eff^T
        # This projects any vector onto the k-dimensional manifold
        P_M = U_eff @ U_eff.T  # [n_features, n_features]

        return P_M

    def project_to_manifold(self, vector: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """
        Project a vector onto the manifold using the projection matrix

        Args:
            vector: [num_heads, head_dim] vector to project
            projection_matrix: [num_heads*head_dim, num_heads*head_dim] projection matrix

        Returns:
            projected_vector: [num_heads, head_dim] projected vector
        """
        original_shape = vector.shape
        # Flatten to 1D
        vector_flat = vector.reshape(-1)
        # Project
        projected_flat = projection_matrix @ vector_flat
        # Reshape back
        projected = projected_flat.reshape(original_shape)
        return projected

    def compute_distribution_stats(self, sample_ids: List[int], chunk_id: int = 1,
                                   max_layers: int = 28, use_manifold_projection: bool = True,
                                   pca_variance_threshold: float = 0.7) -> Dict:
        """
        Compute distribution statistics across samples and layers with manifold projection

        Args:
            sample_ids: List of sample IDs to analyze
            chunk_id: Chunk ID to analyze (default: 1)
            max_layers: Maximum number of layers
            use_manifold_projection: Whether to use PCA-based manifold projection (default: True)
            pca_variance_threshold: Variance threshold for PCA components (default: 0.7)

        Returns:
            Dictionary containing layer-wise statistics
        """
        print(f"\nComputing distribution statistics for {len(sample_ids)} samples...")
        print(f"Chunk ID: {chunk_id}, Max layers: {max_layers}")
        print(f"Manifold projection: {use_manifold_projection}, Variance threshold: {pca_variance_threshold}")

        # Initialize accumulators for each layer
        # Store mean activations for computing steering vectors
        key_means = {f'layer_{i}': {'bge': [], 'no_prep': []} for i in range(max_layers)}
        value_means = {f'layer_{i}': {'bge': [], 'no_prep': []} for i in range(max_layers)}

        # Collect raw activations for PCA (if manifold projection is enabled)
        key_raw_activations = {f'layer_{i}': [] for i in range(max_layers)} if use_manifold_projection else None
        value_raw_activations = {f'layer_{i}': [] for i in range(max_layers)} if use_manifold_projection else None

        valid_samples = 0

        # Collect activations from all samples
        for sample_id in tqdm(sample_ids, desc="Processing samples"):
            # Load KV caches for both methods
            bge_key, bge_value = self.load_kv_cache('bge', sample_id, chunk_id)
            no_prep_key, no_prep_value = self.load_kv_cache('no_preprocess', sample_id, chunk_id)

            if bge_key is None or no_prep_key is None:
                print(f"  Skipping sample {sample_id}: cache not found")
                continue

            valid_samples += 1

            # Determine number of layers from cache shape
            if bge_key.dim() == 4:
                num_layers = bge_key.shape[0]
            else:
                num_layers = 1
                bge_key = bge_key.unsqueeze(0)
                bge_value = bge_value.unsqueeze(0)
                no_prep_key = no_prep_key.unsqueeze(0)
                no_prep_value = no_prep_value.unsqueeze(0)

            # Process each layer
            for layer_idx in range(min(num_layers, max_layers)):
                layer_name = f'layer_{layer_idx}'

                # Extract layer data
                bge_key_layer = bge_key[layer_idx]      # [num_heads, seq_len, head_dim]
                bge_value_layer = bge_value[layer_idx]
                no_prep_key_layer = no_prep_key[layer_idx]
                no_prep_value_layer = no_prep_value[layer_idx]

                # Compute mean activations for this layer (averaged over seq_len)
                bge_key_mean = self.compute_layer_mean(bge_key_layer)      # [num_heads, head_dim]
                bge_value_mean = self.compute_layer_mean(bge_value_layer)
                no_prep_key_mean = self.compute_layer_mean(no_prep_key_layer)
                no_prep_value_mean = self.compute_layer_mean(no_prep_value_layer)

                # Accumulate means for averaging across samples
                key_means[layer_name]['bge'].append(bge_key_mean)
                key_means[layer_name]['no_prep'].append(no_prep_key_mean)
                value_means[layer_name]['bge'].append(bge_value_mean)
                value_means[layer_name]['no_prep'].append(no_prep_value_mean)

                # Collect raw activations for PCA
                if use_manifold_projection:
                    # Flatten: [num_heads, seq_len, head_dim] -> [seq_len, num_heads * head_dim]
                    bge_key_flat = bge_key_layer.transpose(0, 1).reshape(bge_key_layer.shape[1], -1)
                    no_prep_key_flat = no_prep_key_layer.transpose(0, 1).reshape(no_prep_key_layer.shape[1], -1)
                    bge_value_flat = bge_value_layer.transpose(0, 1).reshape(bge_value_layer.shape[1], -1)
                    no_prep_value_flat = no_prep_value_layer.transpose(0, 1).reshape(no_prep_value_layer.shape[1], -1)

                    # Combine both methods for PCA (to find shared manifold)
                    key_raw_activations[layer_name].append(torch.cat([bge_key_flat, no_prep_key_flat], dim=0))
                    value_raw_activations[layer_name].append(torch.cat([bge_value_flat, no_prep_value_flat], dim=0))

        print(f"\nProcessed {valid_samples} valid samples")

        # Compute steering vectors across samples
        print("\nComputing steering vectors...")
        if use_manifold_projection:
            print("Applying PCA-based manifold projection to eliminate interference noise...")

        final_stats = {
            'key_steering': {},
            'value_steering': {},
            'metadata': {
                'num_samples': valid_samples,
                'num_layers': max_layers,
                'sample_ids': sample_ids,
                'chunk_id': chunk_id,
                'use_manifold_projection': use_manifold_projection,
                'pca_variance_threshold': pca_variance_threshold if use_manifold_projection else None,
            }
        }

        for layer_idx in tqdm(range(max_layers), desc="Computing steering vectors per layer"):
            layer_name = f'layer_{layer_idx}'

            if not key_means[layer_name]['bge']:
                continue

            # ========== Key Steering Vector ==========
            # Stack means from all samples
            bge_key_means = torch.stack(key_means[layer_name]['bge'])         # [num_samples, num_heads, head_dim]
            no_prep_key_means = torch.stack(key_means[layer_name]['no_prep'])  # [num_samples, num_heads, head_dim]

            # Compute average activations across samples
            bge_key_mean_avg = bge_key_means.mean(dim=0)      # [num_heads, head_dim]
            no_prep_key_mean_avg = no_prep_key_means.mean(dim=0)  # [num_heads, head_dim]

            # Compute steering vector: r = mean(KV_bge) - mean(KV_no_preprocess)
            # This is the direction from "overthinking" (no_prep) to "concise" (bge)
            key_steering = bge_key_mean_avg - no_prep_key_mean_avg  # [num_heads, head_dim]

            # Apply manifold projection if enabled
            if use_manifold_projection and key_raw_activations[layer_name]:
                print(f"  Layer {layer_idx} - Key cache:")
                # Concatenate all activations for this layer
                key_activations_all = torch.cat(key_raw_activations[layer_name], dim=0)  # [total_seq_len, num_heads*head_dim]

                # Compute manifold projection matrix P_M
                key_projection_matrix = self.compute_manifold_projection(
                    key_activations_all, variance_threshold=pca_variance_threshold
                )

                # Project steering vector onto the manifold: r_M = P_M @ r
                key_steering_projected = self.project_to_manifold(key_steering, key_projection_matrix)

                # Use projected steering vector
                key_steering = key_steering_projected

            final_stats['key_steering'][layer_name] = {
                'steering_vector': key_steering,       # r_M (projected if enabled)
                'bge_mean': bge_key_mean_avg,         # for reference
                'no_prep_mean': no_prep_key_mean_avg  # for reference
            }

            # ========== Value Steering Vector ==========
            bge_value_means = torch.stack(value_means[layer_name]['bge'])
            no_prep_value_means = torch.stack(value_means[layer_name]['no_prep'])

            bge_value_mean_avg = bge_value_means.mean(dim=0)
            no_prep_value_mean_avg = no_prep_value_means.mean(dim=0)

            # Compute steering vector for value cache
            value_steering = bge_value_mean_avg - no_prep_value_mean_avg

            # Apply manifold projection if enabled
            if use_manifold_projection and value_raw_activations[layer_name]:
                print(f"  Layer {layer_idx} - Value cache:")
                value_activations_all = torch.cat(value_raw_activations[layer_name], dim=0)

                value_projection_matrix = self.compute_manifold_projection(
                    value_activations_all, variance_threshold=pca_variance_threshold
                )

                value_steering_projected = self.project_to_manifold(value_steering, value_projection_matrix)
                value_steering = value_steering_projected

            final_stats['value_steering'][layer_name] = {
                'steering_vector': value_steering,
                'bge_mean': bge_value_mean_avg,
                'no_prep_mean': no_prep_value_mean_avg
            }

        return final_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute KV distribution statistics with optional manifold projection"
    )
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Root directory of KV caches")
    parser.add_argument("--dataset", type=str, default="musique",
                       help="Dataset name")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct",
                       help="Model name")
    parser.add_argument("--sample_ids", type=int, nargs='+', required=True,
                       help="List of sample IDs to analyze")
    parser.add_argument("--chunk_id", type=int, default=1,
                       help="Chunk ID to analyze")
    parser.add_argument("--max_layers", type=int, default=28,
                       help="Maximum number of layers")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for statistics file (.pt)")
    parser.add_argument("--use_manifold_projection", action='store_true', default=True,
                       help="Use PCA-based manifold projection to eliminate interference noise (default: True)")
    parser.add_argument("--no_manifold_projection", dest='use_manifold_projection', action='store_false',
                       help="Disable manifold projection")
    parser.add_argument("--pca_variance_threshold", type=float, default=0.7,
                       help="PCA cumulative variance threshold for component selection (default: 0.7)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = KVDistributionAnalyzer(
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        model_name=args.model_name
    )

    # Compute statistics
    stats = analyzer.compute_distribution_stats(
        sample_ids=args.sample_ids,
        chunk_id=args.chunk_id,
        max_layers=args.max_layers,
        use_manifold_projection=args.use_manifold_projection,
        pca_variance_threshold=args.pca_variance_threshold
    )

    # Save statistics
    print(f"\nSaving statistics to: {args.output_path}")
    torch.save(stats, args.output_path)

    # Save metadata as JSON for easy inspection
    metadata_path = Path(args.output_path).with_suffix('.json')
    metadata = {
        'num_samples': stats['metadata']['num_samples'],
        'num_layers': stats['metadata']['num_layers'],
        'sample_ids': stats['metadata']['sample_ids'],
        'chunk_id': stats['metadata']['chunk_id'],
        'use_manifold_projection': stats['metadata']['use_manifold_projection'],
        'pca_variance_threshold': stats['metadata']['pca_variance_threshold'],
        'layers_processed': list(stats['key_steering'].keys()),
        'method': 'manifold_steering',
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")
    print("\n✓ Steering vector computation complete!")
    print(f"\nSteering vectors summary:")
    print(f"  - Samples analyzed: {stats['metadata']['num_samples']}")
    print(f"  - Layers processed: {len(stats['key_steering'])}")
    print(f"  - Method: Manifold Steering (from overthinking paper)")
    if stats['metadata']['use_manifold_projection']:
        print(f"  - Manifold projection: Enabled (variance threshold: {stats['metadata']['pca_variance_threshold']})")
    else:
        print(f"  - Manifold projection: Disabled")
    print(f"  - Output file: {args.output_path}")
    print(f"\nUsage: KV_aligned = KV_no_preprocess + steering_vector")


if __name__ == "__main__":
    main()
