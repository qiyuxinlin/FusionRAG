#!/usr/bin/env python3
"""
Text KV Cache t-SNE Visualization

This script generates KV cache from arbitrary text passages and visualizes
them using t-SNE dimensionality reduction for better local structure visualization.

Usage:
    # Visualize 2 text passages
    python visualize_text_kv_tsne.py \
        --texts "This is the first passage." "This is the second passage." \
        --model_name Qwen2.5-7B-Instruct \
        --model_path /path/to/model

    # Visualize texts from a file
    python visualize_text_kv_tsne.py \
        --text_file input_texts.txt \
        --model_name Qwen2.5-7B-Instruct \
        --model_path /path/to/model

    # With custom layer selection and perplexity
    python visualize_text_kv_tsne.py \
        --texts "Text 1" "Text 2" "Text 3" \
        --model_name Qwen2.5-7B-Instruct \
        --model_path /path/to/model \
        --layers 0 5 10 15 20 27 \
        --perplexity 30
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")


class TextKVCacheTSNEAnalyzer:
    """Analyzer for generating and visualizing KV cache from text passages using t-SNE"""

    def __init__(self, model_path: str, model_type: str = "qwen2",
                 device: str = "cuda", torch_dtype: str = "float16"):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype)

        # Set environment variable for model type
        os.environ["MODEL_TYPE"] = model_type

        print(f"Loading model from: {model_path}")
        print(f"Model type: {model_type}")
        print(f"Device: {self.device}")
        print(f"Dtype: {torch_dtype}")

        self._load_model()

    def _load_model(self):
        """Load model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.eval()

        # Get model configuration
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers if hasattr(self.config, 'num_hidden_layers') else self.config.num_layers
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)

        print(f"Model loaded successfully!")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Num heads: {self.num_heads}")
        print(f"  Head dim: {self.head_dim}")
        print(f"  KV heads: {self.num_key_value_heads}")

    def generate_kv_cache(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate KV cache for a given text passage

        Returns:
            key_cache: [num_layers, num_kv_heads, seq_len, head_dim]
            value_cache: [num_layers, num_kv_heads, seq_len, head_dim]
        """
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        seq_len = input_ids.shape[1]
        print(f"  Text length: {seq_len} tokens")

        # Prepare KV cache structure
        key_cache = []
        value_cache = []

        # Forward pass through all layers to capture KV cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            # past_key_values is a tuple of tuples
            # Structure: (num_layers, 2 (key/value), batch_size, num_kv_heads, seq_len, head_dim)
            for layer_idx in range(len(past_key_values)):
                # Extract key and value for this layer
                layer_key = past_key_values[layer_idx][0]  # [batch_size, num_kv_heads, seq_len, head_dim]
                layer_value = past_key_values[layer_idx][1]  # [batch_size, num_kv_heads, seq_len, head_dim]

                # Remove batch dimension (assuming batch_size=1)
                layer_key = layer_key.squeeze(0)  # [num_kv_heads, seq_len, head_dim]
                layer_value = layer_value.squeeze(0)  # [num_kv_heads, seq_len, head_dim]

                key_cache.append(layer_key)
                value_cache.append(layer_value)

        # Stack layers
        key_cache = torch.stack(key_cache, dim=0)  # [num_layers, num_kv_heads, seq_len, head_dim]
        value_cache = torch.stack(value_cache, dim=0)  # [num_layers, num_kv_heads, seq_len, head_dim]

        return key_cache, value_cache

    def extract_layer_features(self, kv_cache: torch.Tensor, layer_idx: int,
                               max_tokens: int = None) -> np.ndarray:
        """
        Extract features from a specific layer for t-SNE

        Args:
            kv_cache: [num_layers, num_kv_heads, seq_len, head_dim]
            layer_idx: Layer index
            max_tokens: Maximum tokens to sample (None = all tokens)

        Returns:
            features: [seq_len, num_kv_heads * head_dim] or [max_tokens, num_kv_heads * head_dim]
        """
        layer_cache = kv_cache[layer_idx]  # [num_kv_heads, seq_len, head_dim]

        # Transpose to [seq_len, num_kv_heads, head_dim]
        layer_cache = layer_cache.permute(1, 0, 2)  # [seq_len, num_kv_heads, head_dim]

        # Flatten heads and head_dim
        seq_len, num_kv_heads, head_dim = layer_cache.shape
        layer_cache = layer_cache.reshape(seq_len, num_kv_heads * head_dim)

        # Sample tokens if needed
        if max_tokens is not None and seq_len > max_tokens:
            indices = torch.linspace(0, seq_len-1, max_tokens, device=layer_cache.device).long()
            layer_cache = layer_cache[indices]

        # Move to CPU and convert to numpy
        layer_cache = layer_cache.cpu()
        if layer_cache.dtype == torch.bfloat16:
            layer_cache = layer_cache.float()

        return layer_cache.numpy()

    def compute_tsne(self, features: np.ndarray, n_components: int = 2,
                     perplexity: int = 30, random_state: int = 42) -> np.ndarray:
        """Compute t-SNE on features"""
        n_samples = features.shape[0]
        actual_perplexity = min(perplexity, n_samples - 1)

        if actual_perplexity < perplexity:
            print(f"    Warning: Adjusted perplexity from {perplexity} to {actual_perplexity} (n_samples={n_samples})")

        tsne = TSNE(
            n_components=n_components,
            perplexity=actual_perplexity,
            random_state=random_state,
            n_iter=1000,
            verbose=0
        )
        transformed = tsne.fit_transform(features)

        return transformed

    def analyze_texts(self, texts: List[str], text_labels: List[str] = None,
                     selected_layers: List[int] = None, max_tokens: int = 500,
                     perplexity: int = 30) -> Dict:
        """
        Analyze KV cache for multiple text passages using t-SNE

        Args:
            texts: List of text passages
            text_labels: Optional labels for each text passage
            selected_layers: Layers to analyze (None = evenly spaced layers)
            max_tokens: Maximum tokens per layer
            perplexity: t-SNE perplexity parameter

        Returns:
            results dictionary with t-SNE analysis for all texts
        """
        if text_labels is None:
            text_labels = [f"Text {i}" for i in range(len(texts))]

        if len(text_labels) != len(texts):
            raise ValueError("Number of labels must match number of texts")

        # Select layers
        if selected_layers is None:
            selected_layers = [0, self.num_layers // 5, 2 * self.num_layers // 5,
                             3 * self.num_layers // 5, 4 * self.num_layers // 5, self.num_layers - 1]

        print(f"\n{'='*80}")
        print(f"Analyzing {len(texts)} text passages with t-SNE")
        print(f"Layers: {selected_layers}")
        print(f"Perplexity: {perplexity}")
        print(f"{'='*80}\n")

        # Generate KV cache for all texts
        all_key_caches = []
        all_value_caches = []

        for i, (text, label) in enumerate(zip(texts, text_labels)):
            print(f"[{i+1}/{len(texts)}] Processing: {label}")
            print(f"  Text preview: {text[:100]}...")

            key_cache, value_cache = self.generate_kv_cache(text)
            all_key_caches.append(key_cache)
            all_value_caches.append(value_cache)
            print(f"  ✓ Key cache: {key_cache.shape}")
            print(f"  ✓ Value cache: {value_cache.shape}")
            print()

        results = {
            'texts': text_labels,
            'layers': {},
            'num_texts': len(texts),
            'perplexity': perplexity,
        }

        # Analyze each layer
        for layer_idx in selected_layers:
            print(f"Analyzing Layer {layer_idx}...")

            # Extract features for all texts at this layer
            layer_key_features = []
            layer_val_features = []

            for i in range(len(texts)):
                key_feat = self.extract_layer_features(all_key_caches[i], layer_idx, max_tokens)
                val_feat = self.extract_layer_features(all_value_caches[i], layer_idx, max_tokens)
                layer_key_features.append(key_feat)
                layer_val_features.append(val_feat)

            # Combine all features for joint t-SNE
            all_key_features = np.vstack(layer_key_features)
            all_val_features = np.vstack(layer_val_features)

            # Compute t-SNE
            print(f"  Computing t-SNE for Key cache...")
            key_tsne = self.compute_tsne(all_key_features, n_components=2, perplexity=perplexity)

            print(f"  Computing t-SNE for Value cache...")
            val_tsne = self.compute_tsne(all_val_features, n_components=2, perplexity=perplexity)

            # Split back to individual texts
            n_samples = layer_key_features[0].shape[0]
            key_tsne_split = []
            val_tsne_split = []

            start_idx = 0
            for i in range(len(texts)):
                end_idx = start_idx + n_samples
                key_tsne_split.append(key_tsne[start_idx:end_idx])
                val_tsne_split.append(val_tsne[start_idx:end_idx])
                start_idx = end_idx

            # Store results
            results['layers'][layer_idx] = {
                'key_tsne': key_tsne_split,
                'val_tsne': val_tsne_split,
            }

        return results

    def plot_comparison(self, results: Dict, output_dir: str):
        """Plot t-SNE visualization for all layers and texts"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        text_labels = results['texts']
        layers = sorted(results['layers'].keys())

        n_layers = len(layers)
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        # High-contrast color scheme (similar to PCA script)
        COLOR_PALETTE = [
            "#211fb4",  # Blue
            "#d62728",  # Red
            "#2ca02c",  # Green
            "#9467bd",  # Purple
            "#ff7f0e",  # Orange
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#037678",  # Cyan
            "#bcbd22",  # Yellow-green
        ]

        # Assign colors to texts
        colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(text_labels))]

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
            for i, label in enumerate(text_labels):
                tsne_data = layer_data['key_tsne'][i]
                ax_key.scatter(tsne_data[:, 0], tsne_data[:, 1],
                              alpha=0.6, s=20, c=colors[i], label=label, edgecolors='none')

            ax_key.set_xlabel('t-SNE 1')
            ax_key.set_ylabel('t-SNE 2')
            ax_key.set_title(f'Layer {layer_idx} - Key Cache', fontsize=10)
            ax_key.legend(fontsize=7, loc='best')
            ax_key.grid(True, alpha=0.3)

            # Plot Values
            ax_val = axes_val[row, col]
            for i, label in enumerate(text_labels):
                tsne_data = layer_data['val_tsne'][i]
                ax_val.scatter(tsne_data[:, 0], tsne_data[:, 1],
                              alpha=0.6, s=20, c=colors[i], label=label, edgecolors='none')

            ax_val.set_xlabel('t-SNE 1')
            ax_val.set_ylabel('t-SNE 2')
            ax_val.set_title(f'Layer {layer_idx} - Value Cache', fontsize=10)
            ax_val.legend(fontsize=7, loc='best')
            ax_val.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_layers, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes_key[row, col].axis('off')
            axes_val[row, col].axis('off')

        perplexity = results.get('perplexity', 30)
        fig_key.suptitle(f'Key Cache t-SNE Visualization (perplexity={perplexity})\n{len(text_labels)} Text Passages',
                         fontsize=14, y=0.998)
        fig_val.suptitle(f'Value Cache t-SNE Visualization (perplexity={perplexity})\n{len(text_labels)} Text Passages',
                         fontsize=14, y=0.998)

        fig_key.tight_layout()
        fig_val.tight_layout()

        # Save figures
        key_path = output_path / "tsne_key_cache.png"
        val_path = output_path / "tsne_value_cache.png"

        fig_key.savefig(key_path, dpi=150, bbox_inches='tight')
        fig_val.savefig(val_path, dpi=150, bbox_inches='tight')

        print(f"\n✓ Saved Key t-SNE plot to: {key_path}")
        print(f"✓ Saved Value t-SNE plot to: {val_path}")

        plt.close(fig_key)
        plt.close(fig_val)

    def save_results(self, results: Dict, output_dir: str):
        """Save analysis results to JSON"""
        output_path = Path(output_dir)

        # Prepare JSON-serializable results
        json_results = {
            'texts': results['texts'],
            'num_texts': results['num_texts'],
            'perplexity': results.get('perplexity', 30),
            'layers': {},
        }

        for layer_idx in results['layers'].keys():
            json_results['layers'][layer_idx] = {}

        summary_path = output_path / "summary_statistics.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved summary statistics to: {summary_path}")


def load_texts_from_file(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load texts from a file

    Format options:
    1. One text per line
    2. Label: Text format (e.g., "Passage 1: This is the text...")

    Returns:
        texts: List of text passages
        labels: List of labels
    """
    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue

            # Check if line contains label separator
            if ':' in line and len(line.split(':', 1)) == 2:
                label, text = line.split(':', 1)
                labels.append(label.strip())
                texts.append(text.strip())
            else:
                labels.append(f"Text {line_num}")
                texts.append(line)

    return texts, labels


def main():
    parser = argparse.ArgumentParser(description='Text KV Cache t-SNE Visualization')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model (e.g., /path/to/Qwen2.5-7B-Instruct)')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct',
                       help='Model name (for reference)')
    parser.add_argument('--model_type', type=str, default='qwen2',
                       choices=['qwen2', 'llama', 'mistral'],
                       help='Model architecture type')
    parser.add_argument('--texts', type=str, nargs='+',
                       help='Text passages to analyze (space-separated)')
    parser.add_argument('--text_file', type=str,
                       help='File containing text passages (one per line, or "Label: Text" format)')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Optional labels for each text passage')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Specific layers to analyze (default: evenly spaced 6 layers)')
    parser.add_argument('--all_layers', action='store_true',
                       help='Analyze all layers (overrides --layers)')
    parser.add_argument('--max_tokens', type=int, default=500,
                       help='Maximum tokens to sample per layer')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity (default: 30, typical values: 5-50)')
    parser.add_argument('--output_dir', type=str, default='./text_kv_tsne_analysis',
                       help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--torch_dtype', type=str, default='float16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Torch data type for model')

    args = parser.parse_args()

    # Validate inputs
    if args.texts is None and args.text_file is None:
        raise ValueError("Either --texts or --text_file must be provided")

    if args.texts is not None and args.text_file is not None:
        raise ValueError("Only one of --texts or --text_file should be provided")

    # Load texts
    if args.texts:
        texts = args.texts
        labels = args.labels if args.labels else [f"Text {i}" for i in range(len(texts))]
    else:
        texts, labels = load_texts_from_file(args.text_file)

    if len(labels) != len(texts):
        raise ValueError("Number of labels must match number of texts")

    if len(texts) < 2:
        print("Warning: Only 1 text provided. t-SNE visualization requires at least 2 texts for comparison.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    print(f"\n{'='*80}")
    print("Text KV Cache t-SNE Analysis")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Number of texts: {len(texts)}")
    print(f"Perplexity: {args.perplexity}")
    print(f"{'='*80}\n")

    # Print text preview
    for label, text in zip(labels, texts):
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  {label}: {preview}")
    print()

    # Initialize analyzer
    analyzer = TextKVCacheTSNEAnalyzer(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        torch_dtype=args.torch_dtype
    )

    # Select layers to analyze
    if args.all_layers:
        selected_layers = list(range(analyzer.num_layers))
        print(f"Analyzing all {len(selected_layers)} layers")
    elif args.layers is not None:
        selected_layers = args.layers
        print(f"Analyzing specified layers: {selected_layers}")
    else:
        # Default: evenly spaced layers
        selected_layers = [0, analyzer.num_layers // 5, 2 * analyzer.num_layers // 5,
                         3 * analyzer.num_layers // 5, 4 * analyzer.num_layers // 5, analyzer.num_layers - 1]
        print(f"Analyzing default layers (evenly spaced): {selected_layers}")

    # Analyze texts
    results = analyzer.analyze_texts(
        texts=texts,
        text_labels=labels,
        selected_layers=selected_layers,
        max_tokens=args.max_tokens,
        perplexity=args.perplexity
    )

    # Plot and save results
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}\n")

    analyzer.plot_comparison(results, args.output_dir)
    analyzer.save_results(results, args.output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - tsne_key_cache.png")
    print(f"  - tsne_value_cache.png")
    print(f"  - summary_statistics.json")
    print(f"\n📌 t-SNE vs PCA:")
    print(f"  • t-SNE: Better for preserving local structure and discovering clusters")
    print(f"  • PCA: Faster, preserves global structure, has variance explained")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
