#!/usr/bin/env python3
"""
Visualize document usage frequency in result_reflect.json dataset
Shows how many times each document is used by main questions (gold_docs)
and sub-questions (retrieve docs)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def analyze_document_usage(data_path):
    """
    Analyze document usage across main questions and sub-questions

    Returns:
        doc_to_idx: mapping from document text to index
        main_usage: dict of document_idx -> count in main questions (gold_docs)
        sub_usage: dict of document_idx -> count in sub-questions (retrieve docs)
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Total questions: {len(dataset)}")

    # Collect all unique documents and assign indices
    all_docs = set()

    # First pass: collect all unique documents
    for data_item in dataset:
        # Documents from gold_docs (main question)
        gold_docs = data_item.get("gold_docs", [])
        all_docs.update(gold_docs)

        # Documents from sub-questions
        intermediate_context = data_item.get("intermediate_context", [])
        for sub_q in intermediate_context:
            retrieve_docs = sub_q.get("retrieve docs", [])
            all_docs.update(retrieve_docs)

    # Create document index mapping
    doc_to_idx = {doc: idx for idx, doc in enumerate(sorted(all_docs))}
    print(f"Total unique documents: {len(doc_to_idx)}")

    # Count usage for each document
    main_usage = defaultdict(int)  # gold_docs usage
    sub_usage = defaultdict(int)   # retrieve docs usage

    # Second pass: count document usage
    for data_item in dataset:
        # Count main question usage (gold_docs)
        gold_docs = data_item.get("gold_docs", [])
        for doc in gold_docs:
            doc_idx = doc_to_idx[doc]
            main_usage[doc_idx] += 1

        # Count sub-question usage (retrieve docs)
        intermediate_context = data_item.get("intermediate_context", [])
        for sub_q in intermediate_context:
            retrieve_docs = sub_q.get("retrieve docs", [])
            for doc in retrieve_docs:
                doc_idx = doc_to_idx[doc]
                sub_usage[doc_idx] += 1

    return doc_to_idx, main_usage, sub_usage


def plot_document_usage(doc_to_idx, main_usage, sub_usage, output_path=None):
    """
    Plot stacked bar chart of document usage

    Args:
        doc_to_idx: mapping from document text to index
        main_usage: dict of document_idx -> count in main questions
        sub_usage: dict of document_idx -> count in sub-questions
        output_path: path to save the figure (optional)
    """
    n_docs = len(doc_to_idx)
    doc_indices = np.arange(n_docs)

    # Prepare data for plotting
    main_counts = np.array([main_usage.get(i, 0) for i in range(n_docs)])
    sub_counts = np.array([sub_usage.get(i, 0) for i in range(n_docs)])
    total_counts = main_counts + sub_counts

    # Statistics
    print("\n" + "="*80)
    print("Document Usage Statistics")
    print("="*80)
    print(f"Total documents: {n_docs}")
    print(f"Documents used in main questions (gold_docs): {np.sum(main_counts > 0)}")
    print(f"Documents used in sub-questions (retrieve docs): {np.sum(sub_counts > 0)}")
    print(f"Documents never used: {np.sum(total_counts == 0)}")
    print(f"\nMax usage in main questions: {np.max(main_counts)}")
    print(f"Max usage in sub-questions: {np.max(sub_counts)}")
    print(f"Max total usage: {np.max(total_counts)}")
    print(f"\nAverage usage in main questions: {np.mean(main_counts):.2f}")
    print(f"Average usage in sub-questions: {np.mean(sub_counts):.2f}")
    print(f"Average total usage: {np.mean(total_counts):.2f}")

    # Find top 10 most used documents
    print("\n" + "="*80)
    print("Top 10 Most Used Documents")
    print("="*80)
    top_indices = np.argsort(total_counts)[::-1][:10]
    idx_to_doc = {idx: doc for doc, idx in doc_to_idx.items()}

    for rank, idx in enumerate(top_indices, 1):
        doc_text = idx_to_doc[idx]
        # Truncate long documents for display
        doc_display = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
        print(f"{rank:2d}. Doc {idx:4d}: Main={main_counts[idx]:2d}, Sub={sub_counts[idx]:3d}, "
              f"Total={total_counts[idx]:3d}")
        print(f"    {doc_display}")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot stacked bars
    bar_width = 0.8
    p1 = ax.bar(doc_indices, main_counts, bar_width,
                label='Main Questions (gold_docs)', color='#3498db', alpha=0.8)
    p2 = ax.bar(doc_indices, sub_counts, bar_width,
                bottom=main_counts, label='Sub-Questions (retrieve docs)',
                color='#e74c3c', alpha=0.8)

    # Styling
    ax.set_xlabel('Document Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Usage Count', fontsize=12, fontweight='bold')
    ax.set_title('Document Usage Frequency in Main Questions vs Sub-Questions',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set x-axis limits
    ax.set_xlim(-0.5, n_docs - 0.5)

    # Add statistics text box
    stats_text = f'Total Docs: {n_docs}\n'
    stats_text += f'Max Usage: {np.max(total_counts)}\n'
    stats_text += f'Avg Usage: {np.mean(total_counts):.2f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def plot_usage_distribution(doc_to_idx, main_usage, sub_usage, output_path=None):
    """
    Plot distribution of usage counts (histogram)
    """
    n_docs = len(doc_to_idx)
    main_counts = np.array([main_usage.get(i, 0) for i in range(n_docs)])
    sub_counts = np.array([sub_usage.get(i, 0) for i in range(n_docs)])
    total_counts = main_counts + sub_counts

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Main question usage distribution
    axes[0].hist(main_counts, bins=range(0, int(np.max(main_counts)) + 2),
                 color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Usage Count', fontweight='bold')
    axes[0].set_ylabel('Number of Documents', fontweight='bold')
    axes[0].set_title('Main Questions (gold_docs)', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Sub-question usage distribution
    axes[1].hist(sub_counts, bins=range(0, int(np.max(sub_counts)) + 2),
                 color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Usage Count', fontweight='bold')
    axes[1].set_ylabel('Number of Documents', fontweight='bold')
    axes[1].set_title('Sub-Questions (retrieve docs)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Total usage distribution
    axes[2].hist(total_counts, bins=range(0, int(np.max(total_counts)) + 2),
                 color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Usage Count', fontweight='bold')
    axes[2].set_ylabel('Number of Documents', fontweight='bold')
    axes[2].set_title('Total Usage', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution figure saved to: {output_path}")
    else:
        plt.show()

    return fig, axes


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize document usage in dataset')
    parser.add_argument('--data_path', type=str,
                        default='./data/result_reflect.json',
                        help='Path to dataset JSON file')
    parser.add_argument('--output', type=str,
                        default='./data/doc_usage_visualization.png',
                        help='Output path for the figure')
    parser.add_argument('--dist_output', type=str,
                        default='./data/doc_usage_distribution.png',
                        help='Output path for the distribution figure')
    parser.add_argument('--show', action='store_true',
                        help='Show plot instead of saving')

    args = parser.parse_args()

    # Analyze document usage
    doc_to_idx, main_usage, sub_usage = analyze_document_usage(args.data_path)

    # Plot stacked bar chart
    output_path = None if args.show else args.output
    plot_document_usage(doc_to_idx, main_usage, sub_usage, output_path)

    # Plot usage distribution
    dist_output_path = None if args.show else args.dist_output
    plot_usage_distribution(doc_to_idx, main_usage, sub_usage, dist_output_path)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
