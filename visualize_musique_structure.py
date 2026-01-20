#!/usr/bin/env python3
"""
Visualize the structure and statistics of musique-200.jsonl dataset
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_musique_dataset(data_path):
    """
    Analyze musique-200.jsonl dataset structure

    Returns:
        data_items: list of all data items
        statistics: dict of various statistics
    """
    print(f"Loading dataset from {data_path}...")

    data_items = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_items.append(json.loads(line))

    print(f"Total questions: {len(data_items)}")

    # Collect statistics
    stats = {
        'question_lengths': [],
        'passage_counts': [],
        'context_lengths': [],
        'answer_counts': [],
        'passage_titles': [],
    }

    for item in data_items:
        # Question length
        stats['question_lengths'].append(len(item['input']))

        # Parse passages
        passages = re.split(r'Passage \d+\n', item['context'])
        passages = [p.strip() for p in passages if p.strip()]
        stats['passage_counts'].append(len(passages))

        # Context length
        stats['context_lengths'].append(len(item['context']))

        # Answer count
        stats['answer_counts'].append(len(item['answers']))

        # Collect passage titles (first line of each passage)
        for passage in passages:
            lines = passage.split('\n')
            if lines:
                stats['passage_titles'].append(lines[0])

    return data_items, stats


def plot_statistics(stats, output_path=None):
    """
    Plot various statistics about the musique dataset
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Musique-200 Dataset Structure Analysis', fontsize=16, fontweight='bold')

    # 1. Question length distribution
    ax = axes[0, 0]
    ax.hist(stats['question_lengths'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Question Length (characters)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Question Length Distribution', fontweight='bold')
    ax.axvline(np.mean(stats['question_lengths']), color='red', linestyle='--',
               label=f'Mean: {np.mean(stats["question_lengths"]):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Passage count per question
    ax = axes[0, 1]
    ax.hist(stats['passage_counts'], bins=range(min(stats['passage_counts']),
                                                 max(stats['passage_counts'])+2),
            color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Passages per Question', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Passage Count Distribution', fontweight='bold')
    ax.axvline(np.mean(stats['passage_counts']), color='darkred', linestyle='--',
               label=f'Mean: {np.mean(stats["passage_counts"]):.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. Context length distribution
    ax = axes[0, 2]
    ax.hist(np.array(stats['context_lengths'])/1000, bins=30,
            color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Context Length (K characters)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Context Length Distribution', fontweight='bold')
    ax.axvline(np.mean(stats['context_lengths'])/1000, color='darkgreen', linestyle='--',
               label=f'Mean: {np.mean(stats["context_lengths"])/1000:.1f}K')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Answer count distribution
    ax = axes[1, 0]
    answer_counter = Counter(stats['answer_counts'])
    ax.bar(answer_counter.keys(), answer_counter.values(),
           color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Answers per Question', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Answer Count Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 5. Top passage topics (from titles)
    ax = axes[1, 1]
    # Extract main topic (before " Part" or first few words)
    topics = []
    for title in stats['passage_titles']:
        # Remove "Part N" suffix
        topic = re.sub(r'\s+Part\s+\d+$', '', title)
        # Take first 30 chars
        topic = topic[:30]
        topics.append(topic)

    topic_counter = Counter(topics)
    top_topics = topic_counter.most_common(15)

    topic_names = [t[0] for t in top_topics]
    topic_counts = [t[1] for t in top_topics]

    y_pos = np.arange(len(topic_names))
    ax.barh(y_pos, topic_counts, color='#f39c12', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topic_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontweight='bold')
    ax.set_title('Top 15 Most Common Passage Topics', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
    DATASET SUMMARY
    {'='*40}

    Total Questions: {len(stats['question_lengths'])}

    Question Length (chars):
      Min: {min(stats['question_lengths'])}
      Max: {max(stats['question_lengths'])}
      Mean: {np.mean(stats['question_lengths']):.1f}
      Median: {np.median(stats['question_lengths']):.1f}

    Passages per Question:
      Min: {min(stats['passage_counts'])}
      Max: {max(stats['passage_counts'])}
      Mean: {np.mean(stats['passage_counts']):.1f}
      Median: {np.median(stats['passage_counts']):.1f}

    Context Length (chars):
      Min: {min(stats['context_lengths']):,}
      Max: {max(stats['context_lengths']):,}
      Mean: {np.mean(stats['context_lengths']):,.0f}
      Median: {np.median(stats['context_lengths']):,.0f}

    Answers per Question:
      Min: {min(stats['answer_counts'])}
      Max: {max(stats['answer_counts'])}
      Mean: {np.mean(stats['answer_counts']):.2f}

    Total Unique Passages: {len(set(stats['passage_titles']))}
    """

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
    else:
        plt.show()

    return fig, axes


def print_sample_questions(data_items, n=5):
    """
    Print sample questions from the dataset
    """
    print("\n" + "="*80)
    print(f"SAMPLE QUESTIONS (first {n})")
    print("="*80)

    for i, item in enumerate(data_items[:n], 1):
        passages = re.split(r'Passage \d+\n', item['context'])
        passages = [p.strip() for p in passages if p.strip()]

        print(f"\n{i}. Question: {item['input']}")
        print(f"   Answer(s): {', '.join(item['answers'])}")
        print(f"   Passages: {len(passages)}")
        print(f"   Context Length: {len(item['context']):,} characters")

        # Show first passage title
        if passages:
            first_line = passages[0].split('\n')[0]
            print(f"   First Passage: {first_line[:60]}...")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize musique-200 dataset structure')
    parser.add_argument('--data_path', type=str,
                        default='./data/musique-200.jsonl',
                        help='Path to musique-200.jsonl file')
    parser.add_argument('--output', type=str,
                        default='./data/musique_structure_visualization.png',
                        help='Output path for the figure')
    parser.add_argument('--show', action='store_true',
                        help='Show plot instead of saving')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of sample questions to print')

    args = parser.parse_args()

    # Analyze dataset
    data_items, stats = analyze_musique_dataset(args.data_path)

    # Plot statistics
    output_path = None if args.show else args.output
    plot_statistics(stats, output_path)

    # Print sample questions
    print_sample_questions(data_items, args.samples)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
