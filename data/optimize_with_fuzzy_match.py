#!/usr/bin/env python3
"""
Improved optimization script with fuzzy matching for documents not found by exact match.
"""
import json
from typing import Dict, List, Any, Union

def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file with proper formatting."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def normalize_text(text: str) -> str:
    """Normalize text by stripping and removing extra whitespace."""
    return ' '.join(text.strip().split())

def build_doc_index(doc_pool: List[Dict]) -> tuple:
    """Build mappings from document text to document ID."""
    # Exact match: original text stripped
    exact_match = {}
    # Normalized match: text with normalized whitespace
    normalized_match = {}

    for doc in doc_pool:
        text = doc['text'].strip()
        doc_id = doc['id']
        exact_match[text] = doc_id

        # Also store normalized version
        normalized_text = normalize_text(text)
        if normalized_text not in normalized_match:
            normalized_match[normalized_text] = doc_id

    return exact_match, normalized_match

def find_doc_id(doc_text: str, exact_match: Dict[str, int], normalized_match: Dict[str, int],
                doc_pool: List[Dict]) -> Union[int, Dict[str, Any]]:
    """
    Find document ID using multiple matching strategies:
    1. Exact match (stripped)
    2. Normalized whitespace match
    3. Prefix match (for truncated documents)
    """
    text = doc_text.strip()

    # Try exact match
    if text in exact_match:
        return exact_match[text]

    # Try normalized match
    normalized = normalize_text(text)
    if normalized in normalized_match:
        return normalized_match[normalized]

    # Try prefix match for potentially truncated documents
    # Check if the text is a prefix of any document in the pool
    for doc in doc_pool:
        pool_text = doc['text'].strip()
        if pool_text.startswith(text) or text.startswith(pool_text):
            # Found a potential match
            return doc['id']

    # Try checking if normalized text is a prefix
    for doc in doc_pool:
        pool_normalized = normalize_text(doc['text'])
        if pool_normalized.startswith(normalized) or normalized.startswith(pool_normalized):
            return doc['id']

    # Document not found
    return {"idx": -1, "error": "not_found_in_pool", "text_preview": text[:200]}

def process_docs_list(docs: List[str], exact_match: Dict[str, int],
                     normalized_match: Dict[str, int], doc_pool: List[Dict]) -> List[Union[int, Dict]]:
    """Process a list of documents, replacing texts with indices."""
    return [find_doc_id(doc, exact_match, normalized_match, doc_pool) for doc in docs]

def optimize_sample(sample: Dict, exact_match: Dict[str, int],
                   normalized_match: Dict[str, int], doc_pool: List[Dict]) -> Dict:
    """Optimize a single sample by replacing all document texts with indices."""
    optimized = sample.copy()

    # Replace gold_docs
    if 'gold_docs' in optimized and optimized['gold_docs']:
        optimized['gold_docs'] = process_docs_list(
            optimized['gold_docs'], exact_match, normalized_match, doc_pool)

    # Replace retrieved_results
    if 'retrieved_results' in optimized and optimized['retrieved_results']:
        optimized['retrieved_results'] = process_docs_list(
            optimized['retrieved_results'], exact_match, normalized_match, doc_pool)

    # Replace intermediate_context
    if 'intermediate_context' in optimized:
        for context in optimized['intermediate_context']:
            if 'retrieve docs' in context and context['retrieve docs']:
                context['retrieve docs'] = process_docs_list(
                    context['retrieve docs'], exact_match, normalized_match, doc_pool)

            if 'retrieve docs supported' in context and context['retrieve docs supported']:
                context['retrieve docs supported'] = process_docs_list(
                    context['retrieve docs supported'], exact_match, normalized_match, doc_pool)

    return optimized

def main():
    # File paths
    doc_pool_path = '/home/shm/document/exp/FusionRAG/data/musique_input.json'
    input_path = '/home/shm/document/exp/FusionRAG/data/result_reflect.json'
    output_path = '/home/shm/document/exp/FusionRAG/data/result_reflect_optimized.json'

    print("Loading document pool...")
    doc_pool = load_json(doc_pool_path)
    print(f"Loaded {len(doc_pool)} documents from pool")

    print("\nBuilding document indices...")
    exact_match, normalized_match = build_doc_index(doc_pool)
    print(f"Built exact match index for {len(exact_match)} documents")
    print(f"Built normalized match index for {len(normalized_match)} documents")

    print("\nLoading result_reflect.json...")
    samples = load_json(input_path)
    print(f"Loaded {len(samples)} samples")

    print("\nOptimizing samples with fuzzy matching...")
    optimized_samples = []
    not_found_count = 0
    not_found_docs = []

    for i, sample in enumerate(samples):
        optimized = optimize_sample(sample, exact_match, normalized_match, doc_pool)
        optimized_samples.append(optimized)

        # Collect documents not found in pool
        for key in ['gold_docs', 'retrieved_results']:
            if key in optimized and optimized[key]:
                for doc in optimized[key]:
                    if isinstance(doc, dict) and doc.get('idx') == -1:
                        not_found_count += 1
                        not_found_docs.append(doc)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples")

    print(f"\nOptimization complete!")
    print(f"  Total samples: {len(optimized_samples)}")
    print(f"  Documents not found in pool: {not_found_count}")

    if not_found_docs:
        print("\n=== Documents Not Found ===")
        for i, doc in enumerate(not_found_docs[:5]):  # Show first 5
            print(f"{i+1}. {doc['text_preview'][:150]}...")

    print(f"\nSaving optimized data to {output_path}...")
    save_json(optimized_samples, output_path)
    print("Done!")

    # Print sample statistics
    print("\n=== Sample Statistics ===")
    if optimized_samples:
        sample = optimized_samples[0]
        print(f"First sample question: {sample.get('question', 'N/A')[:80]}...")
        if 'gold_docs' in sample:
            print(f"  gold_docs: {sample['gold_docs']}")
        if 'retrieved_results' in sample:
            print(f"  retrieved_results: {sample['retrieved_results'][:5]}")

    # Calculate storage savings
    import os
    original_size = os.path.getsize(input_path)
    optimized_size = os.path.getsize(output_path)
    savings = (1 - optimized_size / original_size) * 100
    print(f"\n=== Storage Efficiency ===")
    print(f"Original file size: {original_size / 1024:.2f} KB")
    print(f"Optimized file size: {optimized_size / 1024:.2f} KB")
    print(f"Space saved: {savings:.2f}%")

if __name__ == "__main__":
    main()
