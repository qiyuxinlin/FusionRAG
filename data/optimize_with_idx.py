#!/usr/bin/env python3
"""
Optimize result_reflect.json by replacing full document texts with document pool indices.
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

def build_doc_index(doc_pool: List[Dict]) -> Dict[str, int]:
    """Build a mapping from document text to document ID."""
    text_to_id = {}
    for doc in doc_pool:
        text = doc['text'].strip()
        doc_id = doc['id']
        text_to_id[text] = doc_id
    return text_to_id

def replace_doc_with_idx(doc_text: str, text_to_id: Dict[str, int]) -> Union[int, Dict[str, Any]]:
    """Replace document text with index reference, or return error info if not found."""
    text = doc_text.strip()
    if text in text_to_id:
        return text_to_id[text]
    else:
        # Document not found in pool - keep original or mark as not found
        return {"idx": -1, "error": "not_found_in_pool", "text_preview": text[:100]}

def process_docs_list(docs: List[str], text_to_id: Dict[str, int]) -> List[Union[int, Dict]]:
    """Process a list of documents, replacing texts with indices."""
    return [replace_doc_with_idx(doc, text_to_id) for doc in docs]

def optimize_sample(sample: Dict, text_to_id: Dict[str, int]) -> Dict:
    """Optimize a single sample by replacing all document texts with indices."""
    optimized = sample.copy()

    # Replace gold_docs
    if 'gold_docs' in optimized and optimized['gold_docs']:
        optimized['gold_docs'] = process_docs_list(optimized['gold_docs'], text_to_id)

    # Replace retrieved_results
    if 'retrieved_results' in optimized and optimized['retrieved_results']:
        optimized['retrieved_results'] = process_docs_list(optimized['retrieved_results'], text_to_id)

    # Replace intermediate_context
    if 'intermediate_context' in optimized:
        for context in optimized['intermediate_context']:
            if 'retrieve docs' in context and context['retrieve docs']:
                context['retrieve docs'] = process_docs_list(context['retrieve docs'], text_to_id)

            if 'retrieve docs supported' in context and context['retrieve docs supported']:
                context['retrieve docs supported'] = process_docs_list(context['retrieve docs supported'], text_to_id)

    return optimized

def main():
    # File paths"/home/shm/document/exp/FusionRAG/data/2wiki_input.json"
    doc_pool_path = '/home/shm/document/exp/FusionRAG/data/2wiki_input.json'
    input_path = '/home/shm/document/exp/FusionRAG/data/2wikimqa_reflect.json'
    output_path = '/home/shm/document/exp/FusionRAG/data/2wikimqa_reflect_optimized.json'

    print("Loading document pool...")
    doc_pool = load_json(doc_pool_path)
    print(f"Loaded {len(doc_pool)} documents from pool")

    print("\nBuilding document index...")
    text_to_id = build_doc_index(doc_pool)
    print(f"Built index for {len(text_to_id)} unique documents")

    print("\nLoading result_reflect.json...")
    samples = load_json(input_path)
    print(f"Loaded {len(samples)} samples")

    print("\nOptimizing samples...")
    optimized_samples = []
    not_found_count = 0

    for i, sample in enumerate(samples):
        optimized = optimize_sample(sample, text_to_id)
        optimized_samples.append(optimized)

        # Count documents not found in pool
        for key in ['gold_docs', 'retrieved_results']:
            if key in optimized and optimized[key]:
                for doc in optimized[key]:
                    if isinstance(doc, dict) and doc.get('idx') == -1:
                        not_found_count += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples")

    print(f"\nOptimization complete!")
    print(f"  Total samples: {len(optimized_samples)}")
    print(f"  Documents not found in pool: {not_found_count}")

    print(f"\nSaving optimized data to {output_path}...")
    save_json(optimized_samples, output_path)
    print("Done!")

    # Print sample statistics
    print("\n=== Sample Statistics ===")
    if optimized_samples:
        sample = optimized_samples[0]
        print(f"First sample keys: {list(sample.keys())}")
        if 'gold_docs' in sample:
            print(f"  gold_docs: {sample['gold_docs']}")
        if 'retrieved_results' in sample:
            print(f"  retrieved_results (first 3): {sample['retrieved_results'][:3]}")

if __name__ == "__main__":
    main()
