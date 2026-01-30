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

def build_doc_index(doc_pool: List[Dict]) -> tuple:
    """
    Build mappings from document text to document ID.
    Returns: (exact_match_dict, doc_pool_list)
    """
    text_to_id = {}
    for doc in doc_pool:
        text = doc['text'].strip()
        doc_id = doc['id']
        text_to_id[text] = doc_id
    return text_to_id, doc_pool

def replace_doc_with_idx(doc_text: str, text_to_id: Dict[str, int], doc_pool: List[Dict], stats: Dict[str, int]) -> Union[int, Dict[str, Any]]:
    """
    Replace document text with index reference.
    Try exact match first, then substring match (for documents with title prefixes).
    """
    text = doc_text.strip()

    # Strategy 1: 精确匹配
    if text in text_to_id:
        stats['exact'] += 1
        return text_to_id[text]

    # Strategy 2: 子串匹配（BGE文档可能是池中文档的子串）
    # 例如：BGE="Naseeb(...)" 池="Naseeb (1981 film): Naseeb(...)"
    for pool_doc in doc_pool:
        pool_text = pool_doc['text']
        # 检查BGE文档是否是池中文档的子串（去除池中可能的前缀）
        if text in pool_text and len(text) < len(pool_text):
            # 确保不是部分匹配（至少覆盖池中文档的80%）
            if len(text) >= len(pool_text) * 0.8:
                stats['substring'] += 1
                return pool_doc['id']

    # Strategy 3: 反向子串匹配（池中文档可能是BGE文档的子串）
    for pool_doc in doc_pool:
        pool_text = pool_doc['text'].strip()
        if pool_text in text and len(pool_text) < len(text):
            if len(pool_text) >= len(text) * 0.8:
                stats['substring'] += 1
                return pool_doc['id']

    # 所有匹配策略都失败
    stats['not_found'] += 1
    return {"idx": -1, "error": "not_found_in_pool", "text_preview": text[:100]}

def process_docs_list(docs: List[str], text_to_id: Dict[str, int], doc_pool: List[Dict], stats: Dict[str, int]) -> List[Union[int, Dict]]:
    """Process a list of documents, replacing texts with indices."""
    return [replace_doc_with_idx(doc, text_to_id, doc_pool, stats) for doc in docs]

def optimize_sample(sample: Dict, text_to_id: Dict[str, int], doc_pool: List[Dict], stats: Dict[str, int]) -> Dict:
    """Optimize a single sample by replacing all document texts with indices."""
    optimized = sample.copy()

    # Replace gold_docs
    if 'gold_docs' in optimized and optimized['gold_docs']:
        optimized['gold_docs'] = process_docs_list(optimized['gold_docs'], text_to_id, doc_pool, stats)

    # Replace retrieved_results
    if 'retrieved_results' in optimized and optimized['retrieved_results']:
        optimized['retrieved_results'] = process_docs_list(optimized['retrieved_results'], text_to_id, doc_pool, stats)

    # Replace intermediate_context
    if 'intermediate_context' in optimized:
        for context in optimized['intermediate_context']:
            if 'retrieve docs' in context and context['retrieve docs']:
                context['retrieve docs'] = process_docs_list(context['retrieve docs'], text_to_id, doc_pool, stats)

            if 'retrieve docs supported' in context and context['retrieve docs supported']:
                context['retrieve docs supported'] = process_docs_list(context['retrieve docs supported'], text_to_id, doc_pool, stats)

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
    text_to_id, doc_pool = build_doc_index(doc_pool)
    print(f"Built index for {len(text_to_id)} unique documents")

    print("\nLoading result_reflect.json...")
    samples = load_json(input_path)
    print(f"Loaded {len(samples)} samples")

    print("\nOptimizing samples...")
    optimized_samples = []
    stats = {'exact': 0, 'substring': 0, 'not_found': 0}

    for i, sample in enumerate(samples):
        optimized = optimize_sample(sample, text_to_id, doc_pool, stats)
        optimized_samples.append(optimized)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples")

    total = stats['exact'] + stats['substring'] + stats['not_found']
    print(f"\nOptimization complete!")
    print(f"  Total samples: {len(optimized_samples)}")
    print(f"  Total documents processed: {total}")
    print(f"  Exact matches: {stats['exact']} ({stats['exact']/total*100:.1f}%)")
    print(f"  Substring matches: {stats['substring']} ({stats['substring']/total*100:.1f}%)")
    print(f"  Documents not found: {stats['not_found']} ({stats['not_found']/total*100:.1f}%)")
    print(f"  Success rate: {(stats['exact'] + stats['substring'])/total*100:.1f}%")

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
