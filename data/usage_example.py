#!/usr/bin/env python3
"""
Example script showing how to use the optimized data with document indices.
"""
import json
from typing import Dict, List, Any

def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

class DocumentPool:
    """Document pool for accessing documents by index."""

    def __init__(self, doc_pool_path: str):
        """Initialize the document pool."""
        self.docs = load_json(doc_pool_path)
        self.idx_to_doc = {doc['id']: doc for doc in self.docs}
        print(f"Loaded document pool with {len(self.docs)} documents")

    def get_doc_text(self, idx: int) -> str:
        """Get document text by index."""
        if idx in self.idx_to_doc:
            return self.idx_to_doc[idx]['text']
        else:
            return f"[Document {idx} not found in pool]"

    def get_doc_metadata(self, idx: int) -> Dict:
        """Get document metadata by index."""
        if idx in self.idx_to_doc:
            return self.idx_to_doc[idx].get('metadata', {})
        else:
            return {}

    def get_docs_texts(self, indices: List[int]) -> List[str]:
        """Get multiple document texts by indices."""
        return [self.get_doc_text(idx) for idx in indices]

def demonstrate_usage():
    """Demonstrate how to use the optimized data."""

    # Initialize document pool
    doc_pool = DocumentPool('/home/shm/document/exp/FusionRAG/data/musique_input.json')

    # Load optimized data
    print("\nLoading optimized result data...")
    samples = load_json('/home/shm/document/exp/FusionRAG/data/result_reflect_optimized.json')
    print(f"Loaded {len(samples)} samples")

    # Example 1: Access gold documents for the first sample
    print("\n" + "="*80)
    print("Example 1: Accessing gold documents")
    print("="*80)
    sample = samples[0]
    print(f"Question: {sample['question']}")
    print(f"Gold doc indices: {sample['gold_docs']}")
    print("\nGold documents (full text):")
    for i, idx in enumerate(sample['gold_docs'], 1):
        text = doc_pool.get_doc_text(idx)
        print(f"\n{i}. [Doc {idx}]: {text}")

    # Example 2: Access retrieved results
    print("\n" + "="*80)
    print("Example 2: Accessing retrieved results")
    print("="*80)
    print(f"Retrieved result indices: {sample['retrieved_results']}")
    print("\nRetrieved documents:")
    for i, idx in enumerate(sample['retrieved_results'], 1):
        text = doc_pool.get_doc_text(idx)
        print(f"\n{i}. [Doc {idx}]: {text[:200]}...")  # Show first 200 chars

    # Example 3: Access intermediate context documents
    print("\n" + "="*80)
    print("Example 3: Accessing intermediate context documents")
    print("="*80)
    if 'intermediate_context' in sample and sample['intermediate_context']:
        for i, context in enumerate(sample['intermediate_context'], 1):
            print(f"\nIntermediate step {i}:")
            print(f"  Query: {context['query']}")
            print(f"  Retrieved doc indices: {context['retrieve docs'][:5]}...")  # Show first 5
            print(f"  Supported doc indices: {context['retrieve docs supported']}")

            print(f"\n  Supported documents (full text):")
            for doc_idx in context['retrieve docs supported']:
                text = doc_pool.get_doc_text(doc_idx)
                print(f"    [Doc {doc_idx}]: {text[:150]}...")

    # Example 4: Statistics
    print("\n" + "="*80)
    print("Example 4: Data statistics")
    print("="*80)
    total_gold_docs = sum(len(s.get('gold_docs', [])) for s in samples)
    total_retrieved = sum(len(s.get('retrieved_results', [])) for s in samples)

    # Count intermediate docs
    total_intermediate = 0
    for s in samples:
        if 'intermediate_context' in s:
            for ctx in s['intermediate_context']:
                total_intermediate += len(ctx.get('retrieve docs', []))

    print(f"Total samples: {len(samples)}")
    print(f"Total gold documents references: {total_gold_docs}")
    print(f"Total retrieved documents references: {total_retrieved}")
    print(f"Total intermediate documents references: {total_intermediate}")
    print(f"\nUnique documents in pool: {len(doc_pool.docs)}")

    # Example 5: Batch processing
    print("\n" + "="*80)
    print("Example 5: Batch processing all samples")
    print("="*80)
    for i, sample in enumerate(samples[:3], 1):  # Show first 3 samples
        print(f"\nSample {i}:")
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Gold docs: {sample['gold_docs']}")
        print(f"  Retrieved: {sample['retrieved_results']}")
        print(f"  Judge result: {sample.get('llm_judge', 'N/A')}")

if __name__ == "__main__":
    demonstrate_usage()
