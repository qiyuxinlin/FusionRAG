#!/usr/bin/env python3
"""
Document Pool Manager for global document ID management
"""
import json
from typing import Dict, List

class DocumentPoolManager:
    """
    Manages a global document pool with index-based access.
    This enables cross-question document reuse by using global doc IDs.
    """

    def __init__(self, doc_pool_path: str):
        """
        Initialize the document pool manager.

        Args:
            doc_pool_path: Path to the document pool JSON file (e.g., musique_input.json)
        """
        print(f"Loading document pool from: {doc_pool_path}")
        with open(doc_pool_path, 'r', encoding='utf-8') as f:
            self.doc_pool = json.load(f)

        # Build index: doc_id -> document data
        self.id_to_doc = {doc['id']: doc for doc in self.doc_pool}
        print(f"  Loaded {len(self.doc_pool)} documents")

    def get_doc_text(self, doc_id: int) -> str:
        """Get document text by global doc ID."""
        if doc_id in self.id_to_doc:
            return self.id_to_doc[doc_id]['text']
        else:
            raise KeyError(f"Document ID {doc_id} not found in pool")

    def get_doc_metadata(self, doc_id: int) -> Dict:
        """Get document metadata by global doc ID."""
        if doc_id in self.id_to_doc:
            return self.id_to_doc[doc_id].get('metadata', {})
        else:
            return {}

    def get_docs_texts(self, doc_ids: List[int]) -> List[str]:
        """Get multiple document texts by global doc IDs."""
        return [self.get_doc_text(doc_id) for doc_id in doc_ids]

    def __len__(self):
        """Return the number of documents in the pool."""
        return len(self.doc_pool)

    def __contains__(self, doc_id: int):
        """Check if a doc_id exists in the pool."""
        return doc_id in self.id_to_doc


def load_optimized_data(data_path: str, doc_pool_manager: DocumentPoolManager) -> List[Dict]:
    """
    Load optimized data file where documents are represented as indices.

    Args:
        data_path: Path to optimized data file (e.g., result_reflect_optimized.json)
        doc_pool_manager: DocumentPoolManager instance for resolving doc IDs

    Returns:
        List of data items with doc_ids resolved to texts when needed
    """
    print(f"Loading optimized dataset from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"  Loaded {len(dataset)} samples")
    return dataset


if __name__ == "__main__":
    # Example usage
    doc_pool_path = "/home/shm/document/exp/FusionRAG/data/2wiki_input.json"
    data_path = "/home/shm/document/exp/FusionRAG/data/2wikimqa_reflect_optimized.json"

    # Initialize document pool
    doc_pool = DocumentPoolManager(doc_pool_path)

    # Load optimized data
    dataset = load_optimized_data(data_path, doc_pool)

    # Example: Access first sample's gold documents
    sample = dataset[0]
    print(f"\nExample: First sample")
    print(f"Question: {sample['question']}")
    print(f"Gold doc IDs: {sample['gold_docs']}")
    print(f"\nGold documents:")
    for doc_id in sample['gold_docs']:
        text = doc_pool.get_doc_text(doc_id)
        print(f"  [Doc {doc_id}]: {text[:100]}...")
