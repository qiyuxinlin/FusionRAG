#!/usr/bin/env python3
"""
KV Cache Manager V3 - Global Document ID based KV cache management

Key changes from original:
1. Cache saved as: doc_{doc_id}_key.pt instead of {example_id}_{chunk_id}_key.pt
2. Enables cross-question document reuse
3. Tracks which documents have been cached globally
"""

import os
import torch
from typing import Dict, List, Set


class KVCacheManagerV3:
    """
    Manages KV cache using global document IDs for cross-question reuse.
    """

    def __init__(self, cache_root_dir: str):
        """
        Initialize KV cache manager.

        Args:
            cache_root_dir: Root directory for KV cache storage
        """
        self.cache_root_dir = cache_root_dir
        self.kv_cache_dir = os.path.join(cache_root_dir, 'kv_cache_v3')
        os.makedirs(self.kv_cache_dir, exist_ok=True)

        # Track cached documents
        self.cached_doc_ids: Set[int] = set()
        self._scan_existing_cache()

    def _scan_existing_cache(self):
        """Scan existing cache directory to find already cached documents."""
        if not os.path.exists(self.kv_cache_dir):
            return

        for filename in os.listdir(self.kv_cache_dir):
            if filename.startswith('doc_') and filename.endswith('_key.pt'):
                # Extract doc_id from filename: doc_{doc_id}_key.pt
                try:
                    doc_id = int(filename.split('_')[1])
                    self.cached_doc_ids.add(doc_id)
                except (IndexError, ValueError):
                    continue

        print(f"Found {len(self.cached_doc_ids)} cached documents")

    def get_doc_cache_path(self, doc_id: int, cache_type: str = 'key') -> str:
        """
        Get cache file path for a document.

        Args:
            doc_id: Global document ID
            cache_type: 'key' or 'value'

        Returns:
            Path to cache file
        """
        return os.path.join(self.kv_cache_dir, f'doc_{doc_id}_{cache_type}.pt')

    def is_cached(self, doc_id: int) -> bool:
        """Check if a document's KV cache exists."""
        key_path = self.get_doc_cache_path(doc_id, 'key')
        value_path = self.get_doc_cache_path(doc_id, 'value')
        return os.path.exists(key_path) and os.path.exists(value_path)

    def save_kv_cache(self, doc_id: int, key_cache: torch.Tensor, value_cache: torch.Tensor):
        """
        Save KV cache for a document.

        Args:
            doc_id: Global document ID
            key_cache: Key cache tensor
            value_cache: Value cache tensor
        """
        key_path = self.get_doc_cache_path(doc_id, 'key')
        value_path = self.get_doc_cache_path(doc_id, 'value')

        torch.save(key_cache, key_path)
        torch.save(value_cache, value_path)
        self.cached_doc_ids.add(doc_id)

    def load_kv_cache(self, doc_id: int) -> tuple:
        """
        Load KV cache for a document.

        Args:
            doc_id: Global document ID

        Returns:
            (key_cache, value_cache) tuple

        Raises:
            FileNotFoundError: If cache doesn't exist
        """
        key_path = self.get_doc_cache_path(doc_id, 'key')
        value_path = self.get_doc_cache_path(doc_id, 'value')

        if not self.is_cached(doc_id):
            raise FileNotFoundError(f"KV cache not found for doc_id={doc_id}")

        key_cache = torch.load(key_path, weights_only=True)
        value_cache = torch.load(value_path, weights_only=True)

        return key_cache, value_cache

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_size = 0
        if os.path.exists(self.kv_cache_dir):
            for filename in os.listdir(self.kv_cache_dir):
                filepath = os.path.join(self.kv_cache_dir, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)

        return {
            'num_cached_docs': len(self.cached_doc_ids),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': self.kv_cache_dir,
        }

    def clear_cache(self):
        """Clear all cached KV."""
        import shutil
        if os.path.exists(self.kv_cache_dir):
            shutil.rmtree(self.kv_cache_dir)
            os.makedirs(self.kv_cache_dir, exist_ok=True)
            self.cached_doc_ids.clear()
            print("Cache cleared")


def example_usage_in_main():
    """
    Example showing how to integrate KVCacheManagerV3 into main() function.

    This replaces the old approach:
        OLD: cache_key_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'
        NEW: cache_key_path = kv_cache_manager.get_doc_cache_path(doc_id, 'key')
    """

    # Initialize cache manager
    cache_root = "/mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/musique"
    kv_cache_manager = KVCacheManagerV3(cache_root)

    # Example: Check if document needs caching
    doc_id = 174  # Global doc ID from document pool

    if not kv_cache_manager.is_cached(doc_id):
        print(f"Document {doc_id} not cached, generating KV...")

        # Generate KV cache (pseudo code)
        # key_cache, value_cache = generate_kv_for_document(doc_id)
        # kv_cache_manager.save_kv_cache(doc_id, key_cache, value_cache)
    else:
        print(f"Document {doc_id} already cached, loading...")
        key_cache, value_cache = kv_cache_manager.load_kv_cache(doc_id)
        print(f"Loaded cache shapes: key={key_cache.shape}, value={value_cache.shape}")

    # Get cache statistics
    stats = kv_cache_manager.get_cache_stats()
    print(f"\nCache stats:")
    print(f"  Cached documents: {stats['num_cached_docs']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")


def modified_kv_generation_loop():
    """
    Example showing the modified KV generation loop in main().

    OLD approach:
        for doc_idx, doc_tensor in enumerate(doc_tensors):
            chunk_id = doc_idx + 1
            cache_key_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'
            if not os.path.exists(cache_key_path):
                # Generate KV...

    NEW approach:
        for doc_id, doc_tensor in zip(question_doc_ids, doc_tensors):
            if not kv_cache_manager.is_cached(doc_id):
                # Generate KV and save with global doc_id
    """

    # Pseudo code for the new loop
    print("""
    # NEW: Generate KV cache for documents using global doc IDs
    for doc_id, doc_tensor in zip(q_data['doc_ids'], q_data['doc_tensors']):
        if not kv_cache_manager.is_cached(doc_id):
            print(f"  Generating KV cache for doc_id={doc_id}...")

            # Generate KV cache
            passage_len = doc_tensor.shape[0]
            input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)

            # prefill_and_save_kv_cache modified to accept doc_id instead of (example_id, chunk_id)
            prefill_and_save_kv_cache_v3(
                model, tokenizer, past_key_values, input_tensor.to(input_device),
                kv_cache_manager=kv_cache_manager,
                doc_id=doc_id,  # NEW: Use global doc_id
                system_len=system_len,
                passage_len=passage_len,
                reprocess_method=reprocess_method,
                device=input_device,
                device_map=device_map
            )
        else:
            print(f"  Doc_id={doc_id} already cached, skipping")
    """)


def modified_kv_loading_loop():
    """
    Example showing the modified KV loading loop during preprocessing.

    OLD approach:
        similar_cache_key_path = f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt"
        key_cache = torch.load(similar_cache_key_path)

    NEW approach:
        key_cache, value_cache = kv_cache_manager.load_kv_cache(similar_doc_id)
    """

    print("""
    # NEW: Load KV cache using global doc IDs
    for similar_doc_id in similar_doc_ids[:topk]:
        if not kv_cache_manager.is_cached(similar_doc_id):
            # On-demand generation
            print(f"  On-demand: Generating cache for doc_id={similar_doc_id}...")
            # Generate and save...

        # Load cache
        key_cache, value_cache = kv_cache_manager.load_kv_cache(similar_doc_id)

        # Copy to past_key_values
        doc_len = key_cache.shape[2]
        for layer_idx in range(num_layers):
            past_key_values.key_cache[layer_idx].narrow(2, past_len, doc_len).copy_(
                key_cache[layer_idx]
            )
            past_key_values.value_cache[layer_idx].narrow(2, past_len, doc_len).copy_(
                value_cache[layer_idx]
            )
        past_len += doc_len
    """)


if __name__ == "__main__":
    print("="*80)
    print("KV Cache Manager V3 - Global Document ID Demo")
    print("="*80)
    print()

    example_usage_in_main()

    print("\n" + "="*80)
    print("Modified KV Generation Loop")
    print("="*80)
    modified_kv_generation_loop()

    print("\n" + "="*80)
    print("Modified KV Loading Loop")
    print("="*80)
    modified_kv_loading_loop()
