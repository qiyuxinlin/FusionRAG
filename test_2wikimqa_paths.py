#!/usr/bin/env python3
"""
Quick test to verify 2wikimqa KV cache paths are correct
"""

import torch
import os

# Paths
cache_dir = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/2wikimqa"
sample_id = 0
chunk_id = 1

# Test no_preprocess path
no_prep_key_path = f"{cache_dir}/kv_cache/{sample_id}_{chunk_id}_key.pt"
no_prep_value_path = f"{cache_dir}/kv_cache/{sample_id}_{chunk_id}_value.pt"

# Test BGE path
bge_key_path = f"{cache_dir}/preprocess_kv_cache_global_topk10_bge/{sample_id}_{chunk_id}_key.pt"
bge_value_path = f"{cache_dir}/preprocess_kv_cache_global_topk10_bge/{sample_id}_{chunk_id}_value.pt"

print("Testing 2wikimqa KV cache paths...")
print("="*80)

# Check no_preprocess
print(f"\n1. No_preprocess KV cache:")
print(f"   Key path: {no_prep_key_path}")
print(f"   Exists: {os.path.exists(no_prep_key_path)}")

if os.path.exists(no_prep_key_path):
    key = torch.load(no_prep_key_path, map_location='cpu', weights_only=True)
    print(f"   Shape: {key.shape}")
    print(f"   Type: {key.dtype}")
else:
    print(f"   ✗ File not found!")

# Check BGE
print(f"\n2. BGE KV cache:")
print(f"   Key path: {bge_key_path}")
print(f"   Exists: {os.path.exists(bge_key_path)}")

if os.path.exists(bge_key_path):
    key = torch.load(bge_key_path, map_location='cpu', weights_only=True)
    print(f"   Shape: {key.shape}")
    print(f"   Type: {key.dtype}")
else:
    print(f"   ✗ File not found!")

print("\n" + "="*80)
print("✓ Path test complete!")
