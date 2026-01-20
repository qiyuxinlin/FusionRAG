#!/usr/bin/env python3
"""
Online KV Cache Manager for Dynamic RAG System

核心功能：
1. 动态生成和缓存文档的 KV Cache
2. 支持 LRU 内存管理
3. 支持持久化存储
4. 支持在线融合
"""

import os
import torch
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
from dataclasses import dataclass
from transformers import AutoTokenizer
from ktransformers.util.utils import prefill_and_save_kv_cache
from ktransformers.models.custom_cache import StaticCache


@dataclass
class KVCacheEntry:
    """KV Cache条目"""
    doc_id: str
    key_cache: torch.Tensor  # [num_layers, num_heads, seq_len, head_dim]
    value_cache: torch.Tensor
    doc_text: str
    doc_tokens: torch.Tensor
    timestamp: float
    hit_count: int = 0

    def __sizeof__(self):
        """估算内存占用"""
        key_size = self.key_cache.element_size() * self.key_cache.nelement()
        value_size = self.value_cache.element_size() * self.value_cache.nelement()
        return key_size + value_size


class KVCacheStore:
    """
    KV Cache存储管理器

    支持：
    - 内存缓存（LRU淘汰）
    - 磁盘持久化
    - 快速查询
    """

    def __init__(
        self,
        max_memory_gb: float = 10.0,
        disk_cache_dir: Optional[str] = None,
        enable_lru: bool = True
    ):
        """
        Args:
            max_memory_gb: 最大内存占用（GB）
            disk_cache_dir: 磁盘缓存目录（None表示仅内存）
            enable_lru: 是否启用LRU淘汰策略
        """
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.disk_cache_dir = disk_cache_dir
        self.enable_lru = enable_lru

        # 内存缓存：doc_id -> KVCacheEntry
        self.memory_cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self.current_memory_usage = 0

        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_loads': 0,
            'disk_saves': 0
        }

        if disk_cache_dir:
            os.makedirs(disk_cache_dir, exist_ok=True)

    def get_doc_id(self, doc_text: str) -> str:
        """根据文档内容生成唯一ID"""
        return hashlib.md5(doc_text.encode('utf-8')).hexdigest()

    def _get_disk_path(self, doc_id: str) -> Tuple[str, str]:
        """获取磁盘缓存路径"""
        if not self.disk_cache_dir:
            return None, None
        key_path = os.path.join(self.disk_cache_dir, f"{doc_id}_key.pt")
        value_path = os.path.join(self.disk_cache_dir, f"{doc_id}_value.pt")
        return key_path, value_path

    def _evict_lru(self):
        """淘汰最少使用的缓存"""
        if not self.memory_cache:
            return

        # 移除最旧的条目
        doc_id, entry = self.memory_cache.popitem(last=False)
        self.current_memory_usage -= entry.__sizeof__()
        self.stats['evictions'] += 1

        # 如果启用磁盘缓存，保存到磁盘
        if self.disk_cache_dir:
            self._save_to_disk(doc_id, entry)

        print(f"[KVCacheStore] Evicted {doc_id}, current memory: {self.current_memory_usage / 1024**3:.2f} GB")

    def _save_to_disk(self, doc_id: str, entry: KVCacheEntry):
        """保存到磁盘"""
        key_path, value_path = self._get_disk_path(doc_id)
        if not key_path:
            return

        torch.save(entry.key_cache.cpu(), key_path)
        torch.save(entry.value_cache.cpu(), value_path)
        self.stats['disk_saves'] += 1

    def _load_from_disk(self, doc_id: str) -> Optional[KVCacheEntry]:
        """从磁盘加载"""
        key_path, value_path = self._get_disk_path(doc_id)
        if not key_path or not os.path.exists(key_path):
            return None

        key_cache = torch.load(key_path, weights_only=True)
        value_cache = torch.load(value_path, weights_only=True)
        self.stats['disk_loads'] += 1

        # 注意：这里缺少doc_text和doc_tokens，需要额外存储或从其他地方获取
        # 简化处理：仅返回KV cache
        return KVCacheEntry(
            doc_id=doc_id,
            key_cache=key_cache,
            value_cache=value_cache,
            doc_text="",  # 从磁盘加载时无法恢复
            doc_tokens=torch.tensor([]),
            timestamp=time.time()
        )

    def get(self, doc_text: str) -> Optional[KVCacheEntry]:
        """
        获取KV Cache

        Returns:
            KVCacheEntry if exists, else None
        """
        doc_id = self.get_doc_id(doc_text)

        # 1. 检查内存缓存
        if doc_id in self.memory_cache:
            entry = self.memory_cache[doc_id]
            entry.hit_count += 1
            self.stats['hits'] += 1

            # LRU: 移到末尾
            if self.enable_lru:
                self.memory_cache.move_to_end(doc_id)

            return entry

        # 2. 检查磁盘缓存
        if self.disk_cache_dir:
            entry = self._load_from_disk(doc_id)
            if entry:
                # 加载到内存
                self._put_memory(doc_id, entry)
                self.stats['hits'] += 1
                return entry

        # 3. 未命中
        self.stats['misses'] += 1
        return None

    def _put_memory(self, doc_id: str, entry: KVCacheEntry):
        """放入内存缓存"""
        entry_size = entry.__sizeof__()

        # 确保有足够空间
        while (self.current_memory_usage + entry_size > self.max_memory_bytes
               and len(self.memory_cache) > 0):
            self._evict_lru()

        self.memory_cache[doc_id] = entry
        self.current_memory_usage += entry_size

    def put(
        self,
        doc_text: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        doc_tokens: torch.Tensor
    ):
        """
        存储KV Cache

        Args:
            doc_text: 文档文本
            key_cache: Key cache tensor
            value_cache: Value cache tensor
            doc_tokens: 文档tokens
        """
        doc_id = self.get_doc_id(doc_text)

        entry = KVCacheEntry(
            doc_id=doc_id,
            key_cache=key_cache,
            value_cache=value_cache,
            doc_text=doc_text,
            doc_tokens=doc_tokens,
            timestamp=time.time(),
            hit_count=0
        )

        self._put_memory(doc_id, entry)

    def clear(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        self.current_memory_usage = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'memory_usage_gb': self.current_memory_usage / 1024**3,
            'num_cached': len(self.memory_cache)
        }


class OnlineKVCacheManager:
    """
    在线KV Cache管理器

    核心功能：
    1. 根据Query召回文档
    2. 动态生成或加载KV Cache
    3. 支持在线融合
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        model_type: str = 'qwen2',
        bge_model_path: Optional[str] = None,
        kv_store: Optional[KVCacheStore] = None,
        device: str = "cuda:0",
        device_map: Optional[dict] = None,
        system_prompt: str = "You are a helpful assistant."
    ):
        """
        Args:
            model: 语言模型
            tokenizer: Tokenizer
            model_type: 模型类型
            bge_model_path: BGE模型路径（用于文档召回）
            kv_store: KV缓存存储（如果为None则创建默认）
            device: 设备
            device_map: 多GPU映射
            system_prompt: 系统提示词
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        self.device_map = device_map
        self.input_device = "cuda:0" if device_map is not None else device

        # 系统提示词
        self.system_prompt = system_prompt
        self.system_tokens = torch.tensor(
            tokenizer.encode(system_prompt, add_special_tokens=False),
            dtype=torch.int
        )
        self.system_len = self.system_tokens.shape[0]

        # KV缓存存储
        self.kv_store = kv_store if kv_store else KVCacheStore(
            max_memory_gb=10.0,
            disk_cache_dir=None,
            enable_lru=True
        )

        # BGE召回器（可选）
        self.bge_model = None
        if bge_model_path:
            from FlagEmbedding import FlagModel
            print(f"Loading BGE model from {bge_model_path}...")
            self.bge_model = FlagModel(bge_model_path, use_fp16=True)

    def _generate_kv_cache(
        self,
        doc_text: str,
        save_to_store: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为单个文档生成KV Cache

        Args:
            doc_text: 文档文本
            save_to_store: 是否保存到缓存库

        Returns:
            (key_cache, value_cache, doc_tokens)
        """
        # 1. Tokenize
        doc_tokens = torch.tensor(
            self.tokenizer.encode(doc_text, add_special_tokens=False),
            dtype=torch.int
        )

        # 2. 准备输入（system + doc）
        input_tokens = torch.cat([self.system_tokens, doc_tokens])
        input_tensor = input_tokens.unsqueeze(0).to(self.input_device)

        # 3. 创建StaticCache
        max_cache_len = input_tokens.shape[0] + 1000
        past_key_values = StaticCache(
            self.model.config,
            batch_size=1,
            max_cache_len=max_cache_len,
            device=self.input_device,
            dtype=self.model.config.torch_dtype
        )

        # 4. Forward pass生成KV
        with torch.no_grad():
            inputs_embeds = self.model.model.embed_tokens(input_tensor).to(self.input_device)
            cache_position = torch.arange(0, input_tokens.shape[0], device=self.input_device)

            _ = self.model(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True
            )

        # 5. 提取文档部分的KV（跳过system部分）
        doc_len = doc_tokens.shape[0]
        key_cache = torch.stack([
            cache.cpu() for cache in past_key_values.key_cache
        ])[:, :, :, self.system_len:self.system_len + doc_len, :]

        value_cache = torch.stack([
            cache.cpu() for cache in past_key_values.value_cache
        ])[:, :, :, self.system_len:self.system_len + doc_len, :]

        # 6. 保存到缓存库
        if save_to_store:
            self.kv_store.put(doc_text, key_cache, value_cache, doc_tokens)

        return key_cache, value_cache, doc_tokens

    def retrieve_documents(
        self,
        query: str,
        document_pool: List[str],
        topk: int = 5
    ) -> List[int]:
        """
        根据Query召回相关文档

        Args:
            query: 用户查询
            document_pool: 文档池
            topk: 召回数量

        Returns:
            召回的文档索引列表
        """
        if not self.bge_model:
            # 如果没有BGE模型，返回前topk个文档
            return list(range(min(topk, len(document_pool))))

        # 使用BGE计算相似度
        import faiss
        import numpy as np

        # 编码文档和查询
        doc_embeddings = self.bge_model.encode(document_pool)
        query_embedding = self.bge_model.encode_queries([query])

        # FAISS搜索
        dim = doc_embeddings.shape[-1]
        index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
        doc_embeddings = doc_embeddings.astype(np.float32)
        index.train(doc_embeddings)
        index.add(doc_embeddings)

        query_embedding = query_embedding.astype(np.float32)
        _, indices = index.search(query_embedding, k=topk)

        return indices[0].tolist()

    def process_query(
        self,
        query: str,
        document_pool: List[str],
        topk: int = 5,
        use_fusion: bool = False,
        revert_rope: bool = True,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        处理查询：召回文档 → 检查/生成KV Cache → 准备推理

        Args:
            query: 用户查询
            document_pool: 文档池
            topk: 召回文档数
            use_fusion: 是否使用融合（True=FusionRAG, False=直接拼接）
            revert_rope: 是否revert RoPE
            verbose: 是否打印详细信息

        Returns:
            (query_tokens, key_cache_list, value_cache_list)
        """
        start_time = time.time()

        # 1. 文档召回
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")

        retrieved_indices = self.retrieve_documents(query, document_pool, topk)
        retrieved_docs = [document_pool[idx] for idx in retrieved_indices]

        if verbose:
            print(f"\n[Step 1] Retrieved {len(retrieved_docs)} documents:")
            for i, idx in enumerate(retrieved_indices):
                print(f"  {i+1}. Doc {idx}: {retrieved_docs[i][:80]}...")

        # 2. 检查缓存并生成/加载KV Cache
        key_cache_list = []
        value_cache_list = []
        doc_tokens_list = []

        cache_hits = 0
        cache_misses = 0

        if verbose:
            print(f"\n[Step 2] Processing KV Caches:")

        for i, doc_text in enumerate(retrieved_docs):
            # 检查缓存
            cache_entry = self.kv_store.get(doc_text)

            if cache_entry:
                # Case B: 缓存命中 - 直接加载
                cache_hits += 1
                key_cache = cache_entry.key_cache
                value_cache = cache_entry.value_cache
                doc_tokens = cache_entry.doc_tokens

                if verbose:
                    print(f"  ✓ Doc {i+1}: Cache HIT (hit_count={cache_entry.hit_count})")
            else:
                # Case A: 缓存未命中 - 生成并缓存
                cache_misses += 1
                if verbose:
                    print(f"  ✗ Doc {i+1}: Cache MISS - Generating...", end=" ")

                gen_start = time.time()
                key_cache, value_cache, doc_tokens = self._generate_kv_cache(doc_text)
                gen_time = time.time() - gen_start

                if verbose:
                    print(f"Done ({gen_time:.2f}s)")

            key_cache_list.append(key_cache)
            value_cache_list.append(value_cache)
            doc_tokens_list.append(doc_tokens)

        # 3. 准备查询tokens
        query_tokens = torch.tensor(
            self.tokenizer.encode(query, add_special_tokens=False),
            dtype=torch.int
        )

        total_time = time.time() - start_time

        if verbose:
            print(f"\n[Step 3] Summary:")
            print(f"  Cache hits: {cache_hits}/{len(retrieved_docs)}")
            print(f"  Cache misses: {cache_misses}/{len(retrieved_docs)}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Cache stats: {self.kv_store.get_stats()}")

        return query_tokens, key_cache_list, value_cache_list, doc_tokens_list

    def generate_answer(
        self,
        query_tokens: torch.Tensor,
        key_cache_list: List[torch.Tensor],
        value_cache_list: List[torch.Tensor],
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        使用KV Cache生成答案

        Args:
            query_tokens: 查询tokens
            key_cache_list: Key cache列表
            value_cache_list: Value cache列表
            max_new_tokens: 最大生成tokens数
            temperature: 生成温度

        Returns:
            生成的答案文本
        """
        # 1. 合并所有KV cache
        total_len = self.system_len + sum(kv.shape[3] for kv in key_cache_list)
        max_cache_len = total_len + query_tokens.shape[0] + max_new_tokens + 100

        past_key_values = StaticCache(
            self.model.config,
            batch_size=1,
            max_cache_len=max_cache_len,
            device=self.input_device,
            dtype=self.model.config.torch_dtype
        )

        # 2. 加载system KV (需要预先生成并缓存)
        # TODO: 这里简化处理，实际应该缓存system的KV

        # 3. 依次加载文档KV
        past_len = self.system_len
        for key_cache, value_cache in zip(key_cache_list, value_cache_list):
            doc_len = key_cache.shape[3]
            key_cache = key_cache.to(self.input_device)
            value_cache = value_cache.to(self.input_device)

            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx].narrow(2, past_len, doc_len).copy_(
                    key_cache[layer_idx]
                )
                past_key_values.value_cache[layer_idx].narrow(2, past_len, doc_len).copy_(
                    value_cache[layer_idx]
                )
                past_key_values.past_tokens[layer_idx] += doc_len

            past_len += doc_len

        # 4. 生成答案
        input_tensor = query_tokens.unsqueeze(0).to(self.input_device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


def example_usage():
    """使用示例"""
    from transformers import AutoModelForCausalLM, AutoConfig

    # 1. 加载模型
    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
    config = AutoConfig.from_pretrained(model_path)
    config.torch_dtype = torch.float16
    config._attn_implementation = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. 创建KV Cache管理器
    kv_store = KVCacheStore(
        max_memory_gb=5.0,
        disk_cache_dir="/tmp/kv_cache",
        enable_lru=True
    )

    manager = OnlineKVCacheManager(
        model=model,
        tokenizer=tokenizer,
        model_type='qwen2',
        bge_model_path="/mnt/data/models/bge-m3-FP16",
        kv_store=kv_store,
        device="cuda:0"
    )

    # 3. 准备文档池
    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "Python is a high-level programming language.",
        "The Great Wall of China is over 13,000 miles long.",
        "Machine learning is a subset of artificial intelligence.",
        "Tokyo is the capital city of Japan."
    ]

    # 4. 处理查询
    query = "Where is the Eiffel Tower?"

    query_tokens, key_list, value_list, doc_tokens_list = manager.process_query(
        query=query,
        document_pool=documents,
        topk=3,
        verbose=True
    )

    # 5. 生成答案
    answer = manager.generate_answer(
        query_tokens=query_tokens,
        key_cache_list=key_list,
        value_cache_list=value_list,
        max_new_tokens=50
    )

    print(f"\nAnswer: {answer}")

    # 6. 第二次查询（应该有缓存命中）
    print("\n" + "="*60)
    print("Second Query (should hit cache):")
    print("="*60)

    query2 = "Tell me about programming languages"
    query_tokens2, key_list2, value_list2, doc_tokens_list2 = manager.process_query(
        query=query2,
        document_pool=documents,
        topk=3,
        verbose=True
    )


if __name__ == "__main__":
    example_usage()
