#!/usr/bin/env python3
"""
独立的KV融合预处理模块

从 analyse_data.py 抽离出来的核心KV融合处理逻辑。
用于研究不同预处理方式对KV cache的影响。

主要功能:
1. 生成文档的独立KV cache
2. 基于相似度召回相关文档
3. 融合多个文档的KV cache
4. 保存融合后的preprocessed KV cache

支持的融合方法:
- BGE: 基于BGE模型的语义相似度召回
- RANDOM: 随机采样文档
- REPEAT_SELF: 重复当前文档K次
- FIXED_DOC: 使用固定文档进行召回
- RANDOM_TEXT: BGE召回长度,但融合随机无关文本KV
- BGE_SHUFFLED: BGE召回,但打乱召回文档的KV位置
- RANDOM_DOCS: 随机选择无关文档,使用原始KV长度
- NO_PREPROCESS_WITH_BIAS: 使用no_preprocess KV + 分布对齐
"""

import os
import json
import torch
import shutil
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from transformers import AutoTokenizer, AutoConfig
from FlagEmbedding import BGEM3FlagModel
import torch.nn.functional as F


class RecallMethod(Enum):
    """文档召回方法"""
    BGE = "bge"                                      # BGE语义相似度召回
    RANDOM = "random"                                 # 随机采样
    REPEAT_SELF = "repeat_self"                       # 重复自身
    FIXED_DOC = "fixed_doc"                           # 固定文档
    RANDOM_TEXT = "random_text"                       # 随机文本(调整长度)
    BGE_SHUFFLED = "bge_shuffled"                     # BGE召回+打乱KV
    RANDOM_DOCS = "random_docs"                       # 随机文档(原始长度)
    NO_PREPROCESS_WITH_BIAS = "no_preprocess_with_bias"  # 分布对齐


class PreprocessScope(Enum):
    """文档检索范围"""
    GLOBAL = "global"              # 全局检索
    PER_EXAMPLE = "per_example"    # 单例检索
    SKIP_UNTESTED = "skip_untested" # 跳过未测试样本


def rotate_half(x):
    """旋转embedding的一半维度 (用于RoPE)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def find_group_and_index(corpus_lens: List[int], global_idx: int) -> Tuple[int, int]:
    """
    根据全局索引找到对应的问题组和文档索引

    Args:
        corpus_lens: 每个问题的文档数量列表
        global_idx: 全局文档索引

    Returns:
        (group_idx, doc_idx): 问题组索引和该组内的文档索引
    """
    cumsum = 0
    for group_idx, length in enumerate(corpus_lens):
        if global_idx < cumsum + length:
            return group_idx, global_idx - cumsum
        cumsum += length
    raise ValueError(f"Global index {global_idx} out of range")


class KVCacheFusion:
    """KV Cache融合处理器"""

    def __init__(
        self,
        model,
        tokenizer,
        model_type: str,
        device: str = "cuda:0",
        device_map: Optional[dict] = None
    ):
        """
        初始化KV融合处理器

        Args:
            model: 语言模型
            tokenizer: tokenizer
            model_type: 模型类型 ('qwen', 'qwen2', 'mistral', 'llama', etc.)
            device: 计算设备
            device_map: 多GPU设备映射
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        self.device_map = device_map
        self.input_device = "cuda:0" if device_map is not None else device

    def generate_independent_kv_cache(
        self,
        input_tensor: torch.Tensor,
        save_path: str,
        cache_id: str,
        system_len: int = 0,
        passage_len: Optional[int] = None,
        reprocess_method: Optional[str] = None
    ):
        """
        生成单个文档的独立KV cache

        Args:
            input_tensor: 输入token tensor [seq_len]
            save_path: KV cache保存路径
            cache_id: cache标识符 (如 "example_0_chunk_1")
            system_len: system prompt长度
            passage_len: passage长度 (如果为None则自动计算)
            reprocess_method: 重处理方法 (如 "Cache-Craft")
        """
        from ktransformers.util.utils import prefill_and_save_kv_cache
        from ktransformers.models.custom_cache import StaticCache

        if passage_len is None:
            passage_len = input_tensor.shape[0] - system_len

        # 创建StaticCache
        config = self.model.config
        batch_size = 1
        max_cache_len = input_tensor.shape[0] + 1000  # 预留生成空间

        past_key_values = StaticCache(
            config, batch_size, max_cache_len,
            self.input_device, config.torch_dtype
        )

        # 生成并保存KV cache
        prefill_and_save_kv_cache(
            self.model, self.tokenizer, past_key_values,
            input_tensor.unsqueeze(0).to(self.input_device),
            save_path=save_path,
            example_id=cache_id.split('_')[0] if '_' in cache_id else cache_id,
            chunk_id=int(cache_id.split('_')[-1]) if '_' in cache_id else 0,
            system_len=system_len,
            passage_len=passage_len,
            reprocess_method=reprocess_method,
            device=self.input_device,
            device_map=self.device_map
        )

    def fuse_kv_caches(
        self,
        current_doc_tensor: torch.Tensor,
        similar_doc_info: List[Dict[str, Any]],
        system_tensor: torch.Tensor,
        kv_cache_dir: str,
        output_path: str,
        output_id: str,
        topk: int = 3,
        recall_method: RecallMethod = RecallMethod.BGE,
        revert_rope: bool = False,
        reprocess_method: Optional[str] = None,
        shuffle_seed: Optional[int] = None
    ):
        """
        融合多个文档的KV cache

        Args:
            current_doc_tensor: 当前文档的token tensor
            similar_doc_info: 相似文档信息列表,每个元素包含:
                - 'cache_path_prefix': KV cache路径前缀 (不含_key.pt/_value.pt)
                - 'doc_tensor': 文档的token tensor
                - 'is_random_text': (可选) 是否为随机文本
            system_tensor: system prompt的token tensor
            kv_cache_dir: KV cache目录
            output_path: 输出路径
            output_id: 输出cache ID
            topk: 融合的文档数量
            recall_method: 召回方法
            revert_rope: 是否revert RoPE
            reprocess_method: 重处理方法
            shuffle_seed: 打乱种子 (用于BGE_SHUFFLED模式)
        """
        from ktransformers.util.utils import prefill_with_cache_and_save_preprocess
        from ktransformers.models.custom_cache import StaticCache
        from transformers import TextStreamer

        config = self.model.config
        batch_size = 1
        system_len = system_tensor.shape[0]

        # 计算最大cache长度
        max_cache_len = system_len
        for doc_info in similar_doc_info[:topk]:
            max_cache_len += doc_info['doc_tensor'].shape[0]
        max_cache_len += current_doc_tensor.shape[0] + 1000

        # 创建StaticCache
        past_key_values = StaticCache(
            config, batch_size, max_cache_len,
            self.input_device, config.torch_dtype
        )

        # 1. 加载system prompt的KV cache
        system_key = torch.load(f"{kv_cache_dir}/system_0_key.pt", weights_only=True)
        system_value = torch.load(f"{kv_cache_dir}/system_0_value.pt", weights_only=True)

        for layer_idx in range(len(past_key_values.key_cache)):
            seq_len = system_key[layer_idx].shape[1]
            past_key_values.key_cache[layer_idx].narrow(2, 0, seq_len).copy_(system_key[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2, 0, seq_len).copy_(system_value[layer_idx])
            past_key_values.past_tokens[layer_idx] = seq_len

        past_len = system_len
        corpus_passages = [system_tensor]

        # 2. 加载并融合相似文档的KV cache
        if shuffle_seed is not None:
            torch.manual_seed(shuffle_seed)
            np.random.seed(shuffle_seed)

        for idx, doc_info in enumerate(similar_doc_info[:topk]):
            cache_path_prefix = doc_info['cache_path_prefix']
            doc_tensor = doc_info['doc_tensor']
            is_random_text = doc_info.get('is_random_text', False)

            # 加载KV cache
            chunk_key_cache = torch.load(f"{cache_path_prefix}_key.pt", weights_only=True)
            chunk_value_cache = torch.load(f"{cache_path_prefix}_value.pt", weights_only=True)

            corpus_len = doc_tensor.shape[0]

            # 特殊处理: BGE_SHUFFLED模式打乱KV位置
            if recall_method == RecallMethod.BGE_SHUFFLED and not is_random_text:
                seq_len = chunk_key_cache[0].shape[1]
                shuffle_indices = torch.randperm(seq_len)

                for layer_idx in range(len(chunk_key_cache)):
                    chunk_key_cache[layer_idx] = chunk_key_cache[layer_idx][:, shuffle_indices, :]
                    chunk_value_cache[layer_idx] = chunk_value_cache[layer_idx][:, shuffle_indices, :]

            # 拷贝到past_key_values
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(
                    chunk_key_cache[layer_idx]
                )
                past_key_values.value_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(
                    chunk_value_cache[layer_idx]
                )
                past_key_values.past_tokens[layer_idx] += corpus_len

            past_len += corpus_len
            corpus_passages.append(doc_tensor)

        # 3. 添加当前文档并生成融合后的KV cache
        corpus_passages.append(current_doc_tensor)

        # 使用prefill_with_cache_and_save_preprocess进行最终的融合预处理
        prefill_with_cache_and_save_preprocess(
            self.model, self.tokenizer, past_key_values, corpus_passages,
            output_path,
            example_id=output_id.split('_')[0] if '_' in output_id else output_id,
            chunk_id=int(output_id.split('_')[-1]) if '_' in output_id else 0,
            system_len=system_len,
            revert_rope=revert_rope,
            reprocess_method=reprocess_method,
            device=self.input_device,
            device_map=self.device_map
        )

    def apply_distribution_steering(
        self,
        no_preprocess_kv_path_prefix: str,
        output_path_prefix: str,
        kv_distribution_stats: Dict[str, Any],
        steering_alpha: float = 1.0,
        key_layers: str = "all",
        value_layers: str = "all",
        use_per_head: bool = False
    ):
        """
        应用分布对齐steering vector (NO_PREPROCESS_WITH_BIAS方法)

        Args:
            no_preprocess_kv_path_prefix: no_preprocess KV cache路径前缀
            output_path_prefix: 输出路径前缀
            kv_distribution_stats: KV分布统计数据
            steering_alpha: steering强度系数
            key_layers: 应用key steering的层 (如 "all", "0-10", "0,5,10")
            value_layers: 应用value steering的层
            use_per_head: 是否使用per-head steering
        """
        from analyse_data import parse_layer_selection, apply_steering_vector, apply_per_head_steering_vector

        # 加载no_preprocess KV cache
        no_prep_key = torch.load(f"{no_preprocess_kv_path_prefix}_key.pt", weights_only=True, map_location='cpu')
        no_prep_value = torch.load(f"{no_preprocess_kv_path_prefix}_value.pt", weights_only=True, map_location='cpu')

        # 解析层选择
        num_layers = len(no_prep_key)
        key_layers_to_apply = parse_layer_selection(key_layers, max_layers=num_layers)
        value_layers_to_apply = parse_layer_selection(value_layers, max_layers=num_layers)

        # 检查steering数据格式
        use_steering = 'key_steering' in kv_distribution_stats

        if use_steering and use_per_head:
            # 检查是否有per-head数据
            first_layer = list(kv_distribution_stats['key_steering'].keys())[0]
            if 'per_head' not in kv_distribution_stats['key_steering'][first_layer]:
                use_per_head = False
                print("Warning: Per-head data not found, falling back to layer-level steering")

        steered_key = []
        steered_value = []

        for layer_idx in range(num_layers):
            layer_name = f'layer_{layer_idx}'

            if use_steering:
                # 新的steering vector格式
                if layer_name in kv_distribution_stats['key_steering']:
                    apply_key = layer_idx in key_layers_to_apply
                    apply_value = layer_idx in value_layers_to_apply

                    if use_per_head:
                        # Per-head steering
                        key_result = apply_per_head_steering_vector(
                            no_prep_key[layer_idx],
                            kv_distribution_stats['key_steering'][layer_name],
                            steering_alpha,
                            apply_steering=apply_key
                        )
                        value_result = apply_per_head_steering_vector(
                            no_prep_value[layer_idx],
                            kv_distribution_stats['value_steering'][layer_name],
                            steering_alpha,
                            apply_steering=apply_value
                        )
                    else:
                        # Layer-level steering
                        key_steering = kv_distribution_stats['key_steering'][layer_name]['layer_level']
                        value_steering = kv_distribution_stats['value_steering'][layer_name]['layer_level']

                        key_result = apply_steering_vector(
                            no_prep_key[layer_idx],
                            key_steering,
                            steering_alpha,
                            apply_steering=apply_key
                        )
                        value_result = apply_steering_vector(
                            no_prep_value[layer_idx],
                            value_steering,
                            steering_alpha,
                            apply_steering=apply_value
                        )

                    steered_key.append(key_result)
                    steered_value.append(value_result)
                else:
                    # 层不在统计数据中,保持原样
                    steered_key.append(no_prep_key[layer_idx])
                    steered_value.append(no_prep_value[layer_idx])
            else:
                # 旧的BatchNorm格式 (scale/bias)
                if layer_name in kv_distribution_stats['key_scale']:
                    key_scale = kv_distribution_stats['key_scale'][layer_name]
                    key_bias = kv_distribution_stats['key_bias'][layer_name]
                    value_scale = kv_distribution_stats['value_scale'][layer_name]
                    value_bias = kv_distribution_stats['value_bias'][layer_name]

                    # BatchNorm: (x - bias) / scale
                    normalized_key = (no_prep_key[layer_idx] - key_bias) / (key_scale + 1e-6)
                    normalized_value = (no_prep_value[layer_idx] - value_bias) / (value_scale + 1e-6)

                    steered_key.append(normalized_key)
                    steered_value.append(normalized_value)
                else:
                    steered_key.append(no_prep_key[layer_idx])
                    steered_value.append(no_prep_value[layer_idx])

        # 保存steered KV cache
        os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
        torch.save(steered_key, f"{output_path_prefix}_key.pt")
        torch.save(steered_value, f"{output_path_prefix}_value.pt")


class DocumentRetriever:
    """文档召回器"""

    def __init__(
        self,
        bge_model_path: Optional[str] = None,
        recall_method: RecallMethod = RecallMethod.BGE,
        random_seed: int = 42
    ):
        """
        初始化文档召回器

        Args:
            bge_model_path: BGE模型路径 (如果使用BGE召回)
            recall_method: 召回方法
            random_seed: 随机种子
        """
        self.recall_method = recall_method
        self.random_seed = random_seed

        # 加载BGE模型 (如果需要)
        if recall_method in [RecallMethod.BGE, RecallMethod.BGE_SHUFFLED, RecallMethod.RANDOM_TEXT]:
            if bge_model_path is None:
                raise ValueError(f"BGE model path required for recall method: {recall_method}")
            self.bge_model = BGEM3FlagModel(bge_model_path, use_fp16=True)
        else:
            self.bge_model = None

        # 设置随机种子
        if recall_method == RecallMethod.RANDOM:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def compute_similarity_matrix(
        self,
        documents: List[str],
        preprocess_scope: PreprocessScope = PreprocessScope.GLOBAL,
        corpus_lens: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        计算文档相似度矩阵

        Args:
            documents: 文档列表
            preprocess_scope: 检索范围
            corpus_lens: 每个问题的文档数量 (用于PER_EXAMPLE模式)

        Returns:
            similarity_matrix: [num_docs, num_docs] 相似度矩阵
        """
        if self.bge_model is None:
            raise ValueError("BGE model not loaded")

        # 编码文档
        embeddings = self.bge_model.encode(
            documents,
            batch_size=32,
            max_length=8192
        )['dense_vecs']

        # 归一化
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 计算相似度矩阵
        similarity_matrix = np.matmul(embeddings, embeddings.T)

        # 根据scope调整相似度矩阵
        if preprocess_scope == PreprocessScope.PER_EXAMPLE and corpus_lens is not None:
            # PER_EXAMPLE: 只保留同一问题内的相似度
            mask = np.zeros_like(similarity_matrix)
            offset = 0
            for length in corpus_lens:
                mask[offset:offset+length, offset:offset+length] = 1.0
                offset += length
            similarity_matrix = similarity_matrix * mask

        return similarity_matrix

    def retrieve_similar_documents(
        self,
        query_idx: int,
        similarity_matrix: np.ndarray,
        topk: int = 3,
        exclude_self: bool = True
    ) -> List[int]:
        """
        检索相似文档

        Args:
            query_idx: 查询文档索引
            similarity_matrix: 相似度矩阵
            topk: 返回top-k文档
            exclude_self: 是否排除自身

        Returns:
            similar_indices: 相似文档索引列表
        """
        if self.recall_method == RecallMethod.BGE or self.recall_method == RecallMethod.BGE_SHUFFLED:
            # BGE相似度召回
            similarities = similarity_matrix[query_idx]

            if exclude_self:
                similarities[query_idx] = -1.0  # 排除自身

            # 获取top-k
            similar_indices = np.argsort(similarities)[::-1][:topk].tolist()

        elif self.recall_method == RecallMethod.RANDOM:
            # 随机召回
            all_indices = list(range(len(similarity_matrix)))
            if exclude_self:
                all_indices.remove(query_idx)
            similar_indices = np.random.choice(all_indices, size=min(topk, len(all_indices)), replace=False).tolist()

        elif self.recall_method == RecallMethod.REPEAT_SELF:
            # 重复自身
            similar_indices = [query_idx] * topk

        elif self.recall_method == RecallMethod.FIXED_DOC:
            # 固定文档 (需要在外部指定)
            raise NotImplementedError("FIXED_DOC method requires external specification")

        else:
            raise ValueError(f"Unsupported recall method: {self.recall_method}")

        return similar_indices


def preprocess_documents_pipeline(
    model,
    tokenizer,
    model_type: str,
    documents: List[Dict[str, Any]],
    system_prompt: str,
    output_dir: str,
    bge_model_path: Optional[str] = None,
    topk: int = 3,
    recall_method: RecallMethod = RecallMethod.BGE,
    preprocess_scope: PreprocessScope = PreprocessScope.GLOBAL,
    revert_rope: bool = False,
    reprocess_method: Optional[str] = None,
    device: str = "cuda:0",
    device_map: Optional[dict] = None,
    random_seed: int = 42,
    kv_distribution_stats: Optional[Dict] = None,
    steering_alpha: float = 1.0
):
    """
    完整的文档预处理pipeline

    Args:
        model: 语言模型
        tokenizer: tokenizer
        model_type: 模型类型
        documents: 文档列表,每个元素包含:
            - 'text': 文档文本
            - 'id': 文档ID (可选)
        system_prompt: system prompt文本
        output_dir: 输出目录
        bge_model_path: BGE模型路径
        topk: 融合top-k文档
        recall_method: 召回方法
        preprocess_scope: 检索范围
        revert_rope: 是否revert RoPE
        reprocess_method: 重处理方法
        device: 计算设备
        device_map: 多GPU设备映射
        random_seed: 随机种子
        kv_distribution_stats: KV分布统计 (用于NO_PREPROCESS_WITH_BIAS)
        steering_alpha: steering强度

    Returns:
        result: 包含处理结果的字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建子目录
    no_preprocess_dir = os.path.join(output_dir, "no_preprocess")
    preprocess_dir = os.path.join(output_dir, f"preprocess_topk{topk}_{recall_method.value}")
    os.makedirs(no_preprocess_dir, exist_ok=True)
    os.makedirs(preprocess_dir, exist_ok=True)

    # 初始化KV融合器
    kv_fusion = KVCacheFusion(model, tokenizer, model_type, device, device_map)

    # Tokenize documents
    print("Tokenizing documents...")
    system_tensor = tokenizer.encode(system_prompt, return_tensors='pt')[0]
    doc_tensors = []
    doc_texts = []

    for doc in documents:
        doc_text = doc['text']
        doc_tensor = tokenizer.encode(doc_text, return_tensors='pt')[0]
        doc_tensors.append(doc_tensor)
        doc_texts.append(doc_text)

    # Step 1: 生成独立的KV cache (no_preprocess)
    print("\nStep 1: Generating independent KV caches...")

    # 生成system cache
    kv_fusion.generate_independent_kv_cache(
        system_tensor,
        no_preprocess_dir,
        "system_0",
        system_len=system_tensor.shape[0]
    )

    # 生成每个文档的独立cache
    for doc_idx, doc_tensor in enumerate(doc_tensors):
        cache_id = f"doc_{doc_idx}"
        input_tensor = torch.cat([system_tensor, doc_tensor])

        kv_fusion.generate_independent_kv_cache(
            input_tensor,
            no_preprocess_dir,
            cache_id,
            system_len=system_tensor.shape[0],
            passage_len=doc_tensor.shape[0],
            reprocess_method=reprocess_method
        )
        print(f"  Generated cache for document {doc_idx+1}/{len(doc_tensors)}")

    # Step 2: 文档召回
    print(f"\nStep 2: Document retrieval using {recall_method.value} method...")

    retriever = DocumentRetriever(bge_model_path, recall_method, random_seed)

    # 计算相似度矩阵 (如果需要)
    similarity_matrix = None
    if recall_method in [RecallMethod.BGE, RecallMethod.BGE_SHUFFLED, RecallMethod.RANDOM_TEXT]:
        corpus_lens = [len(documents)]  # 单个问题的情况
        similarity_matrix = retriever.compute_similarity_matrix(
            doc_texts, preprocess_scope, corpus_lens
        )

    # Step 3: KV融合
    print(f"\nStep 3: Fusing KV caches (top-{topk})...")

    for doc_idx in range(len(doc_tensors)):
        print(f"  Processing document {doc_idx+1}/{len(doc_tensors)}...")

        # 检索相似文档
        if recall_method == RecallMethod.REPEAT_SELF:
            similar_indices = [doc_idx] * topk
        elif recall_method in [RecallMethod.BGE, RecallMethod.BGE_SHUFFLED]:
            similar_indices = retriever.retrieve_similar_documents(
                doc_idx, similarity_matrix, topk, exclude_self=True
            )
        elif recall_method == RecallMethod.RANDOM:
            all_indices = list(range(len(doc_tensors)))
            all_indices.remove(doc_idx)
            similar_indices = np.random.choice(all_indices, size=min(topk, len(all_indices)), replace=False).tolist()
        else:
            similar_indices = []

        print(f"    Similar docs: {similar_indices}")

        # 准备相似文档信息
        similar_doc_info = []
        for sim_idx in similar_indices:
            similar_doc_info.append({
                'cache_path_prefix': f"{no_preprocess_dir}/doc_{sim_idx}",
                'doc_tensor': doc_tensors[sim_idx],
                'is_random_text': False
            })

        # 融合KV cache
        if recall_method != RecallMethod.NO_PREPROCESS_WITH_BIAS:
            kv_fusion.fuse_kv_caches(
                current_doc_tensor=doc_tensors[doc_idx],
                similar_doc_info=similar_doc_info,
                system_tensor=system_tensor,
                kv_cache_dir=no_preprocess_dir,
                output_path=preprocess_dir,
                output_id=f"doc_{doc_idx}",
                topk=topk,
                recall_method=recall_method,
                revert_rope=revert_rope,
                reprocess_method=reprocess_method,
                shuffle_seed=random_seed if recall_method == RecallMethod.BGE_SHUFFLED else None
            )
        else:
            # NO_PREPROCESS_WITH_BIAS: 应用steering vector
            if kv_distribution_stats is None:
                raise ValueError("kv_distribution_stats required for NO_PREPROCESS_WITH_BIAS method")

            kv_fusion.apply_distribution_steering(
                no_preprocess_kv_path_prefix=f"{no_preprocess_dir}/doc_{doc_idx}",
                output_path_prefix=f"{preprocess_dir}/doc_{doc_idx}",
                kv_distribution_stats=kv_distribution_stats,
                steering_alpha=steering_alpha
            )

    # 拷贝system cache到preprocess目录
    shutil.copy(
        f"{no_preprocess_dir}/system_0_key.pt",
        f"{preprocess_dir}/system_0_key.pt"
    )
    shutil.copy(
        f"{no_preprocess_dir}/system_0_value.pt",
        f"{preprocess_dir}/system_0_value.pt"
    )

    print("\nPreprocessing completed!")

    return {
        'no_preprocess_dir': no_preprocess_dir,
        'preprocess_dir': preprocess_dir,
        'num_documents': len(doc_tensors),
        'system_len': system_tensor.shape[0],
        'doc_lengths': [t.shape[0] for t in doc_tensors]
    }


if __name__ == "__main__":
    """
    使用示例
    """
    import argparse
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description="KV Fusion Preprocessing")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--model_type', type=str, default='qwen2', help='Model type')
    parser.add_argument('--bge_model_path', type=str, help='Path to BGE model')
    parser.add_argument('--input_json', type=str, required=True, help='Input JSON file with documents')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--topk', type=int, default=3, help='Top-k documents to fuse')
    parser.add_argument('--recall_method', type=str, default='bge',
                       choices=['bge', 'random', 'repeat_self', 'bge_shuffled'],
                       help='Document recall method')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path)
    config.torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=config.torch_dtype,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 加载文档
    print(f"Loading documents from {args.input_json}...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = data['documents']
    system_prompt = data.get('system_prompt', '')

    # 运行预处理pipeline
    recall_method = RecallMethod(args.recall_method)

    result = preprocess_documents_pipeline(
        model=model,
        tokenizer=tokenizer,
        model_type=args.model_type,
        documents=documents,
        system_prompt=system_prompt,
        output_dir=args.output_dir,
        bge_model_path=args.bge_model_path,
        topk=args.topk,
        recall_method=recall_method,
        device=args.device,
        random_seed=args.random_seed
    )

    print("\nResults:")
    print(json.dumps(result, indent=2))
