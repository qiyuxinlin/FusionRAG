import copy
from typing import Dict, Union, Tuple, List
import torch
from torch import nn
import itertools
import time
from ktransformers.util.utils import load_kv


def compare_kv_cache_tokens(kv_tensor: torch.Tensor) -> dict:
    """
    高效量化 KV Cache 中第三维度（Token 维度）上任意两个 Token 之间的 KV 差异。

    参数:
        kv_tensor: torch.Tensor, 形状必须为 [1, 2, x, 128]
                  其中 dim=1 的 0 位代表 Key, 1 位代表 Value

    返回:
        dict: 包含 K 相似度矩阵和 V 差异性矩阵的字典，形状均为 [x, x]
    """
    assert kv_tensor.shape[0] == 1, "Batch size 必须为 1"
    assert kv_tensor.shape[1] == 2, "第二维度必须为 2 (分别代表 K 和 V)"

    # 1. 剥离 Batch 维度，并分离 K 和 V -> 形状变为 [x, 128]
    K = kv_tensor[0, 0, :, :].float()  # 强转 float 避免半精度溢出
    V = kv_tensor[0, 1, :, :].float()

    x = K.shape[0]

    # ----------------------------------------------------
    # 2. 高效对比 K: 利用广播并行计算所有 Token 两两之间的余弦相似度
    # ----------------------------------------------------
    # K 的形状: [x, 128] -> 扩展为 [x, 1, 128] 和 [1, x, 128]
    K_expanded1 = K.unsqueeze(1)
    K_expanded2 = K.unsqueeze(0)

    # 利用 PyTorch 内置的余弦相似度函数，在最后一个维度上规约
    # 结果 k_cosine_matrix 的形状为 [x, x]
    # 矩阵中 (i, j) 位置的值代表第 i 个 Token 和第 j 个 Token 的 K 向量夹角余弦值
    k_cosine_matrix = torch.cosine_similarity(K_expanded1, K_expanded2, dim=-1)

    # ----------------------------------------------------
    # 3. 高效对比 V: 利用广播并行计算所有 Token 两两之间的相对范数差异
    # ----------------------------------------------------
    # 计算 V1 - V2 的差值矩阵，形状为 [x, x, 128]
    V_diff = V.unsqueeze(1) - V.unsqueeze(0)

    # 计算元素级的平方和，并在最后一个维度规约（相当于计算 Frobenius 范数）
    # v_diff_norm 形状为 [x, x]
    v_diff_norm = torch.norm(V_diff, p='fro', dim=-1)

    # 计算分母：||V_i|| + ||V_j|| 用于归一化，规避绝对数值大小的影响
    v_norms = torch.norm(V, p='fro', dim=-1)  # [x]
    v_norms_matrix = v_norms.unsqueeze(1) + v_norms.unsqueeze(0)  # 广播得到 [x, x]

    # 相对差异矩阵 = ||V_i - V_j|| / (||V_i|| + ||V_j|| + epsilon)
    v_diff_matrix = v_diff_norm / (v_norms_matrix + 1e-8)

    return {
        "k_cosine_similarity": k_cosine_matrix,  # 值域 [-1, 1]，越接近 1 越相似
        "v_relative_difference": v_diff_matrix  # 值域 [0, 1]，越接近 0 越相似
    }


def load_kv_and_generate_draft_model(
        model,
        past_key_values,
        past_key_values_compare,
        passages,
        load_path='',
        preprocess_load_path='',
        revert_rope=False,
        device="cuda",
        device_map=None,
        hash_keys=None,
        query="",
        keyword="",
        preprocess=False
):
    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = f"cuda:{device_map['model.embed_tokens']}" if device_map is not None else device

    passages_len = [passage.shape[0] for passage in passages]
    passages_len_sum = sum(passages_len)
    passages_len_sum_without_query = sum([passage.shape[0] for passage in passages[:-1]])

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    system_len = passages[0].shape[0]

    key_cache = []
    value_cache = []

    chunk_ids = list(range(len(passages) - 1))

    for idx, passage in enumerate(passages[:-1]):
        if idx >=2 and preprocess:
            chunk_key_cache = torch.load(f'{preprocess_load_path}/{hash_keys[idx]}_key.pt', weights_only=True).to('cpu')
            chunk_value_cache = torch.load(f'{preprocess_load_path}/{hash_keys[idx]}_value.pt', weights_only=True).to('cpu')
        else:
            chunk_key_cache = torch.load(f'{load_path}/{hash_keys[idx]}_key.pt', weights_only=True).to('cpu')
            print(load_path)
            chunk_value_cache = torch.load(f'{load_path}/{hash_keys[idx]}_value.pt', weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)
    ## load kv cache
    past_len = load_kv(model, passages, chunk_ids, key_cache, value_cache, input_device, past_key_values, revert_rope, system_len, query=query)
    past_len = load_kv(model, passages, chunk_ids, key_cache, value_cache, input_device, past_key_values_compare, revert_rope, system_len, query=query)

    k_need_index = [i for i in range(passages_len_sum)]

    use_sparse_attention = False
    reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(input_device)
    cache_position = torch.tensor(k_need_index, device=input_device)
    with torch.no_grad():
        without_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(input_device)

        # Don't force move to input_device - keep on the device where model output is
        # This avoids cross-GPU transfer deadlock in PP mode
        start_time = time.time()
        model_output = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position,
            past_key_values=past_key_values, return_dict=False, use_cache=True, use_sparse_attention=use_sparse_attention,
        )[0]

        mean_key_after = torch.stack(past_key_values.key_cache).mean(dim=0)[:, :, system_len:passages_len_sum_without_query, :]
        mean_value_after = torch.stack(past_key_values.value_cache).mean(dim=0)[:, :, system_len:passages_len_sum_without_query, :]
        mean_key_before = torch.stack(past_key_values_compare.key_cache).mean(dim=0)[:, :, system_len:passages_len_sum_without_query, :]
        mean_value_before = torch.stack(past_key_values_compare.value_cache).mean(dim=0)[:, :, system_len:passages_len_sum_without_query, :]

        if "mse" in keyword:
            ## mse，越小越相似
            result = analyze_print_and_return_max_mse_map(
                mean_key_before,
                mean_value_before,
                mean_key_after,
                mean_value_after,
                top_n=50
            )
        else:
            ## 余弦相似度，越大越相似
            result = analyze_print_and_return_min_sim_map(
                mean_key_before,
                mean_value_before,
                mean_key_after,
                mean_value_after,
                top_n=50
            )


        return result


def analyze_print_and_return_min_sim_map(
        mean_key_before: torch.Tensor,
        mean_value_before: torch.Tensor,
        mean_key_after: torch.Tensor,
        mean_value_after: torch.Tensor,
        top_n: int = 10
) -> dict:
    """
    全量计算每个位置(Index)的 KV 拯救权重。
    kv_combined_weight_map 采用先相乘相似度、再用 1 减的逻辑。
    """
    assert mean_key_before.shape == mean_key_after.shape, "Before 和 After 的 Shape 必须一致"

    # 1. 提取特征向量维度 -> [seq_len, 128]
    k_before = mean_key_before[0, 0, :, :].float()
    v_before = mean_value_before[0, 0, :, :].float()
    k_after = mean_key_after[0, 0, :, :].float()
    v_after = mean_value_after[0, 0, :, :].float()

    # 2. 计算原始的 Cosine Similarity 向量
    time_start = time.time()
    k_cos = torch.cosine_similarity(k_before, k_after, dim=-1).cpu().numpy()
    v_cos = torch.cosine_similarity(v_before, v_after, dim=-1).cpu().numpy()
    print(f"time to compute similarity: {time.time() - time_start}")

    # 3. 🚀【核心修改】：先让 cos_sim 相乘，再用 1 减
    kv_combined_cos = k_cos * v_cos
    kv_combined_weight = 1.0 - kv_combined_cos

    # 4. 分别转换成 1 - cos 权重用于满足原输出格式
    k_weights = 1.0 - k_cos
    v_weights = 1.0 - v_cos

    # 5. 保持原格式顺序返回
    seq_len = k_before.shape[0]
    return {
        "key_min_sim_map": {i: float(k_weights[i]) for i in range(seq_len)},
        "value_min_sim_map": {i: float(v_weights[i]) for i in range(seq_len)},
        "kv_combined_weight_map": {i: float(kv_combined_weight[i]) for i in range(seq_len)}
    }

def analyze_print_and_return_max_mse_map(
        mean_key_before: torch.Tensor,
        mean_value_before: torch.Tensor,
        mean_key_after: torch.Tensor,
        mean_value_after: torch.Tensor,
        top_n: int = 10
) -> dict:
    """
    找出 KV Cache 均方误差(MSE)最大的 Top-N 个 Token，打印排查结果，
    并返回以原位置(Index)为 key，MSE 值为 value 的字典(Map)。

    参数:
        mean_key_before, mean_value_before: 优化前/对比组的 K, V Tensor [1, 2, x, 128]
        mean_key_after, mean_value_after:   优化后/实验组的 K, V Tensor [1, 2, x, 128]
        top_n: 需要排查的异常 Token 数量

    返回:
        dict: {
            "key_max_mse_map": {int_idx: float_mse, ...},
            "value_max_mse_map": {int_idx: float_mse, ...}
        }
    """
    assert mean_key_before.shape == mean_key_after.shape, "Before 和 After 的 Shape 必须一致"

    # 1. 提取并压缩特征向量维度 -> [x, 128]
    k_before = mean_key_before[0, 0, :, :].float()
    v_before = mean_value_before[0, 0, :, :].float()
    k_after = mean_key_after[0, 0, :, :].float()
    v_after = mean_value_after[0, 0, :, :].float()

    seq_len = k_before.shape[0]
    # 保持原逻辑：强制将 top_n 设为 seq_len 全量排序
    top_n = seq_len

    # 2. 【核心改动】计算各位置 Token 的 MSE (Mean Squared Error)
    # 计算 (Before - After)^2 并在最后一个特征维度 (128) 上求平均
    k_mse = torch.mean((k_before - k_after) ** 2, dim=-1)
    v_mse = torch.mean((v_before - v_after) ** 2, dim=-1)

    # 3. 【核心改动】抓取 MSE 最大（差异最大）的 Top-N 个元素
    # 注意：MSE 越大表示误差和差异越大，因此排序开关改为 largest=True
    k_max_values, k_max_indices = torch.topk(k_mse, k=top_n, largest=True)
    v_max_values, v_max_indices = torch.topk(v_mse, k=top_n, largest=True)

    # 4. 转换为 CPU NumPy 以进行打印和构建 Map
    k_max_values_np = k_max_values.cpu().numpy()
    k_max_indices_np = k_max_indices.cpu().numpy()
    v_max_values_np = v_max_values.cpu().numpy()
    v_max_indices_np = v_max_indices.cpu().numpy()

    # 5. 构建以位置(Index)为 key, MSE 值为 value 的 Map
    key_max_mse_map = {int(idx): float(val) for idx, val in zip(k_max_indices_np, k_max_values_np)}
    value_max_mse_map = {int(idx): float(val) for idx, val in zip(v_max_indices_np, v_max_values_np)}

    # 6. 格式化打印排查报告（限制展示前 30 行，防止长序列爆终端）
    print("=" * 70)
    print(f"🚨 [异常排查] 均方误差（MSE差异最大）的 Top-{top_n} 个 Token 倒序列表")
    print("=" * 70)

    # 如果序列总长不足 30，自适应调整打印行数
    print_range = min(30, seq_len)

    print(f" 📂 分支 1: Key Cache 差异最大 Top-{print_range} (总样本: {top_n})")
    print("-" * 55)
    print(f"{'Rank':^6} | {'Token Index (原位置)':^22} | {'Key MSE':^18}")
    print("-" * 55)
    for r in range(print_range):
        print(f"{r + 1:^6} | {k_max_indices_np[r]:^22} | {k_max_values_np[r]:^18.6f}")

    print("\n" + "-" * 70 + "\n")

    print(f" 📂 分支 2: Value Cache 差异最大 Top-{print_range} (总样本: {top_n})")
    print("-" * 55)
    print(f"{'Rank':^6} | {'Token Index (原位置)':^22} | {'Value MSE':^18}")
    print("-" * 55)
    for r in range(print_range):
        print(f"{r + 1:^6} | {v_max_indices_np[r]:^22} | {v_max_values_np[r]:^18.6f}")

    print("=" * 70)

    # 7. 返回组装好的 map 结构
    return {
        "key_min_sim_map": key_max_mse_map,
        "value_min_sim_map": value_max_mse_map
    }