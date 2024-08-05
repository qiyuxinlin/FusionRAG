import torch
import os, sys

sys.path.append(os.path.dirname(__file__) + "/../cpu_backend")
from cpuinfer import CPUInfer, CPUInferKVCache
from flash_attn import flash_attn_func
import time
import threading


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(41)

# 初始化 cpuinfer
cpuinfer = CPUInfer(32)
layer_num = 10
kv_head_num = 8
q_head_num = 32
head_dim = 64
block_len = 64
anchor_num = 4
cache_total_len = 580
thread_kvcache = CPUInferKVCache(
    layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num, kv_type="FP16"
)

block_table = (
    torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32).contiguous().to("cpu")
)
q_len = 32
past_len = torch.tensor([0, 0], dtype=torch.int32).contiguous().to("cpu")
max_block_num = 3
layer_idx = 0
batch_size = 2
k_cache_0 = torch.randn(
    (batch_size, max_block_num * block_len, kv_head_num, head_dim),
    dtype=torch.float16,
    device="cpu",
).contiguous()
v_cache_0 = torch.randn(
    (batch_size, max_block_num * block_len, kv_head_num, head_dim),
    dtype=torch.float16,
    device="cpu",
).contiguous()

k_comp = k_cache_0.clone()
v_comp = v_cache_0.clone()
cpuinfer.submit(
    thread_kvcache.get_and_update_fp16(
        k_cache_0,
        v_cache_0,
        layer_idx,
        block_table,
        max_block_num,
        past_len,
        q_len,
    )
)

cpuinfer.sync()
print(k_cache_0[1, :32, 0, 0])
print(v_cache_0[1, :32, 0, 0])
pass

k_cache_0.zero_()
v_cache_0.zero_()

past_len = torch.tensor([32, 32], dtype=torch.int32).contiguous().to("cpu")
q_len = 31
cpuinfer.submit(
    thread_kvcache.get_and_update_fp16(
        k_cache_0,
        v_cache_0,
        layer_idx,
        block_table,
        max_block_num,
        past_len,
        q_len,
    )
)

cpuinfer.sync()
print(k_cache_0[1, :32, 0, 0])
print(v_cache_0[1, :32, 0, 0])
pass

importance_0 = torch.randn(
    (batch_size, max_block_num * block_len, q_head_num),
    dtype=torch.float16,
    device="cpu",
).contiguous()

cpuinfer.submit(
    thread_kvcache.update_importance(
        importance_0,
        layer_idx,
        block_table,
        max_block_num,
        past_len,
        q_len,
    )
)

cpuinfer.sync()

q_len = 1
q_in = torch.randn(
    (batch_size, q_len, q_head_num, head_dim),
    dtype=torch.float16,
    device="cpu",
).contiguous()

k_in = torch.randn(
    (batch_size, q_len, kv_head_num, head_dim),
    dtype=torch.float16,
    device="cpu",
).contiguous()

v_in = torch.randn(
    (batch_size, q_len, kv_head_num, head_dim),
    dtype=torch.float16,
    device="cpu",
).contiguous()

output = torch.empty(
    (batch_size, q_len, q_head_num, head_dim),
    dtype=torch.float16,
    device="cpu",
).contiguous()
lse = torch.empty(
    (batch_size, q_len, q_head_num),
    dtype=torch.float32,
    device="cpu",
).contiguous()

# k_comp = torch.concat([k_comp, k_in], dim=1)
# v_comp = torch.concat([v_comp, v_in], dim=1)

past_len = torch.tensor([63, 63], dtype=torch.int32).contiguous().to("cpu")
k_in = k_cache_0[:, 64:65, :, :]
v_in = v_cache_0[:, 64:65, :, :]

past_len = torch.tensor([63, 63], dtype=torch.int32).contiguous().to("cpu")
q_len = 1
cpuinfer.submit(
    thread_kvcache.get_and_update_fp16(
        k_cache_0,
        v_cache_0,
        layer_idx,
        block_table,
        max_block_num,
        past_len,
        q_len,
    )
)

# cpuinfer.submit(
#     thread_kvcache.attn_with_kvcache(
#         q_in,
#         k_in,
#         v_in,
#         output,
#         lse,
#         layer_idx,
#         block_table,
#         past_len,
#     )
# )

# cpuinfer.sync()

# print(output)

past_len = torch.tensor([64, 64], dtype=torch.int32).contiguous().to("cpu")

cpuinfer.submit(
    thread_kvcache.attn(
        q_in,
        output,
        lse,
        layer_idx,
        block_table,
        past_len,
    )
)

cpuinfer.sync()

print(output[:, :, :8, :8])

import flash_attn

k_comp = k_comp[:, :63]
v_comp = v_comp[:, :63]

k_comp = torch.concat([k_comp, k_in], dim=1)
v_comp = torch.concat([v_comp, v_in], dim=1)


output_comp, lse_comp = flash_attn.flash_attn_with_kvcache(
    q_in.to("cuda"),
    k_comp.to("cuda"),
    v_comp.to("cuda"),
    causal=True,
    return_softmax_lse=True,
)
print(output_comp[:, :, :8, :8])

print(lse)
print(lse_comp.transpose(1, 2))
