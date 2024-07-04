import torch
import os, sys

sys.path.append(os.path.dirname(__file__) + "/../")

from cpuinfer import CPUInfer
from cpuinfer_kvcache import CPUInferKVCache
from flash_attn import flash_attn_func
import time


def merge_output_and_lse(output_1, output_2, lse_1, lse_2):
    lse = lse_1 + torch.log(1 + torch.exp(lse_2 - lse_1))
    lse = torch.where(torch.isnan(lse), lse_2, lse)
    output = torch.exp(lse_1 - lse) * output_1 + torch.exp(lse_2 - lse) * output_2
    output = output.to(torch.float16)
    return output, lse


cpuinfer = CPUInfer(96)
layer_num = 10
kv_head_num = 8
q_head_num = 32
head_dim = 64
block_len = 64
anchor_num = 4
cache_total_len = 580
q_len = 5
thread_kvcache = CPUInferKVCache(
    layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num
)

# 准备update_one_block_fp16测试数据
k_in = torch.randn(
    kv_head_num, block_len, head_dim
).half()  # (block_len, kv_head_num, head_dim)
v_in = torch.randn(
    kv_head_num, block_len, head_dim
).half()  # (block_len, kv_head_num, head_dim)

k_in_all = torch.zeros(
    kv_head_num, 0, head_dim
).half()  # (block_len, kv_head_num, head_dim)
v_in_all = torch.zeros(
    kv_head_num, 0, head_dim
).half()  # (block_len, kv_head_num, head_dim)

k_compare = torch.zeros(
    kv_head_num, 0, head_dim
).half()  # (block_len, kv_head_num, head_dim)
v_compare = torch.zeros(
    kv_head_num, 0, head_dim
).half()  # (block_len, kv_head_num, head_dim)

try:
    for layer_idx in range(0, 10):
        for block_idx in range(0, 10):
            k_in = torch.randn(
                kv_head_num, block_len, head_dim
            ).half()  # (block_len, kv_head_num, head_dim)
            v_in = torch.randn(
                kv_head_num, block_len, head_dim
            ).half()  # (block_len, kv_head_num, head_dim)

            cpuinfer.submit(
                thread_kvcache.update_one_block_fp16(k_in, v_in, layer_idx, block_idx)
            )
            cpuinfer.sync()
            if layer_idx == 1:
                k_in_all = torch.concat([k_in_all, k_in], dim=1)
                v_in_all = torch.concat([v_in_all, v_in], dim=1)

    cpuinfer.sync()
    print("update_one_block_fp16 test passed.")
except Exception as e:
    print(f"update_one_block_fp16 test failed: {e}")

# 测试get_one_block_fp16
try:
    k_out = torch.zeros(
        kv_head_num, block_len, head_dim
    ).half()  # (block_len, kv_head_num, head_dim)
    v_out = torch.zeros(
        kv_head_num, block_len, head_dim
    ).half()  # (block_len, kv_head_num, head_dim)
    for layer_idx in range(0, 10):
        for block_idx in range(0, 10):
            cpuinfer.submit(
                thread_kvcache.get_one_block_fp16(k_out, v_out, layer_idx, block_idx)
            )
            cpuinfer.sync()
            if layer_idx == 1:
                k_compare = torch.concat([k_compare, k_out], dim=1)
                v_compare = torch.concat([v_compare, v_out], dim=1)
                print(k_out)
                print(v_out)

    # print(k_out)
    # print(v_out)

    print("get_one_block_fp16 test passed.")
except Exception as e:
    print(f"get_one_block_fp16 test failed: {e}")

# 准备attn测试数据
q_in = torch.randn(
    1, q_head_num, q_len, head_dim
).half()  # (bsz, q_head_num, q_len, head_dim)


k_compare = k_compare[:, :cache_total_len, :].unsqueeze(0)
v_compare = v_compare[:, :cache_total_len, :].unsqueeze(0)

k_in_all = k_in_all[:, :cache_total_len, :].unsqueeze(0)
v_in_all = v_in_all[:, :cache_total_len, :].unsqueeze(0)

print(q_in.shape)
print(k_compare.shape)
print(v_compare.shape)


# 测试量化结果是非正确
# print(k_compare[0,0,:,0])
# print(k_in_all[0,0,:,0])

print("v_compare:")
print(v_compare[0, 0, :, 0])
print("v:")
print(v_in_all[0, 0, :, 0])

# output_compare_sdp = torch.nn.functional.scaled_dot_product_attention(
#     q_in, k_compare, v_compare
# )

output_compare, attn_lse_compare, _ = flash_attn_func(
    q_in.transpose(1, 2).to(torch.device("cuda")),
    k_compare.transpose(1, 2).to("cuda"),
    v_compare.transpose(1, 2).to("cuda"),
    return_attn_probs=True,
)

q_in = q_in.expand(2, q_head_num, q_len, head_dim).transpose(1, 2).contiguous()
output = torch.zeros_like(q_in).contiguous()


attn_lse = torch.zeros(2, q_len, q_head_num).float()  # (bsz, q_len, q_head_num)

block_table = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.int
)
cache_seqlens = torch.tensor(
    [cache_total_len, cache_total_len], dtype=torch.int
)  # (bsz,)

print(output.shape, output_compare.shape)
print(attn_lse_compare.shape, attn_lse.shape)
thread_kvcache.update_cache_total_len(cache_total_len)

cpuinfer.submit(
    thread_kvcache.dump_kvcache(
        block_table[0],
        cache_total_len,
        "/home/chenht/tmp/a.kvcache",
    )
)
cpuinfer.sync()
related_thread_kvcache = CPUInferKVCache(
    layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num
)
cpuinfer.submit(
    related_thread_kvcache.load_kvcache(
        "/home/chenht/tmp/a.kvcache"
    )
)
cpuinfer.sync()

# 测试attn
try:
    cpuinfer.submit(thread_kvcache.attn(q_in, output, attn_lse, 1, None, None))
    cpuinfer.sync()
    print("output:")
    print(output)
    # print(output[1][0][0])
    print("output_compare:")
    print(output_compare)
    # print("output_compare_sdp:")
    # print(output_compare_sdp[0][0][0])
    print("lse:")
    print(attn_lse)
    print("lse compare:")
    print(attn_lse_compare)
    cpuinfer.submit(related_thread_kvcache.attn(q_in, output, attn_lse, 1, None, None))
    cpuinfer.sync()
    print("dump_output:")
    print(output[0][0][0])
    print("dump_lse:")
    print(attn_lse)

    print("attn test passed.")
except Exception as e:
    print(f"attn test failed: {e}")


###
# attention 计算结果正确
###