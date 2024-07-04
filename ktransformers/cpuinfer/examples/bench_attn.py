import torch
import os, sys

sys.path.append(os.path.dirname(__file__) + "/../")

from cpuinfer import CPUInfer
from cpuinfer_kvcache import CPUInferKVCache
from flash_attn import flash_attn_func
import time


layer_num = 3
kv_head_num = 32
q_head_num = 32
head_dim = 128
block_len = 1024
anchor_num = 4
seq_len = 102400
q_len = 1
cpuinfer = CPUInfer(32)
warm_up_iter = 1000
test_iter = 10000

thread_kvcache = CPUInferKVCache(
    layer_num, kv_head_num, q_head_num, head_dim, block_len, anchor_num
)

for layer_idx in range(0, layer_num):
    for block_idx in range(0, seq_len // block_len):
        k_in = torch.randn(kv_head_num, block_len, head_dim, device="cuda").half().cpu()
        v_in = torch.randn(kv_head_num, block_len, head_dim, device="cuda").half().cpu()
        cpuinfer.submit(
            thread_kvcache.update_one_block_fp16(k_in, v_in, layer_idx, block_idx)
        )
        cpuinfer.sync()

thread_kvcache.update_cache_total_len(seq_len)

q_in = torch.randn(1, q_len, q_head_num, head_dim).half()
output = torch.zeros_like(q_in).contiguous()
attn_lse = torch.zeros(1, q_len, q_head_num).float()

# warm up
for i in range(warm_up_iter):
    cpuinfer.submit(thread_kvcache.attn(q_in, output, attn_lse, i % layer_num , None, None))
    cpuinfer.sync()

# test
start = time.time()
for i in range(test_iter):
    cpuinfer.submit(thread_kvcache.attn(q_in, output, attn_lse, i % layer_num , None, None))
    cpuinfer.sync()
end = time.time()
total_time = end - start
print('Time: ', total_time)
print('Iteration: ', test_iter) 
print('Time per iteration: ', total_time / test_iter)
print('Bandwidth: ', seq_len * kv_head_num * head_dim * test_iter / total_time / 1024 / 1024 / 1024, 'GB/s')
print("All tasks completed.")
