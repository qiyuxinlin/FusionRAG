# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared Triton operators for sparse attention across different model architectures.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def selected_query_sparse_attention_fwd_kernel(
    Q_selected, # [batch_size, num_heads, selectd_q_num, head_dim]
    K, # [batch_size, num_heads, context_size, head_dim]
    V,  # [batch_size, num_heads, context_size, head_dim]
    batch_size, num_heads, context_size,
    q_idx,  # [batch_size, num_heads, selected_q_num]
    q_size,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_sqz, stride_sqh, stride_sqm, stride_sqk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_idxz, stride_idxh, stride_idxm,
    NUM_ROWS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
    scale: tl.constexpr,
):
    """
    Triton kernel for sparse attention with selected queries.

    This kernel computes attention for a subset of query tokens, which is more
    efficient when only certain query positions need to attend to the full context.
    """
    # Grid has two dimensions (2, 1)
    start_m = tl.program_id(0)  # Current block index (in q_size dimension)
    off_hz = tl.program_id(1)  # Block index in batch_size*num_heads dimension

    batch_idx = off_hz // num_heads
    head_idx = off_hz % num_heads
    # Current PID offset in query dimension
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # [0,... BLOCK_M-1]
    # Current PID offset in context dimension
    offs_n = tl.arange(0, BLOCK_N) # [0,... BLOCK_N-1]
    # Current PID offset in head_dim dimension
    offs_d = tl.arange(0, BLOCK_DMODEL) # [0,... head_dim-1]
    # Query starting offset in first two dimensions
    sq_offset = (off_hz // num_heads) * stride_sqz + (off_hz % num_heads) * stride_sqh
    o_offset = (off_hz // num_heads) * stride_oz + (off_hz % num_heads) * stride_oh

    # Key starting offset in first two dimensions
    kv_offset = (off_hz // num_heads) * stride_kz + (off_hz % num_heads) * stride_kh

    if start_m * BLOCK_M >= q_size:
        return
     # Load q_idx and ensure no out-of-bounds

    cols_ptr = q_idx + batch_idx * stride_idxz + head_idx * stride_idxh + start_m * BLOCK_M * stride_idxm
    cols_mask = offs_m < q_size
    # Index for specific query
    q_cols = tl.load(cols_ptr + offs_m % BLOCK_M,  mask=cols_mask, other=0)

    # Maximum value in current query index
    max_qcol = tl.max(q_cols, axis=0)

    q_ptrs = Q_selected + sq_offset + offs_m[:, None] * stride_sqm + offs_d[None, :] * stride_sqk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk # [[0],... [head_dim-1]]
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk # [[0,... head_dim-1]]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok


    # Ensure q_cols does not go out of bounds
    valid_mask = q_cols < context_size
    # Load q from Q_selected q_ptrs
    q = tl.load(q_ptrs)


    # Max attention weight for each query
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # Accumulated weight for each query
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Accumulated output
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # qk_scale= head_dim ** -0.5 / 1.44269504
    qk_scale =  1 / scale
    q = (q * qk_scale).to(dtype)
    # Iterate over context in blocks
    for start_n in range(0, max_qcol + 1, BLOCK_N):
        # Current block context index
        cols = start_n + offs_n # [BLOCK_N]
        n_mask = cols < context_size # [BLOCK_N]
        # k_mask = cols <= q_cols # [BLOCK_M, BLOCK_N]
        # cols[None, :] [1, BLOCK_N]
        k_mask = cols <= max_qcol
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)


        # Boundary check to ensure cols and q_cols are in valid range
        qk = tl.where(cols[None, :] <= q_cols[:, None], qk, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp(m_i - m_i_new)
        p = tl.math.exp(qk - m_i_new[:, None])

        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    acc /= l_i[:, None]
    # Ensure no out-of-bounds before writing output
    tl.store(o_ptrs, acc.to(dtype), mask=offs_m[:, None] < q_size)


@triton.jit
def selected_query_sparse_attention_fwd_kernel_gqa(
    Q_selected, # [batch_size, num_q_heads, selected_q_num, head_dim]
    K, # [batch_size, num_kv_heads, context_size, head_dim]
    V, # [batch_size, num_kv_heads, context_size, head_dim]
    batch_size, num_q_heads, num_kv_heads, context_size,
    q_idx,  # [batch_size, num_q_heads, selected_q_num]
    q_size,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_sqz, stride_sqh, stride_sqm, stride_sqk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_idxz, stride_idxh, stride_idxm,
    NUM_ROWS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
    scale: tl.constexpr,
):
    """
    Triton kernel for sparse attention with selected queries and GQA support.

    This variant supports Grouped Query Attention (GQA), where multiple query heads
    share the same key-value heads.
    """
    # Grid has two dimensions (2, 1)
    start_m = tl.program_id(0)  # Current block index (in q_size dimension)
    off_hz = tl.program_id(1)  # Block index in batch_size*num_q_heads dimension

    batch_idx = off_hz // num_q_heads
    q_head_idx = off_hz % num_q_heads

    # GQA: Calculate corresponding kv head index
    # num_q_heads_per_kv = num_q_heads // num_kv_heads
    kv_head_idx = q_head_idx // (num_q_heads // num_kv_heads)

    # Current PID offset in query dimension
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # [0,... BLOCK_M-1]
    # Current PID offset in context dimension
    offs_n = tl.arange(0, BLOCK_N) # [0,... BLOCK_N-1]
    # Current PID offset in head_dim dimension
    offs_d = tl.arange(0, BLOCK_DMODEL) # [0,... head_dim-1]

    # Query starting offset in first two dimensions
    sq_offset = batch_idx * stride_sqz + q_head_idx * stride_sqh
    o_offset = batch_idx * stride_oz + q_head_idx * stride_oh

    # Key/value starting offset in first two dimensions (using kv_head_idx)
    kv_offset = batch_idx * stride_kz + kv_head_idx * stride_kh

    if start_m * BLOCK_M >= q_size:
        return

    # Load q_idx and ensure no out-of-bounds
    cols_ptr = q_idx + batch_idx * stride_idxz + q_head_idx * stride_idxh + start_m * BLOCK_M * stride_idxm
    cols_mask = offs_m < q_size
    # Index for specific query
    q_cols = tl.load(cols_ptr + offs_m % BLOCK_M, mask=cols_mask, other=0)

    # Maximum value in current query index
    max_qcol = tl.max(q_cols, axis=0)

    q_ptrs = Q_selected + sq_offset + offs_m[:, None] * stride_sqm + offs_d[None, :] * stride_sqk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk # [[0],... [head_dim-1]]
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk # [[0,... head_dim-1]]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # Ensure q_cols does not go out of bounds
    valid_mask = q_cols < context_size
    # Load q from Q_selected q_ptrs
    q = tl.load(q_ptrs)

    # Max attention weight for each query
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # Accumulated weight for each query
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Accumulated output
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # qk_scale= head_dim ** -0.5 / 1.44269504
    qk_scale = 1 / scale
    q = (q * qk_scale).to(dtype)

    # Iterate over context in blocks
    for start_n in range(0, max_qcol + 1, BLOCK_N):
        # Current block context index
        cols = start_n + offs_n # [BLOCK_N]
        n_mask = cols < context_size # [BLOCK_N]
        k_mask = cols <= max_qcol
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        # Boundary check to ensure cols and q_cols are in valid range
        qk = tl.where(cols[None, :] <= q_cols[:, None], qk, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp(m_i - m_i_new)
        p = tl.math.exp(qk - m_i_new[:, None])

        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc /= l_i[:, None]
    # Ensure no out-of-bounds before writing output
    tl.store(o_ptrs, acc.to(dtype), mask=offs_m[:, None] < q_size)


def selected_query_attention_entrance(
    sq: torch.Tensor,  # [batch_size, num_heads, selected_q_num, head_dim]
    k: torch.Tensor,  # [batch_size, num_heads, context_size, head_dim]
    v: torch.Tensor,  # [batch_size, num_heads, context_size, head_dim]
    q_idx: torch.Tensor,  # [batch_size, num_heads, selected_q_num]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """
    Entry function for selected query sparse attention (MHA version).

    Args:
        sq: Selected query states
        k: Key states
        v: Value states
        q_idx: Indices of selected queries
        block_size_M: Block size for query dimension
        block_size_N: Block size for context dimension

    Returns:
        Attention output for selected queries
    """
    dq, dk, dv = sq.shape[-1], k.shape[-1], v.shape[-1]
    assert dq == dk and dk == dv

    q_size = q_idx.shape[-1]
    batch_size, num_heads, context_size, head_dim = k.shape

    o = torch.zeros(batch_size, num_heads, q_size, head_dim, device=sq.device, dtype=sq.dtype)
    grid = (triton.cdiv(q_size, block_size_M), batch_size * num_heads)
    dtype = tl.bfloat16 if sq.dtype == torch.bfloat16 else tl.float16
    selected_query_sparse_attention_fwd_kernel[grid](
        sq, k, v,
        batch_size, num_heads, context_size,
        q_idx,
        q_size,
        o,
        sq.stride(0), sq.stride(1), sq.stride(2), sq.stride(3),
        sq.stride(0), sq.stride(1), sq.stride(2), sq.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q_idx.stride(0), q_idx.stride(1), q_idx.stride(2),
        (q_size + block_size_M - 1) // block_size_M,
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=head_dim,
        dtype=dtype,
        num_warps=4, num_stages=2,
        scale = math.sqrt(head_dim)
    )
    return o


def selected_query_attention_entrance_gqa(
    sq: torch.Tensor,  # [batch_size, num_q_heads, selected_q_num, head_dim]
    k: torch.Tensor,  # [batch_size, num_kv_heads, context_size, head_dim]
    v: torch.Tensor,  # [batch_size, num_kv_heads, context_size, head_dim]
    q_idx: torch.Tensor,  # [batch_size, num_q_heads, selected_q_num]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """
    Entry function for selected query sparse attention (GQA version).

    Args:
        sq: Selected query states
        k: Key states (with fewer heads for GQA)
        v: Value states (with fewer heads for GQA)
        q_idx: Indices of selected queries
        block_size_M: Block size for query dimension
        block_size_N: Block size for context dimension

    Returns:
        Attention output for selected queries
    """
    dq, dk, dv = sq.shape[-1], k.shape[-1], v.shape[-1]
    assert dq == dk and dk == dv

    q_size = q_idx.shape[-1]
    batch_size, num_q_heads, _, head_dim = sq.shape
    _, num_kv_heads, context_size, _ = k.shape

    # GQA check: num_q_heads must be a multiple of num_kv_heads
    assert num_q_heads % num_kv_heads == 0, \
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

    o = torch.zeros(batch_size, num_q_heads, q_size, head_dim, device=sq.device, dtype=sq.dtype)
    grid = (triton.cdiv(q_size, block_size_M), batch_size * num_q_heads)
    dtype = tl.bfloat16 if sq.dtype == torch.bfloat16 else tl.float16

    selected_query_sparse_attention_fwd_kernel_gqa[grid](
        sq, k, v,
        batch_size, num_q_heads, num_kv_heads, context_size,
        q_idx,
        q_size,
        o,
        sq.stride(0), sq.stride(1), sq.stride(2), sq.stride(3),
        sq.stride(0), sq.stride(1), sq.stride(2), sq.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q_idx.stride(0), q_idx.stride(1), q_idx.stride(2),
        (q_size + block_size_M - 1) // block_size_M,
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=head_dim,
        dtype=dtype,
        num_warps=4, num_stages=2,
        scale=math.sqrt(head_dim)
    )
    return o


def selected_query_sparse_attention(
    query_selected: torch.Tensor, # [batch_size, num_heads, selected_q_num, head_dim]
    key: torch.Tensor,    # [batch_size, num_heads, context_size, head_dim]
    value: torch.Tensor,  # [batch_size, num_heads, context_size, head_dim]
    q_idx: torch.Tensor,  # [batch_size, num_heads, selected_q_num]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """
    Sparse attention for selected queries with padding (MHA version).

    This function adds padding to ensure dimensions are compatible with block sizes
    and calls the Triton kernel.
    """
    batch_size, num_heads, context_size, head_dim = key.shape

    pad_M = block_size_M - (context_size & (block_size_M -1))
    q_size = q_idx.shape[-1]
    pad_MS = block_size_M - (q_size & (block_size_M - 1))
    query_selected = torch.nn.functional.pad(query_selected, [0, 0, 0, pad_MS, 0, 0, 0, 0])

    pad_N=block_size_N-(context_size&(block_size_N-1))
    key = torch.nn.functional.pad(key, [0, 0, 0, pad_N, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad_N, 0, 0, 0, 0])

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query_selected = torch.nn.functional.pad(query_selected, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    q_idx = q_idx.to(torch.int32).reshape((batch_size, num_heads, -1))
    out = selected_query_attention_entrance(
        query_selected, key, value,
        q_idx, block_size_M, block_size_N
    )
    return out[..., :context_size, :head_dim]


def selected_query_sparse_attention_gqa(
    query_selected: torch.Tensor, # [batch_size, num_q_heads, selected_q_num, head_dim]
    key: torch.Tensor,    # [batch_size, num_kv_heads, context_size, head_dim]
    value: torch.Tensor,  # [batch_size, num_kv_heads, context_size, head_dim]
    q_idx: torch.Tensor,  # [batch_size, num_q_heads, selected_q_num]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    """
    Sparse attention for selected queries with padding (GQA version).

    This function adds padding to ensure dimensions are compatible with block sizes
    and calls the Triton kernel for GQA.
    """
    # Get current device and ensure all tensors are on the same device
    current_device = query_selected.device

    # Verify device consistency (important for multi-GPU PP mode)
    assert key.device == current_device, f"key must be on {current_device}, got {key.device}"
    assert value.device == current_device, f"value must be on {current_device}, got {value.device}"
    assert q_idx.device == current_device, f"q_idx must be on {current_device}, got {q_idx.device}"

    batch_size, num_q_heads, _, head_dim = query_selected.shape
    _, num_kv_heads, context_size, _ = key.shape

    # GQA check
    assert num_q_heads % num_kv_heads == 0, \
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

    q_size = q_idx.shape[-1]

    # Padding for block alignment
    pad_MS = (block_size_M - (q_size % block_size_M)) % block_size_M
    if pad_MS > 0:
        query_selected = torch.nn.functional.pad(query_selected, [0, 0, 0, pad_MS, 0, 0, 0, 0])

    pad_N = (block_size_N - (context_size % block_size_N)) % block_size_N
    if pad_N > 0:
        key = torch.nn.functional.pad(key, [0, 0, 0, pad_N, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, 0, 0, pad_N, 0, 0, 0, 0])

    # Padding for head_dim to be power of 2
    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim))
        pad_dim = target_dim - head_dim
        query_selected = torch.nn.functional.pad(query_selected, [0, pad_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, pad_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, pad_dim, 0, 0, 0, 0, 0, 0])

    q_idx = q_idx.to(torch.int32).reshape((batch_size, num_q_heads, -1))

    out = selected_query_attention_entrance_gqa(
        query_selected, key, value,
        q_idx, block_size_M, block_size_N
    )

    return out[..., :q_size, :head_dim]
