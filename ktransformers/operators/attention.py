'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from torch import nn
import warnings
import torch.nn.functional as F
from ktransformers.operators.models import KLlamaModel
from ktransformers.models.configuration_deepseek import DeepseekV2Config
from ktransformers.models.configuration_llama import LlamaConfig
from ktransformers.models.modeling_llama import LlamaRotaryEmbedding
from ktransformers.models.modeling_deepseek import DeepseekV2Attention, apply_rotary_pos_emb
from typing import Optional, Tuple
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_gguf import GGUFLoader
import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache
import triton
import triton.language as tl
logger = logging.getLogger("attention")
class KDeepseekV2Attention(BaseInjectedModule, DeepseekV2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    attn_mask: Optional[torch.Tensor] = None

    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.

    def get_absorbed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (hasattr(self, 'q_absorb') and hasattr(self, 'out_absorb')):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
            q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :].reshape(-1, self.kv_lora_rank)
            out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :].reshape(-1, self.kv_lora_rank)
            self.q_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, 
                                      bias=False, dtype=q_absorb.dtype, device=q_absorb.device)
            self.q_absorb.weight.data = q_absorb
            self.out_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, 
                                        bias=False, dtype=out_absorb.dtype, device=out_absorb.device)
            self.out_absorb.weight.data = out_absorb
            del self.orig_module.kv_b_proj
        q_absorb = self.q_absorb.weight.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        out_absorb = self.out_absorb.weight.view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        return q_absorb, out_absorb

    def forward_chunck(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            compressed_kv = compressed_kv.unsqueeze(1)
            k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
            compressed_kv = compressed_kv.squeeze(1)
            #if cache_position is not None:  
            #    compressed_kv = compressed_kv[:,: cache_position[-1] + 1,:]
            #    k_pe = k_pe[:,:,: cache_position[-1] + 1,:]
        q_absorb, out_absorb = self.get_absorbed()

        q_nope = torch.matmul(q_nope, q_absorb)
        attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)) * self.softmax_scale
        """
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        """
        if attention_mask is not None:
            """
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            """
            #causal_mask = attention_mask[:, :, :, : kv_seq_len]
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_pe.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)

        attn_output = torch.matmul(attn_output, out_absorb.mT) 

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()
        
        if q_len <= self.chunck_size:
            return self.forward_chunck(
                            hidden_states,
                            attention_mask,
                            position_ids,
                            past_key_value,
                            output_attentions,
                            use_cache,
                            cache_position,
                            **kwargs
                        )

        assert output_attentions == False, "output_attentions is not supported when using chunked attention"
        attn_output = None
        cur_idx = 0
        while cur_idx < q_len:
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, cur_idx:min(cur_idx + self.chunck_size, q_len), ...]
            else:
                # generate chunk_mask automatically.
                self.attn_mask = \
                    torch.zeros(1, 1, self.chunck_size, past_key_value.max_cache_len, device=hidden_states.device) \
                        if self.attn_mask is None \
                            else self.attn_mask
                self.attn_mask[:, :, :, cur_idx:min(cur_idx+self.chunck_size, past_key_value.max_cache_len)] = \
                    -1e+38 * torch.triu(torch.ones(self.chunck_size, self.chunck_size, device=hidden_states.device), diagonal=1)\
                        [:,:min(self.chunck_size, min(past_key_value.max_cache_len-cur_idx, self.chunck_size))]
                self.attn_mask[:, :, :, cur_idx+self.chunck_size:] = -1e+38
                self.attn_mask[:, :, :, :cur_idx] = 0
                chunk_mask = torch.narrow(self.attn_mask, 2, 0, min(self.chunck_size, q_len-cur_idx))

            cur_output, _, _ = self.forward_chunck(
                            hidden_states[:, cur_idx:min(cur_idx + self.chunck_size, q_len), ...],
                            chunk_mask,
                            position_ids[:, cur_idx:min(cur_idx + self.chunck_size, q_len)],
                            past_key_value,
                            output_attentions,
                            use_cache,
                            cache_position[cur_idx:min(cur_idx + self.chunck_size, q_len)],
                            **kwargs
                        )
            cur_idx += self.chunck_size
            if attn_output is None:
                attn_output = cur_output
            else:
                attn_output = torch.cat((attn_output, cur_output), dim=-2)
                
        return attn_output, None, past_key_value
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)




class KLlamaAttention(BaseInjectedModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:

            logger.warning(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if q_len == 1:
            position_ids = position_ids[0][-1].unsqueeze(0).unsqueeze(0)
            query_states = query_states[:, :, -1:]
            key_states = key_states[:, :, -1:]

        attn_output = KLlamaModel.dynamic_sdpa.apply(
            self.layer_idx,
            bsz,
            position_ids[0][0],
            query_states.transpose(1, 2).to(torch.float16),
            key_states.transpose(1, 2).to(torch.float16),
            value_states.transpose(1, 2).to(torch.float16),
            mode="prefill" if q_len > 1 else "generate",
        )


        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

# triton sparse attn
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
    # grid 的两个维度（2，1）
    start_m = tl.program_id(0)  # 当前是第几块（在 q_size 维度）
    off_hz = tl.program_id(1)  # batch_size*num_heads 维上的第几块

    batch_idx = off_hz // num_heads
    head_idx = off_hz % num_heads  
    # 当前pid 在 query 维度的偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # [0,... 15]
    # 当前pid 在 context 维度的偏移
    offs_n = tl.arange(0, BLOCK_N) # [0,... 15]
    # 当前pid 在 head_dim 维度的偏移
    offs_d = tl.arange(0, BLOCK_DMODEL) # [0,... head_dim]
    # query 起始偏移量 在前两个维度
    sq_offset = (off_hz // num_heads) * stride_sqz + (off_hz % num_heads) * stride_sqh
    o_offset = (off_hz // num_heads) * stride_oz + (off_hz % num_heads) * stride_oh
 

    # key 起始偏移量 在前两个维度
    kv_offset = (off_hz // num_heads) * stride_kz + (off_hz % num_heads) * stride_kh
    
    if start_m * BLOCK_M >= q_size:
        return
     # 加载 q_idx 并确保不越界

    cols_ptr = q_idx + batch_idx * stride_idxz + head_idx * stride_idxh + start_m * BLOCK_M * stride_idxm
    cols_mask = offs_m < q_size
    # 特定 query 的索引
    # q_cols = tl.load(cols_ptr + tl.arange(0, BLOCK_M),  mask=cols_mask, other=0)
    q_cols = tl.load(cols_ptr + offs_m % BLOCK_M,  mask=cols_mask, other=0)

    # 当前 query索引中的最大值
    max_qcol = tl.max(q_cols, axis=0)

    q_ptrs = Q_selected + sq_offset + offs_m[:, None] * stride_sqm + offs_d[None, :] * stride_sqk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk # [[0],... [15]]
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk # [[0,... 15]]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

   
    # 确保 q_cols 不越界
    valid_mask = q_cols < context_size
    # 从Q_selected的q_ptrs load q
    q = tl.load(q_ptrs)


    # 每个query 最大注意力权重
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # 每个query 累计权重
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # 累计 o
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # qk_scale= head_dim ** -0.5 / 1.44269504
    qk_scale =  1 / scale
    q = (q * qk_scale).to(dtype)
    # 按块遍历上下文
    for start_n in range(0, max_qcol + 1, BLOCK_N):
        # 当前块的context 索引
        cols = start_n + offs_n # [BLOCK_N]
        n_mask = cols < context_size # [BLOCK_N]
        # k_mask = cols <= q_cols # [BLOCK_M, BLOCK_N]
        # cols[None, :] [1, BLOCK_N]
        k_mask = cols <= max_qcol
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)


        # 边界检查，确保 cols 和 q_cols 都在合法范围内
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
    # 在写回输出之前也要确保不越界
    tl.store(o_ptrs, acc.to(dtype), mask=offs_m[:, None] < q_size)

def selected_query_attention_entrance(
    sq: torch.Tensor,  # [batch_size, num_heads, selected_q_num, head_dim]
    k: torch.Tensor,  # [batch_size, num_heads, context_size, head_dim]
    v: torch.Tensor,  # [batch_size, num_heads, context_size, head_dim]
    q_idx: torch.Tensor,  # [batch_size, num_heads, selected_q_num]
    block_size_M: int = 64, 
    block_size_N: int = 64, 
):
    dq, dk, dv = sq.shape[-1], k.shape[-1], v.shape[-1]
    assert dq == dk and dk == dv
    # assert dk in {16, 32, 64, 128, 256, 512}

    q_size = q_idx.shape[-1]
    batch_size, num_heads, context_size, head_dim = k.shape

    o = torch.zeros(batch_size, num_heads, q_size, head_dim, device=sq.device)
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

def selected_query_sparse_attention(
    query_selected: torch.Tensor, # [batch_size, num_heads, selected_q_num, head_dim]
    key: torch.Tensor,    # [batch_size, num_heads, context_size, head_dim]
    value: torch.Tensor,  # [batch_size, num_heads, context_size, head_dim]
    q_idx: torch.Tensor,  # [batch_size, num_heads, selected_q_num]
    block_size_M: int = 64, 
    block_size_N: int = 64, 
):
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