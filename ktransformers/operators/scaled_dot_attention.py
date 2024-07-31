import torch
from ktransformers.models.configuration_internlm2 import InternLM2Config
import sys, os

# sys.path.append(os.path.dirname(__file__) + "/../pcinfer/pcinfer")
sys.path.append("/root/internlm2_5-7b-chat-1m/pcinfer/pcinfer")
from pcinfer import PCInfer, PCInferKVCache
from flash_attn import flash_attn_func, flash_attn_with_kvcache


class DynamicScaledDotAttention:

    def __init__(
        self,
        max_seq_len: int,
        block_size: int,
        config: InternLM2Config,
        device: torch.device,
        local_windows_len: int,
        topk: int,
        threads_num: int,
        anchor_type: str = "DYNAMIC",
        kv_type: str = "FP16",
        dense_layer_num: int = 0,
        anchor_num: int = 1,
    ):
        assert anchor_num == 1
        assert anchor_type == "DYNAMIC"

        self.max_seq_len = max_seq_len
        self.block_num = max_seq_len // block_size
        self.block_size = block_size

        # model config
        self.kv_head_num = config.num_key_value_heads
        self.q_head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_num = config.num_hidden_layers

        self.device = device
        self.local_windows_len = local_windows_len
        self.topk = topk
        self.dense_layer_num = dense_layer_num
        # self.dense_layer_num = 32
        self.cache_key_states = torch.zeros(
            (self.block_num, block_size, self.kv_head_num, self.head_dim),
            device=device,
            dtype=torch.float16,
        )
        self.cache_value_states = torch.zeros(
            (self.block_num, block_size, self.kv_head_num, self.head_dim),
            device=device,
            dtype=torch.float16,
        )

        self.cache_importance = torch.zeros(
            (self.block_num, block_size, self.q_head_num),
            device=device,
            dtype=torch.float16,
        )

        self.prefix_block_table = torch.arange(
            self.block_num, device=device, dtype=torch.int32
        ).view(1, -1)

        self.pcinfer = PCInfer(threads_num)

        self.local_thread = PCInferKVCache(
            self.layer_num,
            self.kv_head_num,
            self.q_head_num,
            self.head_dim,
            self.block_size,
            anchor_num=anchor_num,
            anchor_type=anchor_type,
            kv_type=kv_type,
            max_batch_size=1,
            max_block_num=self.block_num,
            max_thread_num=threads_num,
        )

        self.shape_mask = (
            self.q_head_num,
            self.block_size,
            self.block_size,
        )

        mask = torch.zeros(
            self.shape_mask, dtype=torch.uint8, device=device
        ).contiguous()
        elm_idx = torch.arange(self.block_size, device=device)

        for i in range(mask.size(-2)):
            idx = i + mask.size(-1) - mask.size(-2) - elm_idx
            idx = idx[idx >= 0]
            mask[..., i, idx] = 1

        self.tril_mask = mask
        self.triu_mask = mask ^ 1

    def get_attn_score_one_block(
        self,
        batch_idx: int,
        max_block_num: int,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: int,
        width: int,
        mask_mode: str | None = None,
    ):
        n_rep = self.q_head_num // self.kv_head_num
        key = key[..., None, :].expand(
            key.size(0), self.kv_head_num, n_rep, self.head_dim
        )
        key = key.reshape(
            key.size(0),
            self.q_head_num,
            self.head_dim,
        )
        qk = torch.einsum(
            "qhd,khd->hqk", query, key
        )  # (num_attention_heads, len_q, len_k)

        if mask_mode == "tril":
            mask = self.tril_mask
            mask = mask[..., -qk.size(-2) :, -qk.size(-1) :]
            qk = qk * mask
        elif mask_mode == "triu":
            mask = self.triu_mask
            mask = mask[..., -qk.size(-2) :, -qk.size(-1) :]
            qk = qk * mask
        qk = torch.sum(qk, dim=-2) / self.block_size
        importance = self.cache_importance.view(-1, self.q_head_num)
        importance = importance.narrow(0, batch_idx * max_block_num + offset, width)
        importance += qk.transpose(-1, -2)

    # key: [bsz, past_len, head_num, head_dim] float16
    # query: [bsz, q_len, q_head_num, head_dim] float16
    def get_attn_score(
        self,
        layer_idx: int,
        batch_size: int,
        offset: torch.Tensor,
        width: int,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        max_seqs_len = offset.max().item() + width
        max_block_num = (max_seqs_len + self.block_size - 1) // self.block_size

        for batch_idx in range(batch_size):
            for idx in range(width // self.block_size):
                offset_cur = idx * self.block_size
                query_cur = query[batch_idx, offset_cur : offset_cur + self.block_size]
                self.get_attn_score_one_block(
                    batch_idx,
                    max_block_num,
                    query_cur,
                    key[
                        batch_idx,
                        offset[batch_idx]
                        + offset_cur : offset[batch_idx]
                        + offset_cur
                        + self.block_size,
                    ],
                    offset[batch_idx].item() + offset_cur,
                    self.block_size,
                    mask_mode="tril",
                )

                offset_key = (
                    offset[batch_idx].item()
                    + idx * self.block_size
                    - self.local_windows_len
                )
                if offset_key >= 0:
                    self.get_attn_score_one_block(
                        batch_idx,
                        max_block_num,
                        query_cur,
                        key[batch_idx, offset_key : offset_key + self.block_size],
                        offset_key,
                        self.block_size,
                        mask_mode="triu",
                    )

                offset_key = max(0, offset_key + self.block_size)
                width_key = (
                    offset[batch_idx].item() + idx * self.block_size - offset_key
                )
                if width_key > 0:
                    self.get_attn_score_one_block(
                        batch_idx,
                        max_block_num,
                        query_cur,
                        key[batch_idx, offset_key : offset_key + width_key],
                        offset_key,
                        width_key,
                        mask_mode=None,
                    )

        importance_cache = self.cache_importance.narrow(
            0, 0, max_block_num * batch_size
        ).view(batch_size, max_block_num * self.block_size, self.q_head_num)
        importance_cache_cpu = torch.empty_like(
            importance_cache, device="cpu", pin_memory=True
        )

        importance_cache_cpu.copy_(importance_cache)

        block_table_cpu = self.prefix_block_table[:,:max_block_num].to("cpu")
        offset_cpu = offset.contiguous().to("cpu")

        self.pcinfer.submit(
            self.local_thread.update_importance(
                importance_cache_cpu,
                layer_idx,
                block_table_cpu,
                max_block_num,
                offset_cpu,
                width,
            )
        )
        self.pcinfer.sync()
        importance_cache.zero_()

    # key: [bsz, q_len, head_num, head_dim] float16
    # value: [bsz, q_len, head_num, head_dim] float16
    def swap_in_and_swap_out(self, layer_idx, past_len, q_len, key, value):
        batch_size = 1
        max_seqs_len = past_len.max().item() + q_len
        max_block_num = (max_seqs_len + self.block_size - 1) // self.block_size
        k_cache = self.cache_key_states.narrow(0, 0, max_block_num * batch_size).view(
            batch_size, max_block_num * self.block_size, self.kv_head_num, self.head_dim
        )
        v_cache = self.cache_value_states.narrow(0, 0, max_block_num * batch_size).view(
            batch_size, max_block_num * self.block_size, self.kv_head_num, self.head_dim
        )

        for batch_idx in range(batch_size):
            offset = past_len[batch_idx]
            width = q_len
            k_cache[batch_idx][offset : offset + width].copy_(
                key[batch_idx].view(-1, self.kv_head_num, self.head_dim)
            )
            v_cache[batch_idx][offset : offset + width].copy_(
                value[batch_idx].view(-1, self.kv_head_num, self.head_dim)
            )

        k_cache_cpu = torch.empty_like(k_cache, device="cpu", pin_memory=True)
        v_cache_cpu = torch.empty_like(v_cache, device="cpu", pin_memory=True)

        k_cache_cpu.copy_(k_cache)
        v_cache_cpu.copy_(v_cache)

        cur_block_num = (
            q_len + past_len[0].item() + self.block_size - 1
        ) // self.block_size
        block_table_cpu = self.prefix_block_table[:,:cur_block_num].to("cpu")
        past_len_cpu = past_len.contiguous().to("cpu")

        self.pcinfer.submit(
            self.local_thread.get_and_update_fp16(
                k_cache_cpu,
                v_cache_cpu,
                layer_idx,
                block_table_cpu,
                max_block_num,
                past_len_cpu,
                q_len,
            )
        )

        self.pcinfer.sync()
        k_cache.copy_(k_cache_cpu)
        v_cache.copy_(v_cache_cpu)

        return k_cache, v_cache

    def calc_anchor(self, cache_seqlens: int):
        cur_block_num = (cache_seqlens + self.block_size - 1) // self.block_size
        block_table_cpu = self.prefix_block_table[:,:cur_block_num].to("cpu")
        cache_seqlens_cpu = torch.tensor(
            [cache_seqlens], device="cpu", dtype=torch.int32
        )

        self.pcinfer.submit(
            self.local_thread.calc_anchor_all_layers(
                block_table_cpu,
                cache_seqlens_cpu,
            )
        )
        self.pcinfer.sync()

    def clear_importance(self, cache_seqlens: int):
        cur_block_num = (cache_seqlens + self.block_size - 1) // self.block_size
        block_table_cpu = self.prefix_block_table[:,:cur_block_num].to("cpu")
        cache_seqlens_cpu = torch.tensor(
            [cache_seqlens], device="cpu", dtype=torch.int32
        )

        self.pcinfer.submit(
            self.local_thread.clear_importance_all_layers(
                block_table_cpu,
                cache_seqlens_cpu,
            )
        )
        self.pcinfer.sync()



    def apply(
        self,
        layer_idx: int,
        bsz: int,
        past_len: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mode: str = "prefill",
    ):
        # key_states: [bsz, q_len, kv_head_num, head_dim]
        # value_states: [bsz, q_len, kv_head_num, head_dim]
        # query_states: [bsz, q_len, q_head_num, head_dim]
        assert query_states.dtype == torch.float16
        assert key_states.dtype == torch.float16
        assert value_states.dtype == torch.float16

        assert key_states.size(2) == self.kv_head_num
        assert value_states.size(2) == self.kv_head_num
        assert query_states.size(2) == self.q_head_num

        q_len = query_states.size(1)
        batch_size = query_states.size(0)
        past_len = torch.tensor(
            [past_len], device=query_states.device, dtype=torch.int32
        )
        device = query_states.device

        if mode == "prefill":
            key, value = self.swap_in_and_swap_out(
                layer_idx,
                past_len,
                q_len,
                key_states,
                value_states,
            )
            self.get_attn_score(
                layer_idx,
                bsz,
                past_len,
                q_len,
                query_states,
                key,
            )
            output = flash_attn_with_kvcache(
                q=query_states,
                k_cache=key,
                v_cache=value,
                cache_seqlens=past_len + q_len,
                causal=True,
            )

        elif mode == "generate":
            output = torch.empty_like(query_states, device="cpu").contiguous()
            lse = torch.empty(
                (batch_size, q_len, self.q_head_num), device="cpu", dtype = torch.float32
            ).contiguous()

            q_in_cpu = query_states.contiguous().to("cpu")
            k_in_cpu = key_states.contiguous().to("cpu")
            v_in_cpu = value_states.contiguous().to("cpu")
            cache_seqlens_cpu = past_len.contiguous().to("cpu")

            cur_block_num = (
                q_len + past_len[0].item() + self.block_size - 1
            ) // self.block_size
            block_table_cpu = (
                self.prefix_block_table[:,:cur_block_num].contiguous().to("cpu")
            )

            if layer_idx < self.dense_layer_num:
                self.pcinfer.submit(
                    self.local_thread.attn_with_kvcache(
                        q_in=q_in_cpu,
                        k_in=k_in_cpu,
                        v_in=v_in_cpu,
                        output=output,
                        attn_lse=lse,
                        layer_idx=layer_idx,
                        block_table=block_table_cpu,
                        cache_seqlens=cache_seqlens_cpu,
                    )
                )
            else:
                self.pcinfer.submit(
                    self.local_thread.attn_with_kvcache(
                        q_in=q_in_cpu,
                        k_in=k_in_cpu,
                        v_in=v_in_cpu,
                        output=output,
                        attn_lse=lse,
                        layer_idx=layer_idx,
                        block_table=block_table_cpu,
                        cache_seqlens=cache_seqlens_cpu,
                        topk=self.topk,
                        local=self.local_windows_len // self.block_size,
                    )
                )
            self.pcinfer.sync()
            output = output.to(device)

    

        return output.transpose(1, 2)
