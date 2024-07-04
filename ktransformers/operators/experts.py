# Copyright 2024 Shaoyuan Chen
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

from socket import NETLINK_ROUTE
from typing import Any, Union
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import sys, os
from operators.base_operator import BaseInjectedModule

sys.path.append(os.path.dirname(__file__) + "/../third_party/pcinfer/build")
import pcinfer
from pcinfer.moe import MOEConfig, MOE
import ctypes
from util.custom_gguf import GGUFLoader
from transformers.configuration_utils import PretrainedConfig

# from gguf.constants import GGMLQuantizationType
# from gguf.quants import quant_shape_to_byte_shape, GGML_QUANT_SIZES
from multiprocessing import cpu_count

pc_infer = pcinfer.PCInfer(cpu_count() - 4)


class MLPExperts(BaseInjectedModule):
    readers: dict = {}  # {filename: GGUFReader}

    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)

    def load(self):
        print("loading MLPExperts", self.key)
        self.gate = self.gguf_loader.get_mmap_tensor(self.key + ".ffn_gate_exps.weight")
        self.up = self.gguf_loader.get_mmap_tensor(self.key + ".ffn_up_exps.weight")
        self.down = self.gguf_loader.get_mmap_tensor(self.key + ".ffn_down_exps.weight")
        self.gate_type = self.gguf_loader.tensor_info[
            self.key + ".ffn_gate_exps.weight"
        ]["ggml_type"]
        self.up_type = self.gguf_loader.tensor_info[self.key + ".ffn_up_exps.weight"][
            "ggml_type"
        ]
        self.down_type = self.gguf_loader.tensor_info[
            self.key + ".ffn_down_exps.weight"
        ]["ggml_type"]
        gate_ptr = ctypes.addressof(
            ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        up_ptr = ctypes.addressof(
            ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        down_ptr = ctypes.addressof(
            ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
        )
        # print(self.gate_qtype, self.up_qtype, self.down_qtype)
        n_routed_experts = len(self.orig_module)
        moe_config = MOEConfig(
            n_routed_experts,
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            64,
            gate_ptr,
            up_ptr,
            down_ptr,
            self.gate_type,
            self.up_type,
            self.down_type,
            30,
        )
        # print(n_routed_experts, hidden_size, moe_intermediate_size)
        self.moe = MOE(moe_config)
        self.pc_infer = pc_infer

    def forward(self, input_tensor, expert_ids, weights):
        input_tensor = input_tensor.contiguous()
        expert_ids = expert_ids.contiguous()
        weights = weights.contiguous().to(torch.float32)
        # torch.set_printoptions(profile="full")
        # print(input_tensor.dtype, expert_ids.dtype, weights.dtype)
        # print(torch.isinf(input_tensor).any()) # prints the whole tensor
        # print(torch.isinf(expert_ids).any()) # prints the whole tensor
        # print(torch.isinf(weights).any()) # prints the whole tensor
        # torch.set_printoptions(profile="default") # reset
        output = torch.empty_like(input_tensor).contiguous()
        self.pc_infer.submit(
            self.moe.forward,
            expert_ids.size(0),
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output.data_ptr(),
        )
        self.pc_infer.sync()
        # print("MLPExperts output",torch.isinf(output).any())
        return output


from models.modeling_deepseek import DeepseekV2MoE
from models.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock


class Qwen2MoeSparseMoeBlockInjected(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states_cpu = hidden_states.cpu()
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_experts_cpu = selected_experts.cpu()
        routing_weights_cpu = routing_weights.cpu()

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        if isinstance(self.experts, MLPExperts):
            y = (
                self.moe_on_pcinfer(
                    hidden_states_cpu, selected_experts_cpu, routing_weights_cpu
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_cpu.size(0) > 10:
            y = self.moe_infer(
                hidden_states_cpu, selected_experts_cpu, routing_weights_cpu, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_cpu, selected_experts_cpu, routing_weights_cpu
            ).to(device=hidden_states.device)
        y += shared_expert_output
        y.resize_(*orig_shape)
        return y, router_logits

    @torch.no_grad()
    def moe_on_pcinfer(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        outs = torch.empty_like(x)
        for token_idx in range(topk_ids.size(0)):
            outs[token_idx] = self.experts(
                x[token_idx], topk_ids[token_idx], topk_weight[token_idx]
            )
        return outs

    @torch.no_grad()
    def moe_infer_simple(
        self,
        hidden_states_cpu: torch.Tensor,
        selected_experts_cpu: torch.Tensor,
        routing_weights_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(hidden_states_cpu[token_idx])
                    * routing_weights_cpu[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    def moe_infer(
        self,
        hidden_states_cpu: torch.Tensor,
        selected_experts_cpu: torch.Tensor,
        routing_weights_cpu: torch.Tensor,
        orig_shape: tuple,
    ) -> torch.Tensor:

        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states_cpu.dtype,
            device=hidden_states_cpu.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts_cpu, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer.forward_cpu(current_state)
                * routing_weights_cpu[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states_cpu.dtype)
            )

        return final_hidden_states


class DeepseekV2MoEInjected(BaseInjectedModule, DeepseekV2MoE):
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        hidden_states_cpu = hidden_states.cpu()
        topk_idx_cpu = topk_idx.cpu()
        topk_weight_cpu = topk_weight.cpu()
        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity)

        if isinstance(self.experts, MLPExperts):
            y = (
                self.moe_on_pcinfer(hidden_states_cpu, topk_idx_cpu, topk_weight_cpu)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_cpu.size(0) > 10:
            y = (
                self.moe_infer(hidden_states_cpu, topk_idx_cpu, topk_weight_cpu)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            y = (
                self.moe_infer_simple(hidden_states_cpu, topk_idx_cpu, topk_weight_cpu)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_on_pcinfer(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        # print("x", x)
        # print("topk_ids", topk_ids)
        # print("topk_weight", topk_weight)
        outs = torch.empty_like(x)
        for token_idx in range(topk_ids.size(0)):
            # print(token_idx)
            # print(x[token_idx])
            # print(topk_ids[token_idx])
            # print(topk_weight[token_idx])
            outs[token_idx] = self.experts(
                x[token_idx], topk_ids[token_idx], topk_weight[token_idx]
            )
        return outs

    @torch.no_grad()
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward_cpu(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out
