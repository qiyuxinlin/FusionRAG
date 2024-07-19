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

from typing import Any, Union
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import sys, os
from ktransformers.operators.base_operator import BaseInjectedModule

sys.path.append(os.path.dirname(__file__) + "/../ktransformers_ext/build")
import cpuinfer_ext
from cpuinfer_ext.moe import MOEConfig, MOE
import ctypes
from ktransformers.util.custom_gguf import GGUFLoader
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod
from ktransformers.operators.linear import QuantizedLinearMarlin, QuantizedLinearTorch, KTransformerLinear


# from gguf.constants import GGMLQuantizationType
# from gguf.quants import quant_shape_to_byte_shape, GGML_QUANT_SIZES
# from multiprocessing import cpu_count

cpu_infer = cpuinfer_ext.CPUInfer(60) # TODO: Auto generate thread_num, or set by user

# class Base(BaseInjectedModule, ABC):
class MLPExpertsBase(ABC):
    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, device: str = "cuda", **kwargs):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device
    
    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None):
        pass
    
    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
                # tensors = self.load_multi(key, [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight"])    
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res

class MLPCPUExperts(MLPExpertsBase):
    input_tensor_cpu:Tensor = None
    expert_ids_cpu:Tensor = None
    weights_cpu:Tensor = None
    output_cpu:Tensor = None
    output_gpu:Tensor = None
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cuda", # this device mean which device the output should on 
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        assert device.lower() == "cpu", "MLPCPUExperts can only be loaded on CPU"
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device

    def load(self, w: dict | nn.Parameter | tuple | None = None, device:str|None = None):
        if device:
            assert device.lower() == "cpu", "MLPCPUExperts can only be loaded on CPU, Parameter \"device\" can be cpu or None."
        if w is None: w = self.load_weights()[self.key]
        self.gate = w["gate"]
        self.up = w["up"]
        self.down = w["down"]
        self.gate_type = w["gate_type"]
        self.up_type = w["up_type"]
        self.down_type = w["down_type"]
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
        n_routed_experts = self.n_routed_experts
        # n_routed_experts = len(self.orig_module)
        moe_config = MOEConfig(
            n_routed_experts,
            self.config.num_experts_per_tok,
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            64,
            10,
            1024,
            gate_ptr,
            up_ptr,
            down_ptr,
            self.gate_type,
            self.up_type,
            self.down_type,
            30, # TODO: get from model.dtype
        )
        # print(n_routed_experts, hidden_size, moe_intermediate_size)
        num_experts_per_tok = self.config.num_experts_per_tok
        self.moe = MOE(moe_config)
        self.cpu_infer = cpu_infer
        self.cpu_infer.submit(self.moe.warm_up)
        self.cpu_infer.sync()
        if MLPCPUExperts.output_gpu == None:
            MLPCPUExperts.input_tensor_cpu = torch.empty((self.config.hidden_size), device="cpu", pin_memory=True)
            MLPCPUExperts.expert_ids_cpu = torch.empty((num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
            MLPCPUExperts.weights_cpu = torch.empty((num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
            MLPCPUExperts.output_cpu = torch.empty((self.config.hidden_size), device="cpu", pin_memory=True)
            MLPCPUExperts.output_gpu = torch.empty((self.config.hidden_size), device=self.out_device)

    def submit_for_one_decode(self, input_tensor, expert_ids, weights):
        MLPCPUExperts.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
        MLPCPUExperts.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
        MLPCPUExperts.weights_cpu.copy_(weights, non_blocking=True)
        self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, self.moe.forward, 1, expert_ids.size(0), MLPCPUExperts.expert_ids_cpu.data_ptr(), MLPCPUExperts.weights_cpu.data_ptr(), MLPCPUExperts.input_tensor_cpu.data_ptr(), MLPCPUExperts.output_cpu.data_ptr())
    
    def sync_for_one_decode(self):
        self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
        MLPCPUExperts.output_gpu.copy_(MLPCPUExperts.output_cpu, non_blocking=True)
        #print("capturing experts finish")
        return MLPCPUExperts.output_gpu

    def forward(self, input_tensor, expert_ids, weights):
        # generate, capture and run cuda graph
        if input_tensor.size(0)==1:
            #print("capturing experts")
            MLPCPUExperts.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            MLPCPUExperts.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
            MLPCPUExperts.weights_cpu.copy_(weights, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, self.moe.forward, 1, expert_ids.size(1), MLPCPUExperts.expert_ids_cpu.data_ptr(), MLPCPUExperts.weights_cpu.data_ptr(), MLPCPUExperts.input_tensor_cpu.data_ptr(), MLPCPUExperts.output_cpu.data_ptr())
            self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
            MLPCPUExperts.output_gpu.copy_(MLPCPUExperts.output_cpu, non_blocking=True)
            #print("capturing experts finish")
            return MLPCPUExperts.output_gpu
        else:
            input_tensor = input_tensor.contiguous().cpu()
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            output = torch.empty_like(input_tensor).contiguous()
            self.cpu_infer.submit(self.moe.forward, expert_ids.size(0), expert_ids.size(1), expert_ids.data_ptr(), weights.data_ptr(), input_tensor.data_ptr(), output.data_ptr())
            self.cpu_infer.sync()
            return output.to(device=object.__getattribute__(self, "device"))
    
    def unload(self):
        return

class MLPExpertsMarlin(MLPExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        self.device = device
        # create empty marlin experts according to the number of experts per token
        # up
        self.up_projs = [QuantizedLinearMarlin(key+ "." + "ffn_up_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # gate
        self.gate_projs = [QuantizedLinearMarlin(key+ "." + "ffn_gate_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # down
        self.down_projs = [QuantizedLinearMarlin(key+ "." + "ffn_down_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None):
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        if w is None: w = self.load_weights()[self.key]

        if isinstance(w, dict):
            self.gate = nn.Parameter(torch.from_numpy(w["gate"]))
            self.up = nn.Parameter(torch.from_numpy(w["up"]))
            self.down = nn.Parameter(torch.from_numpy(w["down"]))
            for i in range(self.expert_num):
                self.up_projs[i].load(self.up[i,...], device=device)
                self.gate_projs[i].load(self.gate[i,...], device=device)
                self.down_projs[i].load(self.down[i,...], device=device)
                self.loaded_experts_idx.append(i)
        return 

    def unload(self):
        for i in self.loaded_experts_idx:
            self.up_projs[i].unload()
            self.gate_projs[i].unload()
            self.down_projs[i].unload()
        self.loaded_experts_idx = []

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.load_gguf_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.load_gguf_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.load_gguf_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
                # tensors = self.load_multi(key, [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight"])    
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res

    def forward(self, input_tensor:torch.Tensor, expert_ids, weights):
        # forward
        device = input_tensor.device
        input_tensor = input_tensor.to("cuda")
        outs = torch.zeros_like(input_tensor)
        for expert_idx in range(expert_ids.size(0)):
            down_proj = self.down_projs[expert_idx]
            gate_proj = self.gate_projs[expert_idx]
            up_proj = self.up_projs[expert_idx]

            outs += down_proj(self.act_fn(gate_proj(input_tensor)) * up_proj(input_tensor)) * weights[expert_idx]
        outs = outs.to(device)
        return outs

class MLPExpertsTorch(MLPExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        self.device = device
        # create empty marlin experts according to the number of experts 
        self.up_projs = [QuantizedLinearTorch(key+ "." + "ffn_up_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        self.gate_projs = [QuantizedLinearTorch(key+ "." + "ffn_gate_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        self.down_projs = [QuantizedLinearTorch(key+ "." + "ffn_down_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None):
        if device is None: device = self.device
        if w is None: w = self.load_weights()[self.key]

        if isinstance(w, dict):
            
            self.gate = torch.from_numpy(w["gate"])
            self.up = torch.from_numpy(w["up"])
            self.down = torch.from_numpy(w["down"])
            for i in range(self.expert_num):
                self.up_projs[i].load(nn.Parameter(self.up[i,...]), device=device)
                self.gate_projs[i].load(nn.Parameter(self.gate[i,...]), device=device)
                self.down_projs[i].load(nn.Parameter(self.down[i,...]), device=device)
                self.loaded_experts_idx.append(i)
        return 

    def unload(self):
        for i in self.loaded_experts_idx:
            self.up_projs[i].unload()
            self.gate_projs[i].unload()
            self.down_projs[i].unload()
        self.loaded_experts_idx = []

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.load_gguf_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.load_gguf_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.load_gguf_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
                # tensors = self.load_multi(key, [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight"])    
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res

    def forward(self, input_tensor:torch.Tensor, expert_ids, weights):
        # forward
        device = input_tensor.device
        input_tensor = input_tensor.to("cuda")
        outs = torch.zeros_like(input_tensor)
        for expert_idx in range(expert_ids.size(0)):
            down_proj = self.down_projs[expert_idx]
            gate_proj = self.gate_projs[expert_idx]
            up_proj = self.up_projs[expert_idx]
            outs += down_proj.forward(self.act_fn(gate_proj.forward(input_tensor)) * up_proj.forward(input_tensor)) * weights[expert_idx]
        outs = outs.to(device)
        return outs

GPU_EXPERTS_MAP={
    "MLPExpertsMarlin": MLPExpertsMarlin,
    "MLPExpertsTorch": MLPExpertsTorch
}

CPU_EXPERTS_MAP={
    "MLPCPUExperts": MLPCPUExperts,
    "MLPExpertsTorch": MLPExpertsTorch
}

class KTransformersMLPExpert(BaseInjectedModule, MLPExpertsBase):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 prefill_device:str = "cpu",
                 gpu_mlp_type: str | None = None,
                 cpu_mlp_type: str | None = None,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        MLPExpertsBase.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        if cpu_mlp_type is not None:
            self.cpu_experts = CPU_EXPERTS_MAP[cpu_mlp_type](key, gguf_loader, config, len(orig_module), device="cpu", **kwargs)
        else:
            self.cpu_experts = None
        if gpu_mlp_type is not None:
            self.gpu_experts = GPU_EXPERTS_MAP[gpu_mlp_type](key, gguf_loader, config, len(orig_module), device="cuda", **kwargs)
        else:
            self.gpu_experts = None
        self.gpu_mlp_type = gpu_mlp_type
        self.cpu_mlp_type = cpu_mlp_type
        self.current_device = device

    def load(self, w: dict = None, device: str= None):
        # TODO support w as input
        if device is None: device = self.device
        # load to device
        if self.current_device.lower() == "cpu":
            print(f'loading {self.key} to {self.device} from class {self.cpu_mlp_type}')
            self.cpu_experts.load()
        else:
            print(f'loading {self.key} to {self.device} from class {self.gpu_mlp_type}')
            self.gpu_experts.load()
        self.current_device = device

    def unload(self):
        if self.current_device.lower() == "cpu":
            self.cpu_experts.unload()
        else:
            self.gpu_experts.unload()
        self.current_device = "cpu"

    def forward(self, input_tensor, expert_ids, weights):
        if self.current_device == "cpu":
            assert self.cpu_experts is not None, "cpu_experts is None"
            return self.cpu_experts.forward(input_tensor, expert_ids, weights)
        else:
            assert self.gpu_experts is not None, "gpu_experts is None"
            return self.gpu_experts.forward(input_tensor, expert_ids, weights)
        
    def load_to(self, target):
        print(f"loading {self.key} to {target}")
        if isinstance(target, str) and target == "cpu":
            self.cpu_experts.load(device=target)
            if self.gpu_experts is not None:
                self.gpu_experts.unload()
            self.current_device = target
        elif isinstance(target, str) and "cuda" in target.lower():
            self.gpu_experts.load(device=target)
            if self.cpu_experts is not None:
                self.cpu_experts.unload()
            self.current_device = target
        elif isinstance(target, str) and target == "restore":
            self.load_to(self.device)
        else:
            raise ValueError("target must be either \"cpu\", \"cuda\", \"cuda:idx\" or \"restore\"")

from ktransformers.models.modeling_deepseek import DeepseekV2MoE
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock


class Qwen2MoeSparseMoeBlockInjected(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1:
            self.experts.cpu_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            y = self.experts.cpu_experts.sync_for_one_decode().unsqueeze(0)
            y += shared_expert_output
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.current_device)  if isinstance(self.experts, MLPExpertsBase) else hidden_states_expert.cpu()
        selected_experts_expert = selected_experts.to(self.experts.current_device) if isinstance(self.experts, MLPExpertsBase) else selected_experts_expert.cpu()
        routing_weights_expert = routing_weights.to(self.experts.current_device) if isinstance(self.experts, MLPExpertsBase) else routing_weights_expert.cpu()

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        if isinstance(self.experts, MLPExpertsBase):
            y = (
                self.moe_on_cpuinfer(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert, sequence_length != 1
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
        y += shared_expert_output
        y.resize_(*orig_shape)
        return y, router_logits
    
    @torch.no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, need_sync: bool) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight)
        if need_sync:
            torch.cuda.synchronize()
        #print("alive")
            #print("alive")
        return outs

    @torch.no_grad()
    # TODO
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @torch.no_grad()
    # TODO
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward_cpu(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states


class DeepseekV2MoEInjected(BaseInjectedModule, DeepseekV2MoE):
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if sequence_length == 1:
            self.experts.cpu_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.cpu_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
            
        if isinstance(self.experts, MLPExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, topk_idx, topk_weight)
        elif hidden_states.size(0) > 10:
            # TODO
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(
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
