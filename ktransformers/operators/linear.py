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

from dataclasses import dataclass
import torch
from torch import nn
from torch import linalg
import qlib
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.ktransformers_ext.custom_marlin.quantize.utils.marlin_perms import marlin_perm
from ktransformers.ktransformers_ext.custom_marlin.quantize.utils.marlin_utils import (
    MarlinWorkspace,
    compute_max_diff,
    is_marlin_supported,
    marlin_24_quantize,
    marlin_quantize,
    marlin_weights,
    GPTQ_MARLIN_TILE,
    GPTQ_MARLIN_MIN_THREAD_N,
    GPTQ_MARLIN_MIN_THREAD_K,
    GPTQ_MARLIN_MAX_PARALLEL,
    GPTQ_MARLIN_SUPPORTED_NUM_BITS,
    GPTQ_MARLIN_SUPPORTED_GROUP_SIZES,
    GPTQ_MARLIN_SUPPORTED_SYM,
)
from ktransformers.ktransformers_ext.custom_marlin.quantize.utils.quant_utils import (
    gptq_pack,
    quantize_weights,
    sort_weights,
)
from ktransformers.operators.base_operator import BaseInjectedModule
from transformers.configuration_utils import PretrainedConfig
from ktransformers.util.utils import _set_param
from abc import ABC, abstractmethod
import time



#class QuantizedLinearBase(BaseInjectedModule, ABC):
class QuantizedLinearBase(ABC):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs,
    ):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        super().__init__()
        self.key = key
        self.gguf_loader = gguf_loader
        self.device = device
        self.config = config

        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.in_features = self.gguf_loader.tensor_info[key + ".weight"]["shape"][0]
        self.out_features = self.gguf_loader.tensor_info[key + ".weight"]["shape"][1]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def load_weight(self, override_key: str | None = None, device: str | None = None):
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        for key in keys:
            if key + ".weight" in self.gguf_loader.tensor_file_map:
                if key + ".bias" in self.gguf_loader.tensor_file_map:
                    tensors = self.load_multi(key, ["weight", "bias"], device=device)
                    tensor = torch.tensor(tensors["weight"], dtype=torch.float32)
                    bias = torch.tensor(tensors["bias"], dtype=torch.float32)
                    # self.qtype = GGML_TYPE_QTYPE_MAP[tensorinfo[key + ".weight"]["ggml_type"]]
                    # print(torch.isinf(tensor).any(), torch.isinf(bias).any())
                    return nn.Parameter(tensor), nn.Parameter(bias)
                else:
                    tensors = self.load_multi(key, ["weight"], device=device)
                    tensor = torch.tensor(tensors["weight"], dtype=torch.float32)
                    # self.qtype = GGML_TYPE_QTYPE_MAP[tensorinfo[key + ".weight"]["ggml_type"]]
                    return nn.Parameter(tensor)
            else:
                raise FileNotFoundError(f"Weight file not found for key {key}")

    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        is_gpu = True if device.lower() != "cpu" else False
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + "." + k, is_gpu=is_gpu)
        return tensors

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = "cuda"):
        pass

    @abstractmethod
    def unload(self):
        pass


class QuantizedLinearTorch(QuantizedLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out_device = x.device
        x = x.to(device=self.device)
        x = x.to(dtype=self.dtype)
        x = x @ self.w
        x = x.to(dtype).to(out_device)
        return x

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if device is None: device = self.device
        if w is None: w = self.load_weight(device=device)

        if isinstance(w, nn.Parameter):
            self.w = w.to(dtype=self.dtype).T
            self.has_bias = False
        elif isinstance(w, tuple):
            self.w = w[0].to(dtype=self.dtype).T
            self.bias = w[1].to(dtype=self.dtype)
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        # self.linear = self.linear.to(device)
        self.w = self.w.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)

    def unload(self):
        if self.w is not None:
            self.w = None
        if self.has_bias is not None:
            self.bias = None


class QuantizedLinearMarlin(QuantizedLinearBase):
    marlin_q_w: torch.Tensor
    marlin_s: torch.Tensor
    g_idx: torch.Tensor
    sort_indices: torch.Tensor
    has_bias: bool
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cuda",
        num_bits: int = 4,  # 4-bit/8-bit is supported
        group_size: int = 64,  # -1, 32, 64, 128
        act_order: bool = False,
        is_k_full=True,
        **kwargs,
    ):
        assert device.lower() != "cpu", "Marlin quantized linear only supports GPU device"
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.num_bits = num_bits
        self.group_size = group_size
        self.act_order = act_order
        self.is_k_full = is_k_full

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = "cuda"):
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin quantized linear only supports GPU device"
        if w is None: w = self.load_weight(device=device)

        if isinstance(w, nn.Parameter):
            # pad weight
            weight = w.T
            self.has_bias = False
        elif isinstance(w, tuple):
            w = list(w)
            weight = w[0].T
            _set_param(self, "bias", w[1])
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        weight = weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
        # Pack Marlin linear
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            weight, self.num_bits, self.group_size, self.act_order
        )
        self.workspace = MarlinWorkspace(
            self.out_features, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL
        )
        self.marlin_q_w = marlin_q_w
        self.marlin_s = marlin_s
        self.g_idx = g_idx
        self.sort_indices = sort_indices
        self.k = weight.shape[0]
        self.n = weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only support input x as BF16 and FP16
        x = x.to(self.device)
        orig_shape = list(x.shape)
        orig_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        marlin_s = self.marlin_s.to(x.dtype)
        x = qlib.gptq_marlin_gemm(
            x,
            self.marlin_q_w,
            marlin_s,
            self.g_idx,
            self.sort_indices,
            self.workspace.scratch,
            self.num_bits,
            x.shape[0],
            self.n,
            x.shape[-1],
            self.is_k_full,
        )
        if self.has_bias:
            x = x + self.bias
        orig_shape[-1] = self.n
        return x.reshape(orig_shape).to(orig_dtype)

    def unload(self):

        if self.has_bias:
            self.bias = None
        self.marlin_q_w = None
        self.marlin_s = None
        self.g_idx = None
        self.sort_indices = None
        self.workspace = None
    

CPU_LINEAR_MAP = {
    "QuantizedLinearTorch": QuantizedLinearTorch,
}
GPU_LINEAR_MAP = {
    "QuantizedLinearMarlin": QuantizedLinearMarlin,
    "QuantizedLinearTorch": QuantizedLinearTorch,
}


class KTransformerLinear(BaseInjectedModule, QuantizedLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        device: str = "cuda",
        gpu_linear_type: str| None = "QuantizedLinearMarlin",
        cpu_linear_type: str| None = "QuantizedLinearTorch",
        **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        QuantizedLinearBase.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        # build all the linear operators
        if cpu_linear_type is not None:
            assert cpu_linear_type in CPU_LINEAR_MAP, f"cpu_linear_type {cpu_linear_type} not supported"
            self.cpu_linear = CPU_LINEAR_MAP[cpu_linear_type](key, gguf_loader, config, orig_module, "cpu", **kwargs)
        else:
            self.cpu_linear = None
        if gpu_linear_type is not None:
            assert gpu_linear_type in GPU_LINEAR_MAP, f"gpu_linear_type {gpu_linear_type} not supported"
            self.gpu_linear = GPU_LINEAR_MAP[gpu_linear_type](key, gguf_loader, config, orig_module, "cuda", **kwargs)
        else:
            self.gpu_linear = None
        self.gpu_linear_type = gpu_linear_type
        self.cpu_linear_type = cpu_linear_type
        self.current_device = device

    def forward(self, x):
        if self.current_device == "cpu":
            assert self.cpu_linear is not None, "cpu linear is not initialized"
            return self.cpu_linear.forward(x)
        else:
            assert self.gpu_linear is not None, "gpu linear is not initialized"
            return self.gpu_linear.forward(x)

    def load(self, w: dict | nn.Parameter | tuple | None = None, device:str|None = None):
        if w is None: self.w = self.load_weight(device="cpu")
        else: self.w = w
        if device is None: device = self.device
        # load to device
        if self.device == "cpu":
            # print(f'loading {self.key} to {self.device} from class {self.cpu_linear_type}')
            self.cpu_linear.load(self.w, device=device)
        elif "cuda" in self.device.lower():
            # print(f'loading {self.key} to {self.device} from class {self.gpu_linear_type}')
            self.gpu_linear.load(self.w, device=device)
        self.current_device = device

    def unload(self):
        if self.cpu_linear is not None:
            self.cpu_linear.unload()
        if self.gpu_linear is not None:
            self.gpu_linear.unload()
        self.w = None

    def load_to(self, target):
        # print(f"loading {self.key} to {target}")
        if isinstance(target, str) and target == "cpu":
            self.cpu_linear.load(w=self.w, device="cpu")
            self.gpu_linear.unload()
            self.current_device = target
        elif isinstance(target, str) and "cuda" in target:
            self.gpu_linear.load(w=self.w, device=target)
            self.cpu_linear.unload()
            self.current_device = target
        elif isinstance(target, str) and target == "restore":
            assert self.device != "restore", "device is already restored"
            self.load_to(self.device)
        else:
            raise ValueError("target must be either \"cpu\", \"cuda\", \"cuda:idx\" or \"restore\"")