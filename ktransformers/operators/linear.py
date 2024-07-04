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
from util.custom_gguf import GGUFLoader
from ktransformers_ext.custom_marlin.quantize.utils.marlin_perms import marlin_perm
from ktransformers_ext.custom_marlin.quantize.utils.marlin_utils import (
    MarlinWorkspace,
    compute_max_diff,
    is_marlin_supported,
    marlin_24_quantize,
    marlin_quantize,
    marlin_weights,
)
from ktransformers_ext.custom_marlin.quantize.utils.quant_utils import (
    gptq_pack,
    quantize_weights,
    sort_weights,
)
from operators.base_operator import BaseInjectedModule
from transformers.configuration_utils import PretrainedConfig
from utils import _set_param

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

GPTQ_MARLIN_SUPPORTED_NUM_BITS = [4, 8]
GPTQ_MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
GPTQ_MARLIN_SUPPORTED_SYM = [True]


def relative_l2_error(standard: torch.Tensor, x: torch.Tensor) -> float:
    sl2 = linalg.vector_norm(standard).item()
    el2 = linalg.vector_norm(standard - x).item()
    return el2 / sl2


class QuantizedLinear(BaseInjectedModule):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(dtype=self.dtype)
        x = self.linear(x)
        x = x.to(dtype)
        return x

    def load(self):
        if w is None:
            w = self.load_weight()

        if isinstance(w, nn.Parameter):
            self.linear = nn.Linear(
                self.in_features, self.out_features, bias=False, dtype=self.dtype
            )
            self.linear.weight.data = w.to(dtype=self.dtype)
        elif isinstance(w, tuple):
            self.linear = nn.Linear(
                self.in_features, self.out_features, bias=True, dtype=self.dtype
            )
            self.linear.weight.data = w[0].to(dtype=self.dtype)
            self.linear.bias.data = w[1].to(dtype=self.dtype)
        else:
            raise ValueError("Invalid weight type")
        self.linear = self.linear.to(self.device)

    def load_weight(self, override_key: str | None = None):
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        for key in keys:
            if key + ".weight" in self.gguf_loader.tensor_file_map:
                if key + ".bias" in self.gguf_loader.tensor_file_map:
                    tensors = self.load_multi(key, ["weight", "bias"])
                    tensor = torch.tensor(tensors["weight"], dtype=torch.float32)
                    bias = torch.tensor(tensors["bias"], dtype=torch.float32)
                    # self.qtype = GGML_TYPE_QTYPE_MAP[tensorinfo[key + ".weight"]["ggml_type"]]
                    self.in_features = self.gguf_loader.tensor_info[key + ".weight"][
                        "shape"
                    ][0]
                    self.out_features = self.gguf_loader.tensor_info[key + ".weight"][
                        "shape"
                    ][1]
                    self.has_bias = True
                    # print(torch.isinf(tensor).any(), torch.isinf(bias).any())
                    return nn.Parameter(tensor), nn.Parameter(bias)
                else:
                    tensors = self.load_multi(key, ["weight"])
                    tensor = torch.tensor(tensors["weight"], dtype=torch.float32)
                    # self.qtype = GGML_TYPE_QTYPE_MAP[tensorinfo[key + ".weight"]["ggml_type"]]
                    self.in_features = self.gguf_loader.tensor_info[key + ".weight"][
                        "shape"
                    ][0]
                    self.out_features = self.gguf_loader.tensor_info[key + ".weight"][
                        "shape"
                    ][1]
                    self.has_bias = False
                    return nn.Parameter(tensor)
            else:
                raise FileNotFoundError(f"Weight file not found for key {key}")

    def load_multi(self, key: str, keys: list[str]):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + "." + k)
        return tensors

    def unload(self):
        self.linear = None


class QuantizedLinearMarlin(QuantizedLinear):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module,
        device: str = "cuda",
        num_bits: int = 4,  # 4-bit/8-bit is supported
        group_size: int = 64,  # -1, 32, 64, 128
        act_order: bool = False,
        is_k_full=True,
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.num_bits = num_bits
        self.group_size = group_size
        self.act_order = act_order
        self.is_k_full = is_k_full
        size_n = self.out_features
        self.workspace = MarlinWorkspace(
            size_n, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL
        )

    def load(self):
        print("loading QuantizedLinearMarlin", self.key)
        w = self.load_weight()

        if isinstance(w, nn.Parameter):
            # pad weight
            weight = w.T.half()
        elif isinstance(w, tuple):
            w = list(w)
            weight = w[0].T.half()
            _set_param(self, "bias", w[1].to(self.device))
        else:
            raise ValueError("Invalid weight type")

        # Pack Marlin linear
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            weight, self.num_bits, self.group_size, self.act_order
        )
        _set_param(self, "marlin_q_w", marlin_q_w.to(self.device))
        _set_param(self, "marlin_s", marlin_s.to(self.device))
        _set_param(self, "g_idx", g_idx.to(self.device))
        _set_param(self, "sort_indices", sort_indices.to(self.device))
        self.k = weight.shape[0]
        self.n = weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

