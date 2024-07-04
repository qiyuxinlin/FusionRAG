// Copyright 2024 Shaoyuan Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// #include "mlp.hpp"
// #include "linear.hpp"
// #include "quantize.hpp"
#include "gptq_marlin/ops.h"
// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
// namespace py = pybind11;
// using namespace qlib;


PYBIND11_MODULE(qlib, m) {
    m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "Function to perform GEMM using Marlin quantization.",
          py::arg("a"), py::arg("b_q_weight"), py::arg("b_scales"), py::arg("g_idx"),
          py::arg("perm"), py::arg("workspace"), py::arg("num_bits"), py::arg("size_m"),
          py::arg("size_n"), py::arg("size_k"), py::arg("is_k_full"));
    /*m.def("add", &add, "Add two tensors.");
    m.def("add", &add_torch, "Add two tensors.");*/
}
