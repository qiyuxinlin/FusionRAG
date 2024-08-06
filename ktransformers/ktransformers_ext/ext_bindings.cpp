/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-08-06 11:00:04
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
// Python bindings
#include <cstdint>
#include <iostream>
#include <memory>
#include "cpu_backend/cpuinfer.h"
#include "device_launch_parameters.h"
#include "llamafile/flags.h"
#include "operators/llamafile/linear.h"
#include "operators/llamafile/mlp.h"
#include "operators/llamafile/moe.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cpuinfer_ext, m) {
    auto linear_module = m.def_submodule("linear");

    py::class_<LinearConfig>(linear_module, "LinearConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride, int group_max_len, intptr_t proj, int proj_type, int hidden_type) {
            return LinearConfig(hidden_size, intermediate_size, stride, group_max_len, (void*)proj, (ggml_type)proj_type, (ggml_type)hidden_type);
        }));

    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())
        .def("warm_up", [](Linear& linear) {
            std::function<void(void*)> func = [&linear](void* cpu_infer_ptr) {
                CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
                cpuinfer->enqueue(&Linear::warm_up, &linear);
            };
            return func;
        })
        .def("forward", [](Linear& linear, int qlen, intptr_t input, intptr_t output) {
            std::function<void(void*)> func = [=, &linear](void* cpu_infer_ptr) {
                CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
                cpuinfer->enqueue(&Linear::forward, &linear, qlen, (const void*)input, (void*)output);
            };
            return func;
        });

    auto mlp_module = m.def_submodule("mlp");

    py::class_<MLPConfig>(mlp_module, "MLPConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride, int group_max_len, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type, int hidden_type) {
            return MLPConfig(hidden_size, intermediate_size, stride, group_max_len, (void*)gate_proj, (void*)up_proj, (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type);
        }));

    py::class_<MLP>(mlp_module, "MLP")
        .def(py::init<MLPConfig>())
        .def("warm_up", [](MLP& mlp) {
            std::function<void(void*)> func = [&mlp](void* cpu_infer_ptr) {
                CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
                cpuinfer->enqueue(&MLP::warm_up, &mlp);
            };
            return func;
        })
        .def("forward", [](MLP& mlp, int qlen, intptr_t input, intptr_t output) {
            std::function<void(void*)> func = [=, &mlp](void* cpu_infer_ptr) {
                CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
                cpuinfer->enqueue(&MLP::forward, &mlp, qlen, (const void*)input, (void*)output);
            };
            return func;
        });

    auto moe_module = m.def_submodule("moe");

    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type, int hidden_type) {
            return MOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size, stride, group_min_len, group_max_len, (void*)gate_proj, (void*)up_proj, (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type);
        }));

    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up", [](MOE& moe) {
            std::function<void(void*)> func = [&moe](void* cpu_infer_ptr) {
                CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
                cpuinfer->enqueue(&MOE::warm_up, &moe);
            };
            return func;
        })
        .def("forward", [](MOE& moe, int qlen, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output) {
            std::function<void(void*)> func = [=, &moe](void* cpu_infer_ptr) {
                CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
                cpuinfer->enqueue(&MOE::forward, &moe, qlen, k, (const uint64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output);
            };
            return func;
        });

    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit", &CPUInfer::submit)
        .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
        .def("sync", &CPUInfer::sync)
        .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream);
}
