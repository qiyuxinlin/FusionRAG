/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:34:23
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
// Python bindings
#include "cpu_backend/cpuinfer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "llamafile/flags.h"
#include "operators/kvcache/kvcache.h"
#include "operators/llamafile/linear.h"
#include "operators/llamafile/mlp.h"
#include "operators/llamafile/moe.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <cstdint>
#include <iostream>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;

// Binding functions for the KVCache class
class KVCacheBindings {
  public:
    static void bind_attn(CPUInfer &cpuinfer, KVCache *kv_cache, py::args args,
                          py::kwargs kwargs) {
        auto q_in = args[0].cast<intptr_t>();
        auto output = args[1].cast<intptr_t>();
        auto attn_lse = args[2].cast<intptr_t>();
        auto layer_idx = args[3].cast<int>();
        auto q_len = args[4].cast<int>();
        auto batch_size = args[5].cast<int>();

        int max_block_num = 0;
        if (!args[6].is_none()) {
            max_block_num = args[6].cast<int>();
        }
        intptr_t block_table = 0;

        if (!args[7].is_none()) {
            block_table = args[7].cast<intptr_t>();
        }

        intptr_t cache_seqlens = 0;
        if (!args[8].is_none()) {
            cache_seqlens = args[8].cast<intptr_t>();
        }

        int pick_block_num = -1;
        int init_block_num = -1;
        int local_block_num = -1;
        if (!args[9].is_none() && !args[10].is_none() && !args[11].is_none()) {
            pick_block_num = args[9].cast<int>();
            init_block_num = args[10].cast<int>();
            local_block_num = args[11].cast<int>();
        }

        cpuinfer.submit(&KVCache::attn, kv_cache, (const ggml_fp16_t *)q_in,
                        (ggml_fp16_t *)output, (float *)attn_lse, layer_idx,
                        q_len, batch_size, max_block_num, (int *)block_table,
                        (int *)cache_seqlens, pick_block_num, init_block_num,
                        local_block_num);
    }

    static void bind_get_all_kv_one_layer(CPUInfer &cpuinfer, KVCache *kv_cache,
                                          py::args args, py::kwargs kwargs) {
        auto k_in = args[0].cast<intptr_t>();
        auto v_in = args[1].cast<intptr_t>();
        auto layer_id = args[2].cast<int>();

        cpuinfer.submit(&KVCache::get_all_kv_one_layer, kv_cache, layer_id,
                        (ggml_fp16_t *)k_in, (ggml_fp16_t *)v_in);
    }

    static void bind_get_and_update_fp16(CPUInfer &cpuinfer, KVCache *kv_cache,
                                         py::args args, py::kwargs kwargs) {

        auto k_in = args[0].cast<intptr_t>();
        auto v_in = args[1].cast<intptr_t>();
        auto layer_id = args[2].cast<int>();
        auto block_table = args[3].cast<intptr_t>();
        auto batch_size = args[4].cast<int>();
        auto max_block_num = args[5].cast<int>();
        auto cache_seqlens = args[6].cast<intptr_t>();
        auto q_len = args[7].cast<int>();

        cpuinfer.submit(&KVCache::get_and_update_fp16, kv_cache,
                        (ggml_fp16_t *)k_in, (ggml_fp16_t *)v_in, layer_id,
                        (int *)block_table, batch_size, max_block_num,
                        (int *)cache_seqlens, q_len);
    }
    static void bind_update_importance(CPUInfer &cpuinfer, KVCache *kv_cache,
                                       py::args args, py::kwargs kwargs) {

        auto importance = args[0].cast<intptr_t>();
        auto layer_id = args[1].cast<int>();
        auto block_table = args[2].cast<intptr_t>();
        auto batch_size = args[3].cast<int>();
        auto max_block_num = args[4].cast<int>();
        auto offset = args[5].cast<intptr_t>();
        auto width = args[6].cast<int>();

        cpuinfer.submit(&KVCache::update_importance, kv_cache,
                        (const ggml_fp16_t *)importance, layer_id,
                        (int *)block_table, batch_size, max_block_num,
                        (int *)offset, width);
    }

    static void bind_attn_with_kvcache(CPUInfer &cpuinfer, KVCache *kv_cache,
                                       py::args args, py::kwargs kwargs) {
        auto q_in = args[0].cast<intptr_t>();
        auto k_in = args[1].cast<intptr_t>();
        auto v_in = args[2].cast<intptr_t>();
        auto output = args[3].cast<intptr_t>();
        auto attn_lse = args[4].cast<intptr_t>();
        auto layer_idx = args[5].cast<int>();
        auto q_len = args[6].cast<int>();
        auto batch_size = args[7].cast<int>();
        auto max_block_num = args[8].cast<int>();
        auto block_table = args[9].cast<intptr_t>();
        auto cache_seqlens = args[10].cast<intptr_t>();
        auto topk = args[11].cast<int>();
        auto local = args[12].cast<int>();

        cpuinfer.submit(&KVCache::attn_with_kvcache, kv_cache,
                        (const ggml_fp16_t *)q_in, (const ggml_fp16_t *)k_in,
                        (const ggml_fp16_t *)v_in, (ggml_fp16_t *)output,
                        (float *)attn_lse, layer_idx, q_len, batch_size,
                        max_block_num, (int *)block_table, (int *)cache_seqlens,
                        topk, local);
    }

    static void bind_clear_importance_all_layers(CPUInfer &cpuinfer,
                                                 KVCache *kv_cache,
                                                 py::args args,
                                                 py::kwargs kwargs) {
        auto block_table = args[0].cast<intptr_t>();
        auto cache_seqlens = args[1].cast<intptr_t>();
        auto batch_size = args[2].cast<int>();
        auto max_block_num = args[3].cast<int>();

        cpuinfer.submit(&KVCache::clear_importance_all_layers, kv_cache,
                        (int *)block_table, (int *)cache_seqlens, batch_size,
                        max_block_num);
    }

    static void bind_calc_anchor_all_layers(CPUInfer &cpuinfer,
                                            KVCache *kv_cache, py::args args,
                                            py::kwargs kwargs) {

        auto block_table = args[0].cast<intptr_t>();
        auto cache_seqlens = args[1].cast<intptr_t>();
        auto batch_size = args[2].cast<int>();
        auto max_block_num = args[3].cast<int>();

        cpuinfer.submit(&KVCache::calc_anchor_all_layers, kv_cache,
                        (int *)block_table, (int *)cache_seqlens, batch_size,
                        max_block_num);
    }

    // Dispatches the binding of KVCache class functions based on the
    // function name This function identifies the function to bind and calls
    // the appropriate binding function
    static void bind_functions(CPUInfer &cpuinfer, py::object func,
                               py::args args, py::kwargs kwargs) {
        auto kv_cache = func.attr("__self__").cast<KVCache *>();
        std::string func_name = py::str(func.attr("__func__").attr("__name__"));

        if (func_name == "get_all_kv_one_layer") {
            bind_get_all_kv_one_layer(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "attn") {
            bind_attn(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "get_and_update_fp16") {
            bind_get_and_update_fp16(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "update_importance") {
            bind_update_importance(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "attn_with_kvcache") {
            bind_attn_with_kvcache(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "clear_importance_all_layers") {
            bind_clear_importance_all_layers(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "calc_anchor_all_layers") {
            bind_calc_anchor_all_layers(cpuinfer, kv_cache, args, kwargs);
        } else {
            // handle other functions
            throw py::value_error("Unsupported function: " +
                                  std::string(func_name));
        }
    }
};

// Binding functions for the Linear class
class LinearBindings {
  public:
    static void bind_forward(CPUInfer &cpuinfer, Linear *linear, py::args args,
                             py::kwargs kwargs) {
        auto input = args[0].cast<intptr_t>();
        auto output = args[1].cast<intptr_t>();
        cpuinfer.submit(&Linear::forward, linear, (const void *)input,
                        (void *)output);
    }

    static void bind_warm_up(CPUInfer &cpuinfer, Linear *linear, py::args args,
                             py::kwargs kwargs) {
        cpuinfer.submit(&Linear::warm_up, linear);
    }

    static void bind_functions(CPUInfer &cpuinfer, py::object func,
                               py::args args, py::kwargs kwargs) {
        auto linear = func.attr("__self__").cast<Linear *>();
        std::string func_name = py::str(func.attr("__func__").attr("__name__"));

        if (func_name == "forward") {
            bind_forward(cpuinfer, linear, args, kwargs);
        } else if (func_name == "warm_up") {
            bind_warm_up(cpuinfer, linear, args, kwargs);
        } else {
            throw py::value_error("Unsupported function: " +
                                  std::string(func_name));
        }
    }
};

// Binding functions for the MLP class
class MLPBindings {
  public:
    static void bind_forward(CPUInfer &cpuinfer, MLP *mlp, py::args args,
                             py::kwargs kwargs) {
        auto input = args[0].cast<intptr_t>();
        auto output = args[1].cast<intptr_t>();
        cpuinfer.submit(&MLP::forward, mlp, (const void *)input,
                        (void *)output);
    }

    static void bind_warm_up(CPUInfer &cpuinfer, MLP *mlp, py::args args,
                             py::kwargs kwargs) {
        cpuinfer.submit(&MLP::warm_up, mlp);
    }

    static void bind_functions(CPUInfer &cpuinfer, py::object func,
                               py::args args, py::kwargs kwargs) {
        auto mlp = func.attr("__self__").cast<MLP *>();
        std::string func_name = py::str(func.attr("__func__").attr("__name__"));

        if (func_name == "forward") {
            bind_forward(cpuinfer, mlp, args, kwargs);
        } else if (func_name == "warm_up") {
            bind_warm_up(cpuinfer, mlp, args, kwargs);
        } else {
            throw py::value_error("Unsupported function: " +
                                  std::string(func_name));
        }
    }
};

// Binding functions for the MOE class
class MOEBindings {
  public:
    static void bind_forward(CPUInfer &cpuinfer, MOE *moe, py::args args,
                             py::kwargs kwargs) {
        int qlen = args[0].cast<int>();
        int k = args[1].cast<int>();
        auto expert_ids = args[2].cast<intptr_t>();
        auto weights = args[3].cast<intptr_t>();
        auto input = args[4].cast<intptr_t>();
        auto output = args[5].cast<intptr_t>();
        cpuinfer.submit(&MOE::forward, moe, qlen, k,
                        (const uint64_t *)expert_ids, (const float *)weights,
                        (const void *)input, (void *)output);
    }

    static void bind_warm_up(CPUInfer &cpuinfer, MOE *moe, py::args args,
                             py::kwargs kwargs) {
        cpuinfer.submit(&MOE::warm_up, moe);
    }

    static void bind_functions(CPUInfer &cpuinfer, py::object func,
                               py::args args, py::kwargs kwargs) {
        auto moe = func.attr("__self__").cast<MOE *>();
        std::string func_name = py::str(func.attr("__func__").attr("__name__"));

        if (func_name == "forward") {
            bind_forward(cpuinfer, moe, args, kwargs);
        } else if (func_name == "warm_up") {
            bind_warm_up(cpuinfer, moe, args, kwargs);
        } else {
            throw py::value_error("Unsupported function: " +
                                  std::string(func_name));
        }
    }
};

struct MOEForwardArgs {
    CPUInfer *cpuinfer;
    MOE *moe;
    int qlen;
    int k;
    uint64_t *expert_ids;
    float *weights;
    void *input;
    void *output;
};

void submit_moe_forward_with_host_args_ptr(void *host_args_ptr) {
    MOEForwardArgs *host_args = (MOEForwardArgs *)host_args_ptr;
    host_args->cpuinfer->submit(&MOE::forward, host_args->moe, host_args->qlen,
                                host_args->k, host_args->expert_ids,
                                host_args->weights, host_args->input,
                                host_args->output);
}

void cpuinfer_sync(void *host_args_ptr) {
    CPUInfer *cpuinfer = (CPUInfer *)host_args_ptr;
    cpuinfer->sync();
}

PYBIND11_MODULE(cpuinfer_ext, m) {
    auto kvcache_module = m.def_submodule("kvcache");

    py::enum_<AnchorType>(kvcache_module, "AnchorType")
        .value("FIXED", AnchorType::FIXED)
        .value("DYNAMIC", AnchorType::DYNAMIC)
        .value("QUEST", AnchorType::QUEST);
    py::enum_<ggml_type>(kvcache_module, "ggml_type")
        .value("FP16", ggml_type::GGML_TYPE_F16)
        .value("FP32", ggml_type::GGML_TYPE_F32)
        .value("Q4_0", ggml_type::GGML_TYPE_Q4_0)
        .value("Q8_0", ggml_type::GGML_TYPE_Q8_0);

    py::class_<KVCacheConfig>(kvcache_module, "KVCacheConfig")
        .def(py::init<int, int, int, int, int, int, AnchorType, ggml_type, int,
                      int, int>())
        .def_readwrite("layer_num", &KVCacheConfig::layer_num)
        .def_readwrite("kv_head_num", &KVCacheConfig::kv_head_num)
        .def_readwrite("q_head_num", &KVCacheConfig::q_head_num)
        .def_readwrite("head_dim", &KVCacheConfig::head_dim)
        .def_readwrite("block_len", &KVCacheConfig::block_len)
        .def_readwrite("anchor_num", &KVCacheConfig::anchor_num)
        .def_readwrite("anchor_type", &KVCacheConfig::anchor_type)
        .def_readwrite("kv_type", &KVCacheConfig::kv_type)
        .def_readwrite("max_block_num", &KVCacheConfig::max_block_num)
        .def_readwrite("max_batch_size", &KVCacheConfig::max_batch_size)
        .def_readwrite("max_thread_num", &KVCacheConfig::max_thread_num);

    py::class_<KVCache>(kvcache_module, "KVCache")
        .def(py::init<KVCacheConfig>())
        .def("get_cache_total_len", &KVCache::get_cache_total_len)
        .def("update_cache_total_len",
             [](KVCache &kvcache, int cache_total_len) {
                 kvcache.update_cache_total_len(cache_total_len);
             })
        .def("attn",
             [](KVCache &kvcache, intptr_t q_in, intptr_t output,
                intptr_t attn_lse, int layer_idx, int q_len, int batch_size,
                int max_block_num, intptr_t block_table, intptr_t cache_seqlens,
                int pick_block_num, int init_block_num, int local_block_num,
                Backend *backend) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("get_all_kv_one_layer",
             [](KVCache &kvcache, int layer_id, intptr_t k_in, intptr_t v_in,
                Backend *backend) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("get_and_update_fp16",
             [](KVCache &kvcache, intptr_t k_in, intptr_t v_in, int layer_id,
                intptr_t block_table, int batch_size, int max_block_num,
                intptr_t cache_seqlens, int q_len, Backend *backend) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("update_importance",
             [](KVCache &kvcache, intptr_t importance, int layer_id,
                intptr_t block_table, int batch_size, int max_block_num,
                intptr_t offset, int width, Backend *backend) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("attn_with_kvcache",
             [](KVCache &kvcache, intptr_t q_in, intptr_t k_in, intptr_t v_in,
                intptr_t output, intptr_t attn_lse, int layer_idx, int q_len,
                int batch_size, int max_block_num, intptr_t block_table,
                intptr_t cache_seqlens, int topk, int local, Backend *backend) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("clear_importance_all_layers",
             [](KVCache &kvcache, intptr_t block_table, intptr_t cache_seqlens,
                int batch_size, int max_block_num, Backend *backend) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("calc_anchor_all_layers",
             [](KVCache &kvcache, intptr_t block_table, intptr_t cache_seqlens,
                int batch_size, int max_block_num, Backend *backend) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             });

    auto linear_module = m.def_submodule("linear");

    py::class_<LinearConfig>(linear_module, "LinearConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride,
                         intptr_t proj, int proj_type, int hidden_type) {
            return LinearConfig(hidden_size, intermediate_size, stride,
                                (void *)proj, (ggml_type)proj_type,
                                (ggml_type)hidden_type);
        }));

    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())
        .def("warm_up",
             [](Linear &linear) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("forward", [](Linear &linear, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use "
                                     "CPUInfer.submit to call it!!!\n");
        });

    auto mlp_module = m.def_submodule("mlp");

    py::class_<MLPConfig>(mlp_module, "MLPConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride,
                         intptr_t gate_proj, intptr_t up_proj,
                         intptr_t down_proj, int gate_type, int up_type,
                         int down_type, int hidden_type) {
            return MLPConfig(hidden_size, intermediate_size, stride,
                             (void *)gate_proj, (void *)up_proj,
                             (void *)down_proj, (ggml_type)gate_type,
                             (ggml_type)up_type, (ggml_type)down_type,
                             (ggml_type)hidden_type);
        }));

    py::class_<MLP>(mlp_module, "MLP")
        .def(py::init<MLPConfig>())
        .def("warm_up",
             [](MLP &mlp) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("forward", [](MLP &mlp, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use "
                                     "CPUInfer.submit to call it!!!\n");
        });

    auto moe_module = m.def_submodule("moe");

    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size,
                         int intermediate_size, int stride, int group_min_len,
                         int group_max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj, int gate_type,
                         int up_type, int down_type, int hidden_type) {
            return MOEConfig(expert_num, routed_expert_num, hidden_size,
                             intermediate_size, stride, group_min_len,
                             group_max_len, (void *)gate_proj, (void *)up_proj,
                             (void *)down_proj, (ggml_type)gate_type,
                             (ggml_type)up_type, (ggml_type)down_type,
                             (ggml_type)hidden_type);
        }));

    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up",
             [](MOE &moe) {
                 throw std::runtime_error("!!! Doing nothing, please use "
                                          "CPUInfer.submit to call it!!!\n");
             })
        .def("forward", [](MOE &moe, int k, uint64_t expert_ids,
                           intptr_t weights, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use "
                                     "CPUInfer.submit to call it!!!\n");
        });

    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit",
             [linear_module, mlp_module,
              moe_module](CPUInfer &cpuinfer, py::object func, py::args args,
                          py::kwargs kwargs) {
                 if (py::hasattr(func, "__self__") &&
                     py::hasattr(func, "__func__")) {
                     std::string class_name = py::str(func.attr("__self__")
                                                          .attr("__class__")
                                                          .attr("__name__"));
                     if (class_name == "Linear") {
                         LinearBindings::bind_functions(cpuinfer, func, args,
                                                        kwargs);
                     } else if (class_name == "MLP") {
                         MLPBindings::bind_functions(cpuinfer, func, args,
                                                     kwargs);
                     } else if (class_name == "MOE") {
                         MOEBindings::bind_functions(cpuinfer, func, args,
                                                     kwargs);
                     } else if (class_name == "KVCache") {
                         KVCacheBindings::bind_functions(cpuinfer, func, args,
                                                         kwargs);
                     } else {
                         // handle other classes
                         throw py::type_error("Unsupported class type: " +
                                              class_name);
                     }
                 } else {
                     // handle cases where func does not have __self__ or
                     // __func__
                     throw py::type_error("Invalid function object: missing "
                                          "__self__ or __func__ attribute.");
                 }
             })
        .def("submit_with_cuda_stream",
             [linear_module, mlp_module,
              moe_module](CPUInfer &cpuinfer, intptr_t user_cuda_stream,
                          py::object func, py::args args, py::kwargs kwargs) {
                 if (py::hasattr(func, "__self__") &&
                     py::hasattr(func, "__func__")) {
                     std::string class_name = py::str(func.attr("__self__")
                                                          .attr("__class__")
                                                          .attr("__name__"));
                     if (class_name == "MOE") {
                         std::string func_name =
                             py::str(func.attr("__func__").attr("__name__"));
                         if (func_name == "forward") {
                             auto moe = func.attr("__self__").cast<MOE *>();
                             int qlen = args[0].cast<int>();
                             int k = args[1].cast<int>();
                             auto expert_ids = args[2].cast<intptr_t>();
                             auto weights = args[3].cast<intptr_t>();
                             auto input = args[4].cast<intptr_t>();
                             auto output = args[5].cast<intptr_t>();
                             MOEForwardArgs *moe_forward_args =
                                 new MOEForwardArgs{&cpuinfer,
                                                    moe,
                                                    qlen,
                                                    k,
                                                    (uint64_t *)expert_ids,
                                                    (float *)weights,
                                                    (void *)input,
                                                    (void *)output};
                             // submit_moe_forward_with_host_args_ptr(moe_forward_args);
                             cudaLaunchHostFunc(
                                 (cudaStream_t)user_cuda_stream,
                                 (cudaHostFn_t)
                                     submit_moe_forward_with_host_args_ptr,
                                 moe_forward_args);
                         } else {
                             throw py::value_error("Unsupported function: " +
                                                   std::string(func_name));
                         }
                     } else {
                         // handle other classes
                         throw py::type_error("Unsupported class type: " +
                                              class_name);
                     }
                 } else {
                     // handle cases where func does not have __self__ or
                     // __func__
                     throw py::type_error("Invalid function object: missing "
                                          "__self__ or __func__ attribute.");
                 }
             })
        .def("sync_with_cuda_stream",
             [](CPUInfer &cpuinfer, intptr_t user_cuda_stream) {
                 // cpuinfer_sync((void*)(&cpuinfer));
                 cudaLaunchHostFunc((cudaStream_t)user_cuda_stream,
                                    (cudaHostFn_t)cpuinfer_sync,
                                    (void *)(&cpuinfer));
             })
        .def("sync", &CPUInfer::sync);
}
