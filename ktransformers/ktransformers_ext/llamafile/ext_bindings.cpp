// Python bindings
#include <iostream>
#include <memory>
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "cpuinfer.h"
#include "llamafile/flags.h"
#include "operator/kvcache.h"
#include "operator/linear.h"
#include "operator/mlp.h"
#include "operator/moe.h"

namespace py = pybind11;
using namespace pybind11::literals;

// binding the functions of the KVCache class.
// converts the parameters from Python types to C++ types
// and submits the task to the cpuinfer
class KVCacheBindings {
   public:
    static void bind_attn(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
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

        cpuinfer.submit(&KVCache::attn, kv_cache, (const ggml_fp16_t*)q_in, (ggml_fp16_t*)output,
                        (float*)attn_lse, layer_idx, q_len, batch_size,
                        max_block_num, (int*)block_table,
                        (int*)cache_seqlens, pick_block_num, init_block_num,
                        local_block_num);
    }

    static void bind_update_one_block_fp16(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto k_in = args[0].cast<intptr_t>();
        auto v_in = args[1].cast<intptr_t>();
        auto layer_id = args[2].cast<int>();
        auto block_idx = args[3].cast<int>();

        cpuinfer.submit(&KVCache::update_one_block_fp16, kv_cache,
                        (const ggml_fp16_t*)k_in, (const ggml_fp16_t*)v_in, layer_id,
                        block_idx);
    }

    static void bind_get_one_block_fp16(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto k_in = args[0].cast<intptr_t>();
        auto v_in = args[1].cast<intptr_t>();
        auto layer_id = args[2].cast<int>();
        auto block_idx = args[3].cast<int>();

        cpuinfer.submit(&KVCache::get_one_block_fp16, kv_cache,
                        (ggml_fp16_t*)k_in, (ggml_fp16_t*)v_in, layer_id,
                        block_idx);
    }

    static void bind_update_importance_one_block(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto importance = args[0].cast<intptr_t>();
        auto layer_id = args[1].cast<int>();
        auto block_idx = args[2].cast<int>();

        assert(block_idx >= 0 && block_idx < kv_cache->get_block_num(layer_id));

        cpuinfer.submit(&KVCache::update_importance_one_block, kv_cache,
                        (const ggml_fp16_t*)importance, layer_id, block_idx);
    }

    static void bind_get_importance_one_block(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto importance = args[0].cast<intptr_t>();
        auto layer_id = args[1].cast<int>();
        auto block_idx = args[2].cast<int>();

        assert(block_idx >= 0 && block_idx < kv_cache->get_block_num(layer_id));

        cpuinfer.submit(&KVCache::get_importance_one_block, kv_cache,
                        (ggml_fp16_t*)importance, layer_id, block_idx);
    }

    static void bind_calc_anchor_all_layers(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto block_table = args[0].cast<intptr_t>();
        auto cache_seqlens = args[1].cast<intptr_t>();
        auto batch_size = args[2].cast<int>();
        auto max_block_num = args[3].cast<int>();

        cpuinfer.submit(&KVCache::calc_anchor_all_layers, kv_cache,
                        (int*)block_table, (int*)cache_seqlens, batch_size,
                        max_block_num);
    }

    static void bind_load_kvcache(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto tensor_file_path = args[0].cast<std::string>();

        cpuinfer.submit(&KVCache::load_kvcache, kv_cache, tensor_file_path);
    }

    static void bind_dump_kvcache(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto block_table = args[0].cast<intptr_t>();
        auto cache_total_len = args[1].cast<int>();
        auto tensor_file_path = args[2].cast<std::string>();

        cpuinfer.submit(&KVCache::dump_kvcache, kv_cache,
                        (int*)block_table, cache_total_len, tensor_file_path);
    }

    static void bind_update_anchor_one_block(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto anchor = args[0].cast<intptr_t>();
        auto layer_id = args[1].cast<int>();
        auto block_idx = args[2].cast<int>();

        assert(block_idx >= 0 && block_idx < kv_cache->get_block_num(layer_id));

        cpuinfer.submit(&KVCache::update_anchor_one_block, kv_cache,
                        (const ggml_fp16_t*)anchor, layer_id, block_idx);
    }

    static void bind_get_all_kv_one_layer(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto k_in = args[0].cast<intptr_t>();
        auto v_in = args[1].cast<intptr_t>();
        auto layer_id = args[2].cast<int>();

        cpuinfer.submit(&KVCache::get_all_kv_one_layer, kv_cache,
                        layer_id, (ggml_fp16_t*)k_in, (ggml_fp16_t*)v_in);
    }

    static void bind_get_anchor_one_block(CPUInfer& cpuinfer, KVCache* kv_cache, py::args args, py::kwargs kwargs) {
        auto anchor = args[0].cast<intptr_t>();
        auto layer_id = args[1].cast<int>();
        auto block_idx = args[2].cast<int>();

        assert(block_idx >= 0 && block_idx < kv_cache->get_block_num(layer_id));

        cpuinfer.submit(&KVCache::get_anchor_one_block, kv_cache,
                        (ggml_fp16_t*)anchor, layer_id, block_idx);
    }

    // Dispatches the binding of KVCache class functions based on the
    // function name This function identifies the function to bind and calls
    // the appropriate binding function
    static void bind_functions(CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
        auto kv_cache = func.attr("__self__").cast<KVCache*>();
        std::string func_name = py::str(func.attr("__func__").attr("__name__"));

        if (func_name == "attn") {
            bind_attn(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "update_one_block_fp16") {
            bind_update_one_block_fp16(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "get_one_block_fp16") {
            bind_get_one_block_fp16(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "update_importance_one_block") {
            bind_update_importance_one_block(cpuinfer, kv_cache, args,
                                             kwargs);
        } else if (func_name == "calc_anchor_all_layers") {
            bind_calc_anchor_all_layers(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "load_kvcache") {
            bind_load_kvcache(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "dump_kvcache") {
            bind_dump_kvcache(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "get_importance_one_block") {
            bind_get_importance_one_block(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "update_anchor_one_block") {
            bind_update_anchor_one_block(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "get_all_kv_one_layer") {
            bind_get_all_kv_one_layer(cpuinfer, kv_cache, args, kwargs);
        } else if (func_name == "get_anchor_one_block") {
            bind_get_anchor_one_block(cpuinfer, kv_cache, args, kwargs);
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
    static void bind_forward(CPUInfer& cpuinfer, Linear* linear, py::args args, py::kwargs kwargs) {
        auto input = args[0].cast<intptr_t>();
        auto output = args[1].cast<intptr_t>();
        cpuinfer.submit(&Linear::forward, linear,
                        (const void*)input, (void*)output);
    }

    static void bind_warm_up(CPUInfer& cpuinfer, Linear* linear, py::args args, py::kwargs kwargs) {
        cpuinfer.submit(&Linear::warm_up, linear);
    }

    static void bind_functions(CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
        auto linear = func.attr("__self__").cast<Linear*>();
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
    static void bind_forward(CPUInfer& cpuinfer, MLP* mlp, py::args args, py::kwargs kwargs) {
        auto input = args[0].cast<intptr_t>();
        auto output = args[1].cast<intptr_t>();
        cpuinfer.submit(&MLP::forward, mlp,
                        (const void*)input, (void*)output);
    }

    static void bind_warm_up(CPUInfer& cpuinfer, MLP* mlp, py::args args, py::kwargs kwargs) {
        cpuinfer.submit(&MLP::warm_up, mlp);
    }

    static void bind_functions(CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
        auto mlp = func.attr("__self__").cast<MLP*>();
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
    static void bind_forward(CPUInfer& cpuinfer, MOE* moe, py::args args, py::kwargs kwargs) {
        int k = args[0].cast<int>();
        auto expert_ids = args[1].cast<intptr_t>();
        auto weights = args[2].cast<intptr_t>();
        auto input = args[3].cast<intptr_t>();
        auto output = args[4].cast<intptr_t>();
        cpuinfer.submit(&MOE::forward, moe,
                        k, (const uint64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output);
    }

    static void bind_warm_up(CPUInfer& cpuinfer, MOE* moe, py::args args, py::kwargs kwargs) {
        cpuinfer.submit(&MOE::warm_up, moe);
    }

    static void bind_functions(CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
        auto moe = func.attr("__self__").cast<MOE*>();
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

PYBIND11_MODULE(cpuinfer_ext, m) {
    auto kvcache_module = m.def_submodule("kvcache");

    // Defines the AnchorType enumeration
    py::enum_<AnchorType>(kvcache_module, "AnchorType")
        .value("FIXED", AnchorType::FIXED)
        .value("DYNAMIC", AnchorType::DYNAMIC);

    py::class_<KVCacheConfig>(kvcache_module, "KVCacheConfig")
        .def(py::init<int, int, int, int, int, int, AnchorType>())
        .def_readwrite("layer_num", &KVCacheConfig::layer_num)
        .def_readwrite("kv_head_num", &KVCacheConfig::kv_head_num)
        .def_readwrite("q_head_num", &KVCacheConfig::q_head_num)
        .def_readwrite("head_dim", &KVCacheConfig::head_dim)
        .def_readwrite("block_len", &KVCacheConfig::block_len)
        .def_readwrite("anchor_num", &KVCacheConfig::anchor_num)
        .def_readwrite("anchor_type", &KVCacheConfig::anchor_type);

    py::class_<KVCache>(kvcache_module, "KVCache")
        .def(py::init<KVCacheConfig>())
        .def("attn",
             [](KVCache& kvcache, intptr_t q_in, intptr_t output,
                intptr_t attn_lse, int layer_idx, int q_len, int batch_size,
                int max_block_num, intptr_t block_table, intptr_t cache_seqlens,
                int pick_block_num, int init_block_num, int local_block_num) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("update_one_block_fp16",
             [](KVCache& kvcache, intptr_t k_in, intptr_t v_in, int layer_id,
                int block_idx) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("get_one_block_fp16",
             [](KVCache& kvcache, intptr_t k_in, intptr_t v_in, int layer_id,
                int block_idx) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("update_importance_one_block",
             [](KVCache& kvcache, intptr_t importance, int layer_id,
                int block_idx) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("calc_anchor_all_layers",
             [](KVCache& kvcache, intptr_t block_table, intptr_t cache_seqlens,
                int batch_size, int max_block_num) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("load_kvcache",
             [](KVCache& kvcache, std::string tensor_file_path) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("dump_kvcache",
             [](KVCache& kvcache, intptr_t block_table, int cache_total_len,
                std::string tensor_file_path) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("get_importance_one_block",
             [](KVCache& kvcache, intptr_t importance, int layer_id,
                int block_idx) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("update_anchor_one_block",
             [](KVCache& kvcache, intptr_t anchor, int layer_id, int block_idx) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("get_anchor_one_block",
             [](KVCache& kvcache, intptr_t anchor, int layer_id, int block_idx) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("get_all_kv_one_layer",
             [](KVCache& kvcache, int layer_id, intptr_t k_in, intptr_t v_in,
                Backend* backend) {
                 throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
             })
        .def("get_cache_total_len", &KVCache::get_cache_total_len)
        .def("update_cache_total_len",
             [](KVCache& kvcache, int cache_total_len) {
                 kvcache.update_cache_total_len(cache_total_len);
             });

    auto linear_module = m.def_submodule("linear");

    py::class_<LinearConfig>(linear_module, "LinearConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride, intptr_t proj, int proj_type, int hidden_type) {
            return LinearConfig(hidden_size, intermediate_size, stride, (void*)proj, (ggml_type)proj_type, (ggml_type)hidden_type);
        }));

    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())
        .def("warm_up", [](Linear& linear) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        })
        .def("forward", [](Linear& linear, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        });

    auto mlp_module = m.def_submodule("mlp");

    py::class_<MLPConfig>(mlp_module, "MLPConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type, int hidden_type) {
            return MLPConfig(hidden_size, intermediate_size, stride, (void*)gate_proj, (void*)up_proj, (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type);
        }));

    py::class_<MLP>(mlp_module, "MLP")
        .def(py::init<MLPConfig>())
        .def("warm_up", [](MLP& mlp) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        })
        .def("forward", [](MLP& mlp, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        });

    auto moe_module = m.def_submodule("moe");

    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int hidden_size, int intermediate_size, int stride, intptr_t gate_proj, intptr_t up_proj, intptr_t down_proj, int gate_type, int up_type, int down_type, int hidden_type) {
            return MOEConfig(expert_num, hidden_size, intermediate_size, stride, (void*)gate_proj, (void*)up_proj, (void*)down_proj, (ggml_type)gate_type, (ggml_type)up_type, (ggml_type)down_type, (ggml_type)hidden_type);
        }));

    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up", [](MOE& moe) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        })
        .def("forward", [](MOE& moe, int k, uint64_t expert_ids, intptr_t weights, intptr_t input, intptr_t output) {
            throw std::runtime_error("!!! Doing nothing, please use CPUInfer.submit to call it!!!\n");
        });

    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit",
             [kvcache_module, linear_module, mlp_module, moe_module](CPUInfer& cpuinfer, py::object func, py::args args, py::kwargs kwargs) {
                 if (py::hasattr(func, "__self__") &&
                     py::hasattr(func, "__func__")) {
                     std::string class_name = py::str(func.attr("__self__")
                                                          .attr("__class__")
                                                          .attr("__name__"));
                     if (class_name == "KVCache") {
                         KVCacheBindings::bind_functions(cpuinfer, func,
                                                         args, kwargs);
                     } else if (class_name == "Linear") {
                         LinearBindings::bind_functions(cpuinfer, func,
                                                        args, kwargs);
                     } else if (class_name == "MLP") {
                         MLPBindings::bind_functions(cpuinfer, func,
                                                     args, kwargs);
                     } else if (class_name == "MOE") {
                         MOEBindings::bind_functions(cpuinfer, func,
                                                     args, kwargs);
                     } else {
                         // handle other classes
                         throw py::type_error("Unsupported class type: " +
                                              class_name);
                     }
                 } else {
                     // handle cases where func does not have __self__ or
                     // __func__
                     throw py::type_error(
                         "Invalid function object: missing "
                         "__self__ or __func__ attribute.");
                 }
             })
        .def("sync", &CPUInfer::sync);
}
