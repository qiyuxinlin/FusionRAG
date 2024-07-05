#ifndef CPUINFER_OPERATOR_MOE_H
#define CPUINFER_OPERATOR_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "../../cpu_backend/backend.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

struct MOEConfig {
    int expert_num;
    int hidden_size;
    int intermediate_size;
    int stride;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;

    MOEConfig() {}

    MOEConfig(int expert_num, int hidden_size, int intermediate_size, int stride, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type)
        : expert_num(expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
};

class MOE {
   public:
    MOE(MOEConfig);
    void warm_up(Backend* backend);
    void forward(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);

   public:
    MOEConfig config_;
    void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

    std::vector<float> input_fp32_;                      // [hidden_size]
    std::vector<uint8_t> gate_input_;                    // [hidden_size * 4]
    std::vector<uint8_t> up_input_;                      // [hidden_size * 4]
    std::vector<std::vector<float>> gate_output_;        // [expert_num, intermediate_size]
    std::vector<std::vector<float>> up_output_;          // [expert_num, intermediate_size]
    std::vector<std::vector<float>> intermediate_fp32_;  // [expert_num, intermediate_size]
    std::vector<std::vector<uint8_t>> down_input_;       // [expert_num, intermediate_size * 4]
    std::vector<std::vector<float>> down_output_;        // [expert_num, hidden_size]
    std::vector<float> output_fp32_;                     // [hidden_size]
};

#endif