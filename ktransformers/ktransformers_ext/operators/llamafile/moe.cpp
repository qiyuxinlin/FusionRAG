#include "moe.h"

MOE::MOE(MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    input_fp32_.resize(config_.hidden_size);
    gate_input_.resize(config_.hidden_size * 4);
    up_input_.resize(config_.hidden_size * 4);
    gate_output_.resize(config_.expert_num);
    up_output_.resize(config_.expert_num);
    intermediate_fp32_.resize(config_.expert_num);
    down_input_.resize(config_.expert_num);
    down_output_.resize(config_.expert_num);
    for (int i = 0; i < config_.expert_num; i++) {
        gate_output_[i].resize(config_.intermediate_size);
        up_output_[i].resize(config_.intermediate_size);
        intermediate_fp32_[i].resize(config_.intermediate_size);
        down_input_[i].resize(config_.intermediate_size * 4);
        down_output_[i].resize(config_.hidden_size);
    }
    output_fp32_.resize(config_.hidden_size);
}

void MOE::warm_up(Backend* backend) {
    int k = config_.expert_num;
    std::vector<uint64_t> expert_ids(k);
    std::vector<float> weights(k);
    for (int i = 0; i < k; i++) {
        expert_ids[i] = i;
        weights[i] = 0;
    }
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    forward(k, expert_ids.data(), weights.data(), input.data(), output.data(), backend);
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void MOE::forward(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, input_fp32_.data(), config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(input_fp32_.data(), gate_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = gate_input_.data();
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(input_fp32_.data(), gate_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = gate_input_.data();
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(input_fp32_.data(), up_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = up_input_.data();
            } else {
                up_input_ptr = input;
            }
        }
    }
    int nth = config_.intermediate_size / config_.stride;
    backend->do_work_stealing_job(nth * k, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
        llamafile_sgemm(config_.intermediate_size, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ + expert_id * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type), config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_[expert_idx].data(), config_.intermediate_size, ith, nth, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        llamafile_sgemm(config_.intermediate_size, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ + expert_id * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type), config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_[expert_idx].data(), config_.intermediate_size, ith, nth, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    });
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < config_.intermediate_size; j++) {
            intermediate_fp32_[i][j] = act_fn(gate_output_[i][j]) * up_output_[i][j];
        }
    }
    for (int i = 0; i < k; i++) {
        from_float(intermediate_fp32_[i].data(), down_input_[i].data(), config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
    }
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth * k, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
        llamafile_sgemm(config_.hidden_size, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ + expert_id * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type), config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_[expert_idx].data(), config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_[expert_idx].data(), config_.hidden_size, ith, nth, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    });
    for (int i = 0; i < config_.hidden_size; i++) {
        output_fp32_[i] = 0;
        for (int j = 0; j < k; j++) {
            output_fp32_[i] += down_output_[j][i] * weights[j];
        }
    }
    from_float(output_fp32_.data(), output, config_.hidden_size, config_.hidden_type);
}
