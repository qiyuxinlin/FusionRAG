#include "moe.h"
#include <iostream>
#include "unistd.h"

int MOE::group_min_len = 10;
int MOE::group_max_len = 1024;

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
    forward(1, k, expert_ids.data(), weights.data(), input.data(), output.data(), backend);
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
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
        void* gate_proj_ptr = gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        float* gate_output_ptr = gate_output_[expert_idx].data() + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_proj_ptr = up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        float* up_output_ptr = up_output_[expert_idx].data() + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            intermediate_fp32_[expert_idx][i] = act_fn(gate_output_[expert_idx][i]) * up_output_[expert_idx][i];
        }
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = intermediate_fp32_[expert_idx].data() + ith * config_.stride;
            void* down_input_ptr = down_input_[expert_idx].data() + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    });
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(intermediate_fp32_[i].data(), down_input_[i].data(), config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];
            void* down_proj_ptr = down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            float* down_output_ptr = down_output_[expert_idx].data() + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_[expert_idx].data(), config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                output_fp32_[i] += down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = output_fp32_.data() + ith * config_.stride;
            void* output_ptr = output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    });
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(output_fp32_.data(), output, config_.hidden_size, config_.hidden_type);
    }
}

void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    std::vector<std::vector<float>> input_fp32(qlen);
    std::vector<std::vector<uint8_t>> gate_input(qlen);
    std::vector<std::vector<uint8_t>> up_input(qlen);
    auto start = std::chrono::high_resolution_clock::now();
    backend->do_work_stealing_job(qlen, [&](int i) {
        input_fp32[i].resize(config_.hidden_size);
        gate_input[i].resize(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
        up_input[i].resize(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        to_float(input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), input_fp32[i].data(), config_.hidden_size, config_.hidden_type);
        from_float(input_fp32[i].data(), gate_input[i].data(), config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        from_float(input_fp32[i].data(), up_input[i].data(), config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
    });
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken prepare: " << elapsed.count() << " s\n";
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> local_tokens(config_.expert_num);
    std::vector<std::vector<int>> local_pos(qlen, std::vector<int>(k));
    std::vector<std::vector<uint8_t>> local_gate_input(config_.expert_num);
    std::vector<std::vector<uint8_t>> local_up_input(config_.expert_num);
    std::vector<std::vector<float>> local_gate_output(config_.expert_num);
    std::vector<std::vector<float>> local_up_output(config_.expert_num);
    std::vector<std::vector<float>> local_intermediate_fp32(config_.expert_num);
    std::vector<std::vector<uint8_t>> local_down_input(config_.expert_num);
    std::vector<std::vector<float>> local_down_output(config_.expert_num);
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            local_pos[i][j] = local_tokens[expert_ids[j + i * k]].size();
            local_tokens[expert_ids[j + i * k]].push_back(i);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken serial: " << elapsed.count() << " s\n";
    start = std::chrono::high_resolution_clock::now();
    backend->do_work_stealing_job(config_.expert_num, [&](int i) {
        int local_num = local_tokens[i].size();
        if (local_num > 0) {
            local_gate_input[i].resize(local_num * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            local_up_input[i].resize(local_num * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
            local_gate_output[i].resize(local_num * config_.intermediate_size);
            local_up_output[i].resize(local_num * config_.intermediate_size);
            local_intermediate_fp32[i].resize(local_num * config_.intermediate_size);
            local_down_input[i].resize(local_num * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
            local_down_output[i].resize(local_num * config_.hidden_size);
            for (int j = 0; j < local_num; j++) {
                memcpy(local_gate_input[i].data() + j * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input[local_tokens[i][j]].data(), config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
                memcpy(local_up_input[i].data() + j * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input[local_tokens[i][j]].data(), config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
            }
        }
    });
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken for alloc: " << elapsed.count() << " s\n";
    int stride = QK_K;
    int nth = config_.intermediate_size / stride;
    start = std::chrono::high_resolution_clock::now();
    backend->do_work_stealing_job(nth * config_.expert_num, [&](int task_id) {
        int expert_idx = task_id / nth;
        int ith = task_id % nth;
        int local_num = local_tokens[expert_idx].size();
        if (local_num > 0) {
            void* gate_input_ptr = local_gate_input[expert_idx].data();
            void* gate_proj_ptr = gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
            float* gate_output_ptr = local_gate_output[expert_idx].data() + ith * stride;
            llamafile_sgemm(stride, local_num, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            void* up_input_ptr = local_up_input[expert_idx].data();
            void* up_proj_ptr = up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
            float* up_output_ptr = local_up_output[expert_idx].data() + ith * stride;
            llamafile_sgemm(stride, local_num, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = 0; i < local_num; i++) {
                for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                    local_intermediate_fp32[expert_idx][i * config_.intermediate_size + j] = act_fn(local_gate_output[expert_idx][i * config_.intermediate_size + j]) * local_up_output[expert_idx][i * config_.intermediate_size + j];
                }
                float* intermediate_fp32_ptr = local_intermediate_fp32[expert_idx].data() + i * config_.intermediate_size + ith * stride;
                void* down_input_ptr = local_down_input[expert_idx].data() + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
                from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            }
        }
    });
    stride = QK_K;
    nth = config_.hidden_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, [&](int task_id) {
        int expert_idx = task_id / nth;
        int ith = task_id % nth;
        int local_num = local_tokens[expert_idx].size();
        if (local_num > 0) {
            void* down_input_ptr = local_down_input[expert_idx].data();
            void* down_proj_ptr = down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            float* down_output_ptr = local_down_output[expert_idx].data() + ith * stride;
            llamafile_sgemm(stride, local_num, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        }
    });
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken for sgemm: " << elapsed.count() << " s\n";
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> output_fp32(qlen);
    backend->do_work_stealing_job(qlen, [&](int i) {
        output_fp32[i].resize(config_.hidden_size);
        for (int e = 0; e < config_.hidden_size; e++) {
            output_fp32[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                output_fp32[i][e] += local_down_output[expert_ids[j + i * k]][local_pos[i][j] * config_.hidden_size + e] * weights[j + i * k];
            }
        }
        from_float(output_fp32[i].data(), output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    });
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time taken for reduce: " << elapsed.count() << " s\n";
}

void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    if (qlen < group_min_len) {
        for (int i = 0; i < qlen; i++) {
            forward_one(k, expert_ids + i * k, weights + i * k, (input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type)), output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
        }
        return;
    }
    if (qlen > group_max_len) {
        }
}