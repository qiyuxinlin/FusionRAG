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

    s_input_fp32_.resize(config_.hidden_size);
    s_gate_input_.resize(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
    s_up_input_.resize(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
    s_gate_output_.resize(config_.expert_num);
    s_up_output_.resize(config_.expert_num);
    s_intermediate_fp32_.resize(config_.expert_num);
    s_down_input_.resize(config_.expert_num);
    s_down_output_.resize(config_.expert_num);
    for (int i = 0; i < config_.expert_num; i++) {
        s_gate_output_[i].resize(config_.intermediate_size);
        s_up_output_[i].resize(config_.intermediate_size);
        s_intermediate_fp32_[i].resize(config_.intermediate_size);
        s_down_input_[i].resize(config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
        s_down_output_[i].resize(config_.hidden_size);
    }
    s_output_fp32_.resize(config_.hidden_size);

    m_input_fp32_.resize(group_max_len);
    m_gate_input_.resize(group_max_len);
    m_up_input_.resize(group_max_len);
    for (int i = 0; i < group_max_len; i++) {
        m_input_fp32_[i].resize(config_.hidden_size);
        m_gate_input_[i].resize(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
        m_up_input_[i].resize(config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
    }
    m_local_pos.resize(group_max_len);
    for (int i = 0; i < group_max_len; i++) {
        m_local_pos[i].reserve(config_.expert_num);
    }
    m_local_tokens_.resize(config_.expert_num);
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_tokens_[i].reserve(group_max_len);
    }
    m_local_gate_input_.resize(config_.expert_num);
    m_local_up_input_.resize(config_.expert_num);
    m_local_gate_output_.resize(config_.expert_num);
    m_local_up_output_.resize(config_.expert_num);
    m_local_intermediate_fp32_.resize(config_.expert_num);
    m_local_down_input_.resize(config_.expert_num);
    m_local_down_output_.resize(config_.expert_num);
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_[i].resize(group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
        m_local_up_input_[i].resize(group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        m_local_gate_output_[i].resize(group_max_len * config_.intermediate_size);
        m_local_up_output_[i].resize(group_max_len * config_.intermediate_size);
        m_local_intermediate_fp32_[i].resize(group_max_len * config_.intermediate_size);
        m_local_down_input_[i].resize(group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type));
        m_local_down_output_[i].resize(group_max_len * config_.hidden_size);
    }
    m_output_fp32_.resize(group_max_len);
    for (int i = 0; i < group_max_len; i++) {
        m_output_fp32_[i].resize(config_.hidden_size);
    }
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
        to_float(input, s_input_fp32_.data(), config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_.data(), s_gate_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_.data();
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_.data(), s_gate_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_.data();
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_.data(), s_up_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_.data();
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
        float* gate_output_ptr = s_gate_output_[expert_idx].data() + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_proj_ptr = up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        float* up_output_ptr = s_up_output_[expert_idx].data() + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx].data() + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx].data() + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    });
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i].data(), s_down_input_[i].data(), config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];
            void* down_proj_ptr = down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            float* down_output_ptr = s_down_output_[expert_idx].data() + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx].data(), config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_.data() + ith * config_.stride;
            void* output_ptr = output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    });
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(s_output_fp32_.data(), output, config_.hidden_size, config_.hidden_type);
    }
}

void MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    std::vector<std::vector<int>> local_tokens(config_.expert_num);
    std::vector<std::vector<int>> local_pos(qlen, std::vector<int>(k));
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            local_pos[i][j] = local_tokens[expert_ids[j + i * k]].size();
            local_tokens[expert_ids[j + i * k]].push_back(i);
        }
    }
    backend->do_work_stealing_job(qlen, [&](int i) {
        to_float(input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i].data(), config_.hidden_size, config_.hidden_type);
        from_float(m_input_fp32_[i].data(), m_gate_input_[i].data(), config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        from_float(m_input_fp32_[i].data(), m_up_input_[i].data(), config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
    });
    backend->do_work_stealing_job(config_.expert_num, [&](int i) {
        int local_num = local_tokens[i].size();
        if (local_num > 0) {
            for (int j = 0; j < local_num; j++) {
                memcpy(m_local_gate_input_[i].data() + j * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), m_gate_input_[local_tokens[i][j]].data(), config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
                memcpy(m_local_up_input_[i].data() + j * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), m_up_input_[local_tokens[i][j]].data(), config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
            }
        }
    });
    int stride = QK_K;
    int nth = config_.intermediate_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, [&](int task_id) {
        int expert_idx = task_id / nth;
        int ith = task_id % nth;
        int local_num = local_tokens[expert_idx].size();
        if (local_num > 0) {
            void* gate_input_ptr = m_local_gate_input_[expert_idx].data();
            void* gate_proj_ptr = gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
            float* gate_output_ptr = m_local_gate_output_[expert_idx].data() + ith * stride;
            llamafile_sgemm(stride, local_num, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            void* up_input_ptr = m_local_up_input_[expert_idx].data();
            void* up_proj_ptr = up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
            float* up_output_ptr = m_local_up_output_[expert_idx].data() + ith * stride;
            llamafile_sgemm(stride, local_num, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = 0; i < local_num; i++) {
                for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                    m_local_intermediate_fp32_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_[expert_idx][i * config_.intermediate_size + j];
                }
                float* intermediate_fp32_ptr = m_local_intermediate_fp32_[expert_idx].data() + i * config_.intermediate_size + ith * stride;
                void* down_input_ptr = m_local_down_input_[expert_idx].data() + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
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
            void* down_input_ptr = m_local_down_input_[expert_idx].data();
            void* down_proj_ptr = down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            float* down_output_ptr = m_local_down_output_[expert_idx].data() + ith * stride;
            llamafile_sgemm(stride, local_num, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        }
    });
    backend->do_work_stealing_job(qlen, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_[expert_ids[j + i * k]][local_pos[i][j] * config_.hidden_size + e] * weights[j + i * k];
            }
        }
        from_float(m_output_fp32_[i].data(), output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    });
}

void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    if (qlen < group_min_len) {
        for (int i = 0; i < qlen; i++) {
            forward_one(k, expert_ids + i * k, weights + i * k, input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
        }
        return;
    }
    int forward_len = std::min(group_max_len, qlen);
    forward_many(forward_len, k, expert_ids, weights, input, output, backend);
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
}