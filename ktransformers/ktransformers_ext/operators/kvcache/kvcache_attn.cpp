#include "kvcache.h"

void KVCache::attn(const ggml_fp16_t *q_in, ggml_fp16_t *output,
                   float *attn_lse, int layer_idx, int q_len, int batch_size,
                   int max_block_num, int *block_table, int *cache_seqlens,
                   int pick_block_num, int init_block_num, int local_block_num,
                   Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_idx;
    int thread_num = backend->get_thread_num();
    batch_size = batch_size * q_len;
    // printf("layer idx: %d, q_len: %d, batch_size: %d, max_block_num: %d, "
    //        "pick_block_num: %d, init_block_num: %d, local_block_num: %d\n",
    //        layer_idx, q_len, batch_size, max_block_num, pick_block_num,
    //        init_block_num, local_block_num);

    // if (cache_seqlens != nullptr) {
    //     for (int i = 0; i < batch_size; i++) {
    //         printf("cache_seqlens[%d]: %d\n", i, cache_seqlens[i]);
    //         for (int j = 0; j < max_block_num; j++) {
    //             printf("block_table[%d]: %d\n", j,
    //                    block_table[i * max_block_num + j]);
    //         }
    //     }
    // }

    const uint16_t *q_in_data = const_cast<const uint16_t *>(q_in);

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
            // quantize q
            for (int i = 0; i < config_.kv_head_num; i++) {
                for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
                    q_fp32_[batch_idx][i][j] = GGML_FP16_TO_FP32(
                        q_in_data[batch_idx * config_.kv_head_num * n_gqa_ *
                                      config_.head_dim +
                                  i * n_gqa_ * config_.head_dim + j]);
                }
            }
        } else {
            // quantize q
            for (int i = 0; i < config_.kv_head_num; i++) {
                for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
                    q_fp32[j] = GGML_FP16_TO_FP32(
                        q_in_data[batch_idx * config_.kv_head_num * n_gqa_ *
                                      config_.head_dim +
                                  i * n_gqa_ * config_.head_dim + j]);
                }
                quantize_row_q8_0(q_fp32.data(), q_q8_0_[batch_idx][i].data(),
                                  n_gqa_ * config_.head_dim);
            }
        }

        // initialize output_fp32_ and attn_lse_
        for (int i = 0; i < config_.kv_head_num; i++) {
            for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
                output_fp32_[batch_idx][i][j] = 0;
            }
            for (int j = 0; j < n_gqa_; j++) {
                attn_lse_[batch_idx][i][j] = 0;
            }
        }
        // clear top_similar_block_

        while (!top_similar_block_[batch_idx].empty())
            top_similar_block_[batch_idx].pop();
    }

    // get block_table_before_retrieval_ and cache_seqlens_
    if (block_table == nullptr) {
        max_block_num = past_block_num_[layer_idx];
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            if (cache_total_len_ != 0)
                cache_seqlens_[batch_idx] = cache_total_len_;
            else
                cache_seqlens_[batch_idx] = max_block_num * config_.block_len;
            for (int i = 0; i < max_block_num; i++) {
                block_table_before_retrieval_[batch_idx][i] = i;
                block_similar_[batch_idx][i] = 0;
            }
        }
    } else {
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            cache_seqlens_[batch_idx] = cache_seqlens[batch_idx];
            for (int i = 0; i < max_block_num; i++) {
                block_table_before_retrieval_[batch_idx][i] =
                    block_table[batch_idx * max_block_num + i];
                block_similar_[batch_idx][i] = 0;
            }
        }
    }

    // retrieval kvcache, get the init_block_num block at beginning, top
    // pick_block_num similar and last local_block_num blocks. Each task
    // calculates the simlarity of a certain block with the query, then push the
    // block into the priority queue. Finally, the required blocks are pushed
    // into the block_table_after_retrieval_.
    int max_block_num_after_retrieval = 0;
    // printf("%d\n", cache_seqlens[0]);
    if (pick_block_num != -1) {
        max_block_num_after_retrieval =
            std::min(max_block_num,
                     init_block_num + pick_block_num + local_block_num + 1);
        auto start_1 = std::chrono::high_resolution_clock::now();

        if (batch_size == 1 &&
            config_.anchor_num == 1) { // TODO: improve batch_size > 1
            for (int batch_id = 0; batch_id < batch_size; batch_id++) {
                if (q_len == 1) {
                    for (int j = 0; j < config_.head_dim * config_.q_head_num;
                         j++) {
                        avg_q[batch_id][j] = GGML_FP16_TO_FP32(
                            q_in_data[batch_id * q_len * config_.q_head_num *
                                          config_.head_dim +
                                      j]);
                        avg_q_fp16[batch_id][j] =
                            q_in_data[batch_id * q_len * config_.q_head_num *
                                          config_.head_dim +
                                      j];
                    }
                } else {
                    for (int j = 0; j < config_.head_dim * config_.q_head_num;
                         j++) {
                        avg_q[batch_id][j] = 0;
                    }
                    for (int i = 0; i < q_len; i++) {
                        for (int j = 0; j < config_.head_dim; j++) {
                            avg_q[batch_id][j] += GGML_FP16_TO_FP32(
                                q_in_data[batch_id * q_len *
                                              config_.q_head_num *
                                              config_.head_dim +
                                          i * config_.q_head_num *
                                              config_.head_dim +
                                          j]);
                        }
                    }
                    for (int j = 0; j < config_.head_dim * config_.q_head_num;
                         j++) {
                        avg_q[batch_id][j] /= q_len;
                        avg_q_fp16[batch_id][j] =
                            GGML_FP32_TO_FP16(avg_q[batch_id][j]);
                    }
                }
                int seq_len = cache_seqlens_[batch_id];
                int block_num = (seq_len / config_.block_len) -
                                local_block_num - init_block_num;
                if (block_num <= 0) {
                    continue;
                }
                bool is_seq = true;
                for (int i = init_block_num + 1;
                     i < (seq_len / config_.block_len) - local_block_num; i++) {
                    if (block_table_before_retrieval_[batch_id][i] !=
                        block_table_before_retrieval_[batch_id][i - 1] + 1) {
                        is_seq = false;
                        break;
                    }
                }
                if (is_seq) {
                    int nth = backend->get_thread_num();
                    backend->do_work_stealing_job(
                        nth, nullptr,
                        [&](int task_id) {
                            int ith = task_id;
                            bool ok = llamafile_sgemm(
                                block_num, 1,
                                config_.q_head_num * config_.head_dim,
                                anchor_.data() +
                                    (layer_idx * config_.max_block_num +
                                     block_table_before_retrieval_
                                         [batch_id][init_block_num]) *
                                        config_.anchor_num *
                                        config_.q_head_num * config_.head_dim,
                                config_.q_head_num * config_.head_dim,
                                avg_q_fp16[batch_id].data(),
                                config_.q_head_num * config_.head_dim,
                                block_similar_[batch_id].data() +
                                    init_block_num,
                                block_num, ith, nth, GGML_TASK_TYPE_COMPUTE,
                                GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32,
                                GGML_PREC_DEFAULT);
                            if (!ok) {
                                printf("llamafile_sgemm failed\n");
                            }
                        },
                        nullptr);
                } else {
                    backend->do_work_stealing_job(
                        block_num, nullptr,
                        [&](int task_id) {
                            int block_id = task_id + init_block_num;
                            int block_idx =
                                block_table_before_retrieval_[batch_id]
                                                             [block_id];
                            bool ok = llamafile_sgemm(
                                1, 1, config_.q_head_num * config_.head_dim,
                                anchor_.data() +
                                    (layer_idx * config_.max_block_num +
                                     block_table_before_retrieval_[batch_id]
                                                                  [block_idx]) *
                                        config_.anchor_num *
                                        config_.q_head_num * config_.head_dim,
                                config_.q_head_num * config_.head_dim,
                                avg_q_fp16[batch_id].data(),
                                config_.q_head_num * config_.head_dim,
                                block_similar_[batch_id].data() + block_id, 1,
                                0, 1, GGML_TASK_TYPE_COMPUTE, GGML_TYPE_F16,
                                GGML_TYPE_F16, GGML_TYPE_F32,
                                GGML_PREC_DEFAULT);
                            if (!ok) {
                                printf("llamafile_sgemm failed\n");
                            }
                        },
                        nullptr);
                }
            }
        } else {
            backend->do_work_stealing_job(
                batch_size * max_block_num, nullptr,
                [&](int task_id) {
                    int batch_id = task_id / max_block_num;
                    int block_id = task_id % max_block_num;
                    int seq_len = cache_seqlens_[batch_id];

                    if (block_id < init_block_num ||
                        block_id >=
                            (seq_len / config_.block_len) - local_block_num) {
                        return;
                    }

                    int block_idx =
                        block_table_before_retrieval_[batch_id][block_id];
                    float sim = 0;

                    for (int head_id = 0; head_id < config_.q_head_num;
                         head_id++) {
                        for (int i = 0; i < config_.head_dim; i++) {
                            float q_i = 0,
                                  qa_i = std::numeric_limits<float>::lowest();
                            for (int q_id = 0; q_id < q_len; q_id++) {
                                q_i += GGML_FP16_TO_FP32(
                                    q_in_data[batch_id * q_len *
                                                  config_.q_head_num *
                                                  config_.head_dim +
                                              q_id * config_.q_head_num *
                                                  config_.head_dim +
                                              head_id * config_.head_dim + i]);
                            }
                            q_i /= q_len;
                            for (int anchor_id = 0;
                                 anchor_id < config_.anchor_num; anchor_id++) {
                                qa_i = std::max(
                                    qa_i,
                                    GGML_FP16_TO_FP32(
                                        anchor_[layer_idx *
                                                    config_.max_block_num *
                                                    config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                block_idx * config_.anchor_num *
                                                    config_.q_head_num *
                                                    config_.head_dim +
                                                anchor_id * config_.q_head_num *
                                                    config_.head_dim +
                                                head_id * config_.head_dim +
                                                i]) *
                                        q_i);
                                // printf(
                                //     "layer_idx: %d, block_idx: %d, anchor_id:
                                //     %d, " "head_id: %d, i: %d\n", layer_idx,
                                //     block_idx, anchor_id, head_id, i);
                                // printf("q: %f, a: %f\n", q_i,
                                //        GGML_FP16_TO_FP32(
                                //            anchor_[layer_idx][block_idx][anchor_id]
                                //                   [head_id][i]));
                            }
                            sim += qa_i;
                        }
                    }
                    // printf("batch_id: %d, block_id: %d, sim: %f\n", batch_id,
                    //        block_id, sim);
                    block_similar_[batch_id][block_id] = sim;
                },
                nullptr);
        }
        auto end_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_1 = end_1 - start_1;
        // printf("layer %d time of calculating similarity: %f s\n", layer_idx,
        //        diff_1.count());

        auto start_2 = std::chrono::high_resolution_clock::now();

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            if (cache_seqlens_[batch_idx] / config_.block_len <=
                init_block_num + pick_block_num + local_block_num) {
                block_table_after_retrieval_[batch_idx].swap(
                    block_table_before_retrieval_[batch_idx]);
                continue;
            }

            for (int block_id = init_block_num;
                 block_id < (cache_seqlens_[batch_idx] / config_.block_len) -
                                local_block_num;
                 block_id++) {
                top_similar_block_[batch_idx].push(std::make_pair(
                    block_similar_[batch_idx][block_id],
                    block_table_before_retrieval_[batch_idx][block_id]));
                if (top_similar_block_[batch_idx].size() > pick_block_num) {
                    top_similar_block_[batch_idx].pop();
                }
            }

            int i = 0;
            for (; i < init_block_num; i++) {
                block_table_after_retrieval_[batch_idx][i] =
                    block_table_before_retrieval_[batch_idx][i];
            }
            while (!top_similar_block_[batch_idx].empty()) {
                block_table_after_retrieval_[batch_idx][i] =
                    top_similar_block_[batch_idx].top().second;
                top_similar_block_[batch_idx].pop();
                i++;
            }
            for (; i < init_block_num + pick_block_num + local_block_num; i++) {
                block_table_after_retrieval_[batch_idx][i] =
                    block_table_before_retrieval_
                        [batch_idx]
                        [(cache_seqlens_[batch_idx] / config_.block_len) -
                         local_block_num + i - init_block_num - pick_block_num];
            }
            if (cache_seqlens_[batch_idx] % config_.block_len != 0) {
                block_table_after_retrieval_[batch_idx][i] =
                    block_table_before_retrieval_[batch_idx][(
                        cache_seqlens_[batch_idx] / config_.block_len)];
                i++;
            }
            // printf("cache_seqlens_[%d]: %d\n", batch_idx,
            //        cache_seqlens_[batch_idx]);
            cache_seqlens_[batch_idx] =
                (cache_seqlens_[batch_idx] % config_.block_len) +
                i * config_.block_len;

            // printf("cache_seqlens_[%d]: %d\n", batch_idx,
            //        cache_seqlens_[batch_idx]);
            // for (int j = 0; j < i; j++) {
            //     printf("block_table_after_retrieval_[%d][%d]: %d\n",
            //     batch_idx,
            //            j, block_table_after_retrieval_[batch_idx][j]);
            // }
        }

        auto end_2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_2 = end_2 - start_2;
        // printf("layer %d time of selecting blocks: %f s\n", layer_idx,
        //        diff_2.count());
    } else {
        max_block_num_after_retrieval = max_block_num;
        block_table_after_retrieval_.swap(block_table_before_retrieval_);
    }

    // Calculate the attention output. Each task calculates the
    // attention output of a certain block.
    // each thread has a local_cur_output, thread merge local_cur_output to
    // output when batch_id or head_id changes.
    seq_len_ = config_.block_len;

    // if (true) {
    //     for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    //         printf("cache_seqlens_[%d]: %d\n", batch_idx,
    //                cache_seqlens_[batch_idx]);
    //         for (int i = 0; i < max_block_num; i++) {
    //             printf("block_table_after_retrieval_[%d][%d]: %d\n",
    //             batch_idx,
    //                    i, block_table_after_retrieval_[batch_idx][i]);
    //         }
    //     }
    // }

    // printf("max_block_num_after_retrieval: %d\n",
    //        max_block_num_after_retrieval);
    backend->do_work_stealing_job(
        batch_size * config_.kv_head_num * max_block_num_after_retrieval,
        [&](int thread_id) {
            thread_cur_head_idx_[thread_id].first = -1;
            thread_cur_head_idx_[thread_id].second = -1;
        },
        [&](int task_id) {
            int batch_id =
                task_id / (config_.kv_head_num * max_block_num_after_retrieval);
            int head_id = (task_id % (config_.kv_head_num *
                                      max_block_num_after_retrieval)) /
                          max_block_num_after_retrieval;
            int block_id = task_id % max_block_num_after_retrieval;
            int thread_id = Backend::thread_local_id;
            // printf("batch_id: %d, head_id: %d, block_id: %d, thread_id: %d\n
            // ",
            //        batch_id, head_id, block_id, thread_id);
            // If the block is out of the sequence length, skip it.
            if (cache_seqlens_[batch_id] / config_.block_len < block_id) {
                return;
            }
            int block_idx = block_table_after_retrieval_[batch_id][block_id];
            if (cache_seqlens_[batch_id] / config_.block_len == block_id) {
                int seq_len = cache_seqlens_[batch_id] % config_.block_len;
                if (seq_len == 0)
                    return;

                // printf("seq_len: %d\n", seq_len);
                // Prepare the attention mask for the last block.
                int full_blocks = seq_len / 8;
                int remaining_bits = seq_len % 8;
                // printf("full_blocks: %d, remaining_bits: %d\n", full_blocks,
                //        remaining_bits);
                // Fill full blocks with 1s
                for (int i = 0; i < full_blocks; ++i) {
                    thread_local_attn_mask_[thread_id][i] = 0xFF;
                }
                // Fill the remaining bits in the next block
                if (remaining_bits > 0 && full_blocks < seq_len_ / 8) {
                    thread_local_attn_mask_[thread_id][full_blocks] =
                        (1 << remaining_bits) - 1;
                } else {
                    thread_local_attn_mask_[thread_id][full_blocks] = 0;
                }

                for (int i = full_blocks + 1; i < seq_len_ / 8; ++i) {
                    thread_local_attn_mask_[thread_id][i] = 0;
                }
                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {

                    // q_in_data[batch_id * config_.kv_head_num * n_gqa_ *
                    //                      config_.head_dim +
                    //                  head_id * n_gqa_ * config_.head_dim]
                    attn_with_kvcache_one_block(
                        config_.head_dim,
                        config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                        (void *)&q_in_data[batch_id * config_.kv_head_num *
                                               n_gqa_ * config_.head_dim +
                                           head_id * n_gqa_ * config_.head_dim],
                        seq_len_, 0, false,
                        thread_local_attn_mask_[thread_id].data(),
                        GGML_TYPE_F16, 0,
                        k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr, GGML_TYPE_F16, 1,
                        v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr,
                        thread_local_attn_score_[thread_id].data(),
                        thread_local_output_fp32_[thread_id].data(),
                        thread_local_attn_lse_[thread_id].data(),
                        thread_local_draft_[thread_id].data(), nullptr,
                        cos_.data(), sin_.data());
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                    attn_with_kvcache_one_block(
                        config_.head_dim,
                        config_.q_head_num / config_.kv_head_num,
                        GGML_TYPE_Q8_0, q_q8_0_[batch_id][head_id].data(),
                        seq_len_, 0, false,
                        thread_local_attn_mask_[thread_id].data(),
                        GGML_TYPE_Q4_0, 0,
                        k_cache_q4[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr, GGML_TYPE_Q4_0, 1,
                        v_cache_q4[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr,
                        thread_local_attn_score_[thread_id].data(),
                        thread_local_output_q8_0_[thread_id].data(),
                        thread_local_attn_lse_[thread_id].data(),
                        thread_local_draft_[thread_id].data(), nullptr,
                        cos_.data(), sin_.data());
                    dequantize_row_q8_0(
                        thread_local_output_q8_0_[thread_id].data(),
                        thread_local_output_fp32_[thread_id].data(),
                        n_gqa_ * config_.head_dim);
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                    attn_with_kvcache_one_block(
                        config_.head_dim,
                        config_.q_head_num / config_.kv_head_num,
                        GGML_TYPE_Q8_0, q_q8_0_[batch_id][head_id].data(),
                        seq_len_, 0, false,
                        thread_local_attn_mask_[thread_id].data(),
                        GGML_TYPE_Q8_0, 0,
                        k_cache_q8[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr, GGML_TYPE_Q8_0, 1,
                        v_cache_q8[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr,
                        thread_local_attn_score_[thread_id].data(),
                        thread_local_output_q8_0_[thread_id].data(),
                        thread_local_attn_lse_[thread_id].data(),
                        thread_local_draft_[thread_id].data(), nullptr,
                        cos_.data(), sin_.data());
                    dequantize_row_q8_0(
                        thread_local_output_q8_0_[thread_id].data(),
                        thread_local_output_fp32_[thread_id].data(),
                        n_gqa_ * config_.head_dim);
                }
            } else {
                if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                    attn_with_kvcache_one_block(
                        config_.head_dim,
                        config_.q_head_num / config_.kv_head_num, GGML_TYPE_F16,
                        (void *)&q_in_data[batch_id * config_.kv_head_num *
                                               n_gqa_ * config_.head_dim +
                                           head_id * n_gqa_ * config_.head_dim],
                        seq_len_, 0, true, nullptr, GGML_TYPE_F16, 0,
                        k_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr, GGML_TYPE_F16, 1,
                        v_cache_fp16_[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr,
                        thread_local_attn_score_[thread_id].data(),
                        thread_local_output_fp32_[thread_id].data(),
                        thread_local_attn_lse_[thread_id].data(),
                        thread_local_draft_[thread_id].data(), nullptr,
                        cos_.data(), sin_.data());
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                    attn_with_kvcache_one_block(
                        config_.head_dim,
                        config_.q_head_num / config_.kv_head_num,
                        GGML_TYPE_Q8_0, q_q8_0_[batch_id][head_id].data(),
                        seq_len_, 0, true, nullptr, GGML_TYPE_Q4_0, 0,
                        k_cache_q4[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr, GGML_TYPE_Q4_0, 1,
                        v_cache_q4[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr,
                        thread_local_attn_score_[thread_id].data(),
                        thread_local_output_q8_0_[thread_id].data(),
                        thread_local_attn_lse_[thread_id].data(),
                        thread_local_draft_[thread_id].data(), nullptr,
                        cos_.data(), sin_.data());
                    dequantize_row_q8_0(
                        thread_local_output_q8_0_[thread_id].data(),
                        thread_local_output_fp32_[thread_id].data(),
                        n_gqa_ * config_.head_dim);
                } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                    attn_with_kvcache_one_block(
                        config_.head_dim,
                        config_.q_head_num / config_.kv_head_num,
                        GGML_TYPE_Q8_0, q_q8_0_[batch_id][head_id].data(),
                        seq_len_, 0, true, nullptr, GGML_TYPE_Q8_0, 0,
                        k_cache_q8[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr, GGML_TYPE_Q8_0, 1,
                        v_cache_q8[layer_id_][head_id][block_idx].data(), 0,
                        nullptr, nullptr,
                        thread_local_attn_score_[thread_id].data(),
                        thread_local_output_q8_0_[thread_id].data(),
                        thread_local_attn_lse_[thread_id].data(),
                        thread_local_draft_[thread_id].data(), nullptr,
                        cos_.data(), sin_.data());
                    dequantize_row_q8_0(
                        thread_local_output_q8_0_[thread_id].data(),
                        thread_local_output_fp32_[thread_id].data(),
                        n_gqa_ * config_.head_dim);
                }
            }
            int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
            int cur_head_id = thread_cur_head_idx_[thread_id].second;
            if (batch_id == cur_batch_idx && head_id == cur_head_id) {
                for (int i = 0; i < n_gqa_; i++) {
                    float new_attn_lse =
                        thread_local_cur_attn_lse_[thread_id][i] +
                        std::log(
                            1.0 +
                            std::exp(thread_local_attn_lse_[thread_id][i] -
                                     thread_local_cur_attn_lse_[thread_id][i]));
                    ggml_vec_scale_f32(
                        config_.head_dim,
                        thread_local_cur_output_fp32_[thread_id].data() +
                            i * config_.head_dim,
                        std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                 new_attn_lse));
                    ggml_vec_scale_f32(
                        config_.head_dim,
                        thread_local_output_fp32_[thread_id].data() +
                            i * config_.head_dim,
                        std::exp(thread_local_attn_lse_[thread_id][i] -
                                 new_attn_lse));
                    for (int j = 0; j < config_.head_dim; j++) {
                        thread_local_cur_output_fp32_[thread_id]
                                                     [i * config_.head_dim +
                                                      j] +=
                            thread_local_output_fp32_[thread_id]
                                                     [i * config_.head_dim + j];
                    }
                    thread_local_cur_attn_lse_[thread_id][i] = new_attn_lse;
                }
            } else {
                if (cur_batch_idx != -1) {
                    mutex_[cur_batch_idx][cur_head_id]->lock();
                    for (int i = 0; i < n_gqa_; i++) {
                        if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) <
                            1e-6) {
                            attn_lse_[cur_batch_idx][cur_head_id][i] =
                                thread_local_cur_attn_lse_[thread_id][i];
                            for (int j = 0; j < config_.head_dim; j++) {
                                output_fp32_[cur_batch_idx][cur_head_id]
                                            [i * config_.head_dim + j] =
                                                thread_local_cur_output_fp32_
                                                    [thread_id]
                                                    [i * config_.head_dim + j];
                            }
                            // printf("cur_batch_idx: %d, cur_head_id: %d, i:
                            // %d, "
                            //        "output_fp32_[cur_batch_idx][cur_head_id]:
                            //        "
                            //        "%f\n ",
                            //        cur_batch_idx, cur_head_id, i,
                            //        output_fp32_[cur_batch_idx][cur_head_id]
                            //                    [i * config_.head_dim]);
                            continue;
                        }
                        float new_attn_lse =
                            attn_lse_[cur_batch_idx][cur_head_id][i] +
                            std::log(
                                1.0 +
                                std::exp(
                                    thread_local_cur_attn_lse_[thread_id][i] -
                                    attn_lse_[cur_batch_idx][cur_head_id][i]));
                        ggml_vec_scale_f32(
                            config_.head_dim,
                            output_fp32_[cur_batch_idx][cur_head_id].data() +
                                i * config_.head_dim,
                            std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] -
                                     new_attn_lse));
                        ggml_vec_scale_f32(
                            config_.head_dim,
                            thread_local_cur_output_fp32_[thread_id].data() +
                                i * config_.head_dim,
                            std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                     new_attn_lse));
                        for (int j = 0; j < config_.head_dim; j++) {
                            output_fp32_[cur_batch_idx][cur_head_id]
                                        [i * config_.head_dim + j] +=
                                thread_local_cur_output_fp32_
                                    [thread_id][i * config_.head_dim + j];
                        }
                        attn_lse_[cur_batch_idx][cur_head_id][i] = new_attn_lse;
                    }
                    mutex_[cur_batch_idx][cur_head_id]->unlock();
                }
                thread_cur_head_idx_[thread_id].first = batch_id;
                thread_cur_head_idx_[thread_id].second = head_id;
                // printf("thread_cur_head_idx_[%d]: %d %d\n", thread_id,
                // batch_id,
                //        head_id);
                for (int i = 0; i < n_gqa_; i++) {
                    thread_local_cur_attn_lse_[thread_id][i] =
                        thread_local_attn_lse_[thread_id][i];
                    for (int j = 0; j < config_.head_dim; j++) {
                        thread_local_cur_output_fp32_
                            [thread_id][i * config_.head_dim + j] =
                                thread_local_output_fp32_[thread_id]
                                                         [i * config_.head_dim +
                                                          j];
                    }
                }
            }
        },
        // Merge the results of the remaining blocks.
        [&](int thread_id) {
            int cur_batch_idx = thread_cur_head_idx_[thread_id].first;
            int cur_head_id = thread_cur_head_idx_[thread_id].second;
            // printf("cur_batch_idx: %d, cur_head_id: %d, thread_id: %d\n",
            //        cur_batch_idx, cur_head_id, thread_id);
            if (cur_head_id != -1) {
                mutex_[cur_batch_idx][cur_head_id]->lock();
                for (int i = 0; i < n_gqa_; i++) {
                    float new_attn_lse;
                    if (std::abs(attn_lse_[cur_batch_idx][cur_head_id][i]) <
                        1e-6) {
                        attn_lse_[cur_batch_idx][cur_head_id][i] =
                            thread_local_cur_attn_lse_[thread_id][i];
                        for (int j = 0; j < config_.head_dim; j++) {
                            output_fp32_[cur_batch_idx][cur_head_id]
                                        [i * config_.head_dim + j] =
                                            thread_local_cur_output_fp32_
                                                [thread_id]
                                                [i * config_.head_dim + j];
                        }
                        // printf("cur_batch_idx: %d, cur_head_id: %d, i: %d,  "
                        //        "output_fp32_[cur_batch_idx][cur_head_id]:
                        //        %f\n", cur_batch_idx, cur_head_id, i,
                        //        output_fp32_[cur_batch_idx][cur_head_id]
                        //                    [i * config_.head_dim]);
                        continue;
                    }
                    new_attn_lse =
                        attn_lse_[cur_batch_idx][cur_head_id][i] +
                        std::log(
                            1.0 +
                            std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                     attn_lse_[cur_batch_idx][cur_head_id][i]));
                    // printf("new_attn_lse: %f %f %f\n", new_attn_lse,
                    //        thread_local_cur_attn_lse_[thread_id][i],
                    //        attn_lse_[cur_batch_idx][cur_head_id][i]);
                    ggml_vec_scale_f32(
                        config_.head_dim,
                        output_fp32_[cur_batch_idx][cur_head_id].data() +
                            i * config_.head_dim,
                        std::exp(attn_lse_[cur_batch_idx][cur_head_id][i] -
                                 new_attn_lse));
                    ggml_vec_scale_f32(
                        config_.head_dim,
                        thread_local_cur_output_fp32_[thread_id].data() +
                            i * config_.head_dim,
                        std::exp(thread_local_cur_attn_lse_[thread_id][i] -
                                 new_attn_lse));
                    for (int j = 0; j < config_.head_dim; j++) {
                        output_fp32_[cur_batch_idx][cur_head_id]
                                    [i * config_.head_dim + j] +=
                            thread_local_cur_output_fp32_[thread_id]
                                                         [i * config_.head_dim +
                                                          j];
                    }
                    attn_lse_[cur_batch_idx][cur_head_id][i] = new_attn_lse;
                }
                mutex_[cur_batch_idx][cur_head_id]->unlock();
            }
        });

    // move the results to output and attn_lse
    uint16_t *output_data = reinterpret_cast<uint16_t *>(output);
    float *attn_lse_data = attn_lse;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int i = 0; i < config_.kv_head_num; i++) {
            for (int j = 0; j < n_gqa_ * config_.head_dim; j++) {
                output_data[batch_idx * config_.kv_head_num * n_gqa_ *
                                config_.head_dim +
                            i * n_gqa_ * config_.head_dim + j] =
                    GGML_FP32_TO_FP16(output_fp32_[batch_idx][i][j]);
            }
            for (int j = 0; j < n_gqa_; j++) {
                attn_lse_data[batch_idx * config_.kv_head_num * n_gqa_ +
                              i * n_gqa_ + j] = attn_lse_[batch_idx][i][j];
            }
        }
    }

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // printf("layer %d time of computing attention: %f s\n", layer_idx,
    //        diff.count());
}

void KVCache::attn_with_kvcache(const ggml_fp16_t *q_in,
                                const ggml_fp16_t *k_in,
                                const ggml_fp16_t *v_in, ggml_fp16_t *output,
                                float *attn_lse, int layer_idx, int q_len,
                                int batch_size, int max_block_num,
                                int *block_table, int *cache_seqlens, int topk,
                                int local, Backend *backend) {
    assert(q_len == 1);
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    // printf("layer_idx: %d, q_len: %d, batch_size: %d, max_block_num: %d, "
    //        "topk: %d, local: %d\n",
    //        layer_idx, q_len, batch_size, max_block_num, topk, local);

    layer_id_ = layer_idx;
    int thread_num = backend->get_thread_num();
    k_data_ = const_cast<uint16_t *>(k_in);
    v_data_ = const_cast<uint16_t *>(v_in);
    // Each task updates the k cache and v cache of a certain header
    backend->do_work_stealing_job(
        batch_size * config_.kv_head_num, nullptr,
        [&](int task_id) {
            // printf("block_idx: %d, task_id: %d\n", block_idx, task_id);

            int batch_id = task_id / (config_.kv_head_num);
            int head_id = task_id % config_.kv_head_num;
            int seq_len = cache_seqlens[batch_id];

            int block_id = seq_len / config_.block_len;
            int block_idx = block_table[batch_id * max_block_num + block_id];
            int pos_in_block = seq_len % config_.block_len;

            if (config_.kv_type == ggml_type::GGML_TYPE_F16) {
                for (int l = 0; l < config_.head_dim; l++) {
                    k_cache_fp16_[layer_id_][head_id][block_idx]
                                 [pos_in_block * config_.head_dim + l] =
                                     k_data_[batch_id *
                                                 (q_len * config_.kv_head_num *
                                                  config_.head_dim) +
                                             head_id * config_.head_dim + l];
                    v_cache_fp16_[layer_id_][head_id][block_idx]
                                 [l * config_.block_len + pos_in_block] =
                                     v_data_[batch_id *
                                                 (q_len * config_.kv_head_num *
                                                  config_.head_dim) +
                                             head_id * config_.head_dim + l];
                }
            } else if (config_.kv_type == ggml_type::GGML_TYPE_Q4_0) {
                std::vector<float> block_fp32(32);
                // fill k_cache_
                for (int l = 0; l < config_.head_dim / 32; l++) {
                    block_q4_0 block;
                    for (int m = 0; m < 32; m++) {
                        // block_fp32[m] = k_in[batch_id][block_id *
                        // config_.block_len + k][head_id][l * 32 + m]

                        block_fp32[m] = GGML_FP16_TO_FP32(
                            k_data_[batch_id * (q_len * config_.kv_head_num *
                                                config_.head_dim) +
                                    head_id * config_.head_dim + l * 32 + m]);
                    }
                    quantize_row_q4_0(block_fp32.data(), &block, 32);
                    // printf("block_idx: %d, head_id: %d, pos_in_block: %d\n",
                    // block_idx,
                    //        head_id, pos_in_block);
                    k_cache_q4[layer_id_][head_id][block_idx]
                              [pos_in_block * config_.head_dim / 32 + l] =
                                  block;
                }

                // fill v_cache_
                for (int l = 0; l < config_.head_dim; l++) {
                    block_q4_0 block = v_cache_q4[layer_id_][head_id][block_idx]
                                                 [l * config_.block_len / 32 +
                                                  pos_in_block / 32];
                    dequantize_row_q4_0(&block, block_fp32.data(), 32);
                    block_fp32[pos_in_block % 32] = GGML_FP16_TO_FP32(
                        v_data_[batch_id * (q_len * config_.kv_head_num *
                                            config_.head_dim) +
                                head_id * config_.head_dim + l]);
                    quantize_row_q4_0(block_fp32.data(), &block, 32);
                    v_cache_q4[layer_id_][head_id][block_idx]
                              [l * config_.block_len / 32 + pos_in_block / 32] =
                                  block;
                }
            } else if (config_.kv_type == ggml_type::GGML_TYPE_Q8_0) {
                std::vector<float> block_fp32(32);
                // fill k_cache_
                for (int l = 0; l < config_.head_dim / 32; l++) {
                    block_q8_0 block;
                    for (int m = 0; m < 32; m++) {
                        // block_fp32[m] = k_in[batch_id][block_id *
                        // config_.block_len + k][head_id][l * 32 + m]

                        block_fp32[m] = GGML_FP16_TO_FP32(
                            k_data_[batch_id * (q_len * config_.kv_head_num *
                                                config_.head_dim) +
                                    head_id * config_.head_dim + l * 32 + m]);
                    }
                    quantize_row_q8_0(block_fp32.data(), &block, 32);
                    // printf("block_idx: %d, head_id: %d, pos_in_block: %d\n",
                    // block_idx,
                    //        head_id, pos_in_block);
                    k_cache_q8[layer_id_][head_id][block_idx]
                              [pos_in_block * config_.head_dim / 32 + l] =
                                  block;
                }

                // fill v_cache_
                for (int l = 0; l < config_.head_dim; l++) {
                    block_q8_0 block = v_cache_q8[layer_id_][head_id][block_idx]
                                                 [l * config_.block_len / 32 +
                                                  pos_in_block / 32];
                    dequantize_row_q8_0(&block, block_fp32.data(), 32);
                    block_fp32[pos_in_block % 32] = GGML_FP16_TO_FP32(
                        v_data_[batch_id * (q_len * config_.kv_head_num *
                                            config_.head_dim) +
                                head_id * config_.head_dim + l]);
                    quantize_row_q8_0(block_fp32.data(), &block, 32);
                    v_cache_q8[layer_id_][head_id][block_idx]
                              [l * config_.block_len / 32 + pos_in_block / 32] =
                                  block;
                }
            }
        },
        nullptr);

    // cache_seqlens memory is modified!
    for (int i = 0; i < batch_size; i++) {
        cache_seqlens[i] += q_len;
        // printf("cache_seqlens[%d]: %d\n", i, cache_seqlens[i]);
    }

    // printf("layer_idx: %d, q_len: %d, batch_size: %d, max_block_num: %d, "
    //        "topk: %d, local: %d\n",
    //        layer_idx, q_len, batch_size, max_block_num, topk, local);

    // printf("update finished.\n");
    int init_block_num = 1;
    if (config_.block_len <= 32) {
        init_block_num = 64 / config_.block_len;
    }
    attn(q_in, output, attn_lse, layer_idx, q_len, batch_size, max_block_num,
         block_table, cache_seqlens, topk, init_block_num, local, backend);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // printf("layer %d time of computing attention with kvcache: %f s\n",
    //        layer_idx, diff.count());
}

void attn_with_kvcache_one_block(
    int head_dim, int bsz,
    ggml_type q_type, // GGML data type of `Q`，只支持 fp16 和 q8_0
    // [bsz, head_dim]
    // 量化一定是 head_dim 这个维度，也就是 per_token，如果 head_dim % 32 !=
    // 0 报错 大小一定是 bsz * head_dim/32 * qtype_size
    const void *q,

    int past_kv_len, int past_kv_offset,
    bool is_full_attn, // true 表示全 1 mask
    // 如果 is_full_attn = false，则传入一个 bit 矩阵表示 mask
    // [bsz, past_kv_len]
    const uint8_t *attn_mask,

    ggml_type k_type, // GGML data type of `K Cache`，只支持 fp16 和 q4_0, q8_0
    int k_quant_type, // 0 per_token, 1 per_channel, 其它报错
    // [seq_len, head_dim]
    // 如果 quant_type == 0， 要求 head_dim % 32 == 0
    // 如果 quant_type == 1， 要求 seq_len % 32 == 0
    const void *k_cache,

    // k_anchor_type 必须为 fp16
    int num_k_anchor, // num_k_anchor == 0 表示没有 anchor
    // [num_k_anchor, head_dim]
    const void *k_cache_anchors,
    // 每个 token 都和前面最近的 pos 的 anchor 管，距离一样归前面管
    const int *k_cache_anchor_pos,

    // v cache 和上面一样
    ggml_type v_type, int v_quant_type,
    // [head_dim, seq_len]
    const void *v_cache, int num_v_anchor, const void *v_cache_anchors,
    const int *v_cache_anchor_pos,

    // [bsz, past_kv_len] 大小的中间预分配 buffer
    // 整个计算内部没有任何 malloc
    float *attn_score,

    // 输出： [bsz, head_dim]，类型和 q_type 保持一致
    void *output,
    // [bsz]
    float *lse,

    // 在外面开好的临时 buffer，(2 * bsz * past_kv_len + 6 * bsz *
    // head_dim + 2 * past_kv_len * head_dim + past_kv_len * head_dim /
    // 32)个字节是足够的大小
    void *draft,

    // 在线加 rope
    const int *rotary_angle, const void *rotary_cos, const void *rotary_sin
    // rotary_cos=None,
    // rotary_sin=None,
    // cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    // cache_batch_idx: Optional[torch.Tensor] = None,
    // rotary_interleaved=True,

    // // 暂时不用支持
    // window_size=(-1, -1),  # -1 means infinite context window
    // alibi_slopes=None,
) {
    assert(head_dim % 32 == 0);
    assert(k_quant_type == 0);
    assert(v_quant_type == 1);
    assert(q_type == GGML_TYPE_F16 || q_type == GGML_TYPE_Q8_0);
    if (q_type == GGML_TYPE_F16) {
        assert(k_type == GGML_TYPE_F16);
        assert(v_type == GGML_TYPE_F16);

        // attn = q * k + q * k_anchor
        // TODO: anchor
        assert(num_k_anchor == 0);

        if (rotary_angle != nullptr) {
            ggml_fp16_t *k_cache_with_rope_fp16 =
                (reinterpret_cast<ggml_fp16_t *>(draft) +
                 sizeof(block_q8_0) * bsz * past_kv_len / QK8_0 +
                 sizeof(float) * bsz * head_dim);
            // dequant k_cache and apply rope
            // k_rope(i) = k(i) * cos(i) - k(i+l) * sin(i)
            // k_rope(i+l) = k(i+l) * cos(i+l) + k(i) * sin(i)

            // k(i)cos(i) -> k_rope(i)
            // k(i)sin(i+l) -> k_rope(i+l)

            // k(i)cos(i) -> k_rope(i)
            // -k(i)sin(i-l) -> k_rope(i-l)

            std::vector<float> block_fp32(32);
            for (int k = 0; k < past_kv_len; k++) {
                int angle = rotary_angle[k];
                for (int l = 0; l < head_dim / 32; l++) {
                    for (int m = 0; m < 32; m++) {
                        float x = GGML_FP16_TO_FP32((
                            (ggml_fp16_t *)k_cache)[k * head_dim + l * 32 + m]);
                        float sin_val = GGML_FP16_TO_FP32(
                            ((ggml_fp16_t *)
                                 rotary_sin)[angle * head_dim + l * 32 + m]);
                        float cos_val = GGML_FP16_TO_FP32(
                            ((ggml_fp16_t *)
                                 rotary_cos)[angle * head_dim + l * 32 + m]);

                        if (l * 32 + m < head_dim / 2) {
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m] =
                                GGML_FP32_TO_FP16(x * cos_val);
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m +
                                                   head_dim / 2] =
                                GGML_FP32_TO_FP16(-x * sin_val);
                        } else {
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m] =
                                GGML_FP32_TO_FP16(
                                    GGML_FP16_TO_FP32(
                                        k_cache_with_rope_fp16[k * head_dim +
                                                               l * 32 + m]) +
                                    x * sin_val);
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m -
                                                   head_dim / 2] =
                                GGML_FP32_TO_FP16(
                                    GGML_FP16_TO_FP32(
                                        k_cache_with_rope_fp16[k * head_dim +
                                                               l * 32 + m -
                                                               head_dim / 2]) -
                                    x * cos_val);
                        }
                    }
                }
            }

            llamafile_sgemm(past_kv_len, bsz, head_dim,
                            (ggml_fp16_t *)k_cache_with_rope_fp16, head_dim,
                            (ggml_fp16_t *)q, head_dim, attn_score, past_kv_len,
                            0, 1, GGML_TASK_TYPE_COMPUTE, k_type, GGML_TYPE_F16,
                            GGML_TYPE_F32, GGML_PREC_DEFAULT);
        } else {
            bool ok = llamafile_sgemm(
                past_kv_len, bsz, head_dim, (ggml_fp16_t *)k_cache, head_dim,
                (ggml_fp16_t *)q, head_dim, attn_score, past_kv_len, 0, 1,
                GGML_TASK_TYPE_COMPUTE, k_type, GGML_TYPE_F16, GGML_TYPE_F32,
                GGML_PREC_DEFAULT);

            // float attn_score_diff[4] = {0, 0, 0, 0};
            // for (int k = 0; k < head_dim; k++) {
            //     for (int i = 0; i < bsz; i++) {
            //         attn_score_diff[i] +=
            //             ggml_fp16_to_fp32(
            //                 ((ggml_fp16_t *)q)[i * head_dim + k]) *
            //             ggml_fp16_to_fp32(((ggml_fp16_t *)k_cache)[k]);
            //     }
            // }
            // for (int i = 0; i < 4; i++) {
            //     printf("attn_score[%d]: %f, attn_score_diff[%d]: %f\n", i,
            //            attn_score[i * past_kv_len], i, attn_score_diff[i]);
            // }
            if (!ok) {
                printf("llamafile_sgemm failed\n");
            }

            // printf("x = %d\n", x);
            // for (int i = 0; i < 16; i++) {
            //     for (int j = 0; j < 16; j++) {
            //         printf("k_cache[%d][%d]: %f\n", i, j,
            //                GGML_FP16_TO_FP32(
            //                    ((ggml_fp16_t *)k_cache)[i * head_dim + j]));
            //     }
            // }
            // for (int i = 0; i < 1; i++) {
            //     for (int j = 0; j < 16; j++) {
            //         printf("q[%d][%d]: %f\n", i, j,
            //                GGML_FP16_TO_FP32(
            //                    ((ggml_fp16_t *)q)[i * head_dim + j]));
            //     }
            // }
            // for (int i = 0; i < 1; i++) {
            //     for (int j = 0; j < 16; j++) {
            //         printf("attn_score[%d][%d]: %f\n", i, j,
            //                attn_score[i * past_kv_len + j]);
            //     }
            // }
        }
        // attn = attn * scale
        float scale_factor = 1.0 / std::sqrt(float(head_dim));
        ggml_vec_scale_f32(bsz * past_kv_len, attn_score, scale_factor);

        // attn = attn & mask
        if (!is_full_attn) {
            for (int i = 0; i < bsz; i++) {
                for (int j = 0; j < past_kv_len; j++) {
                    int index = i * past_kv_len + j;
                    if (!(attn_mask[j / 8] & (1 << (j % 8)))) {
                        attn_score[index] =
                            std::numeric_limits<float>::lowest();
                    }
                }
            }
        }

        // attn = softmax(attn)
        for (int i = 0; i < bsz; i++) {
            float sum_exp = 0;
            for (int j = 0; j < past_kv_len; j++) {
                attn_score[i * past_kv_len + j] =
                    std::exp(attn_score[i * past_kv_len + j]);
                sum_exp += attn_score[i * past_kv_len + j];
            }
            for (int j = 0; j < past_kv_len; j++) {
                attn_score[i * past_kv_len + j] /= sum_exp;
                // printf("attn_score[%d %d: %d]: %f\n", i, j, i * past_kv_len +
                // j,
                //        attn_score[i * past_kv_len + j]);
            }
            if (lse != nullptr) {
                lse[i] = std::log(sum_exp);
            }
        }

        // output = attn * v + attn * v_anchor
        // std::vector<float> sum(bsz * head_dim);
        float *sum = reinterpret_cast<float *>(reinterpret_cast<char *>(draft) +
                                               sizeof(block_q8_0) * bsz *
                                                   past_kv_len / QK8_0);

        // float* attn_score_fp16(bsz, past_kv_len)
        ggml_fp16_t *attn_score_fp16 = (reinterpret_cast<ggml_fp16_t *>(
            reinterpret_cast<char *>(draft) +
            sizeof(block_q8_0) * bsz * past_kv_len / QK8_0 +
            sizeof(float) * bsz * head_dim));

        for (int i = 0; i < bsz * past_kv_len; i++) {
            attn_score_fp16[i] = GGML_FP32_TO_FP16(attn_score[i]);
        }

        // TODO: anchor
        assert(num_v_anchor == 0);
        bool ok = llamafile_sgemm(
            head_dim, bsz, past_kv_len, (ggml_fp16_t *)v_cache, past_kv_len,
            (ggml_fp16_t *)attn_score_fp16, past_kv_len, sum, head_dim, 0, 1,
            GGML_TASK_TYPE_COMPUTE, v_type, GGML_TYPE_F16, GGML_TYPE_F32,
            GGML_PREC_DEFAULT);
        if (!ok) {
            printf("llamafile_sgemm failed\n");
        }
        // float sum_diff[4] = {0, 0, 0, 0};
        // for (int k = 0; k < past_kv_len; k++) {
        //     for (int i = 0; i < bsz; i++) {
        //         sum_diff[i] += ggml_fp16_to_fp32(((ggml_fp16_t *)v_cache)[k])
        //         *
        //                        attn_score[i * past_kv_len + k];
        //     }
        // }
        // for (int i = 0; i < 4; i++) {
        //     printf("sum[%d]: %f, sum_diff[%d]: %f\n", i, sum[i * head_dim],
        //     i,
        //            sum_diff[i]);
        // }

        // copy to output
        for (int i = 0; i < bsz; i++) {
            for (int j = 0; j < head_dim; j++) {
                ((float *)output)[i * head_dim + j] = sum[i * head_dim + j];
            }
        }
    } else {
        assert(k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q8_0);
        assert(v_type == GGML_TYPE_Q4_0 || v_type == GGML_TYPE_Q8_0);

        // attn = q * k + q * k_anchor
        // TODO: anchor
        assert(num_k_anchor == 0);

        if (rotary_angle != nullptr) {
            ggml_fp16_t *k_cache_with_rope_fp16 =
                (reinterpret_cast<ggml_fp16_t *>(draft) +
                 sizeof(block_q8_0) * bsz * past_kv_len / QK8_0 +
                 sizeof(float) * bsz * head_dim);
            block_q4_0 *k_cache_with_rope_q4 =
                (reinterpret_cast<block_q4_0 *>(draft) +
                 sizeof(block_q8_0) * bsz * past_kv_len / QK8_0 +
                 sizeof(float) * bsz * head_dim) +
                sizeof(ggml_fp16_t) * bsz * head_dim;
            // dequant k_cache and apply rope
            // k_rope(i) = k(i) * cos(i) - k(i+l) * sin(i)
            // k_rope(i+l) = k(i+l) * cos(i+l) + k(i) * sin(i)

            // k(i)cos(i) -> k_rope(i)
            // k(i)sin(i+l) -> k_rope(i+l)

            // k(i)cos(i) -> k_rope(i)
            // -k(i)sin(i-l) -> k_rope(i-l)

            std::vector<float> block_fp32(32);
            for (int k = 0; k < past_kv_len; k++) {
                int angle = rotary_angle[k];
                for (int l = 0; l < head_dim / 32; l++) {
                    block_q4_0 block =
                        ((block_q4_0 *)k_cache)[k * head_dim / 32 + l];
                    dequantize_row_q4_0(&block, block_fp32.data(), 32);
                    for (int m = 0; m < 32; m++) {
                        float sin_val = GGML_FP16_TO_FP32(
                            ((ggml_fp16_t *)
                                 rotary_sin)[angle * head_dim + l * 32 + m]);
                        float cos_val = GGML_FP16_TO_FP32(
                            ((ggml_fp16_t *)
                                 rotary_cos)[angle * head_dim + l * 32 + m]);

                        if (l * 32 + m < head_dim / 2) {
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m] =
                                GGML_FP32_TO_FP16(block_fp32[m] * cos_val);
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m +
                                                   head_dim / 2] =
                                GGML_FP32_TO_FP16(-block_fp32[m] * sin_val);
                        } else {
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m] +=
                                GGML_FP32_TO_FP16(block_fp32[m] * sin_val);
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m -
                                                   head_dim / 2] -=
                                GGML_FP32_TO_FP16(block_fp32[m] * cos_val);
                        }
                    }
                }
            }
            // quantize k_cache_with_rope_fp16
            for (int k = 0; k < past_kv_len; k++) {
                for (int l = 0; l < head_dim / 32; l++) {
                    for (int m = 0; m < 32; m++) {
                        block_fp32[m] = GGML_FP16_TO_FP32(
                            k_cache_with_rope_fp16[k * head_dim + l * 32 + m]);
                    }
                    quantize_row_q4_0(
                        block_fp32.data(),
                        &k_cache_with_rope_q4[k * head_dim / 32 + l], 32);
                }
            }

            llamafile_sgemm(past_kv_len, bsz, head_dim / 32,
                            (block_q4_0 *)k_cache_with_rope_q4, head_dim / 32,
                            (block_q8_0 *)q, head_dim / 32, attn_score,
                            past_kv_len, 0, 1, GGML_TASK_TYPE_COMPUTE, k_type,
                            GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        } else {
            llamafile_sgemm(past_kv_len, bsz, head_dim / 32,
                            (block_q4_0 *)k_cache, head_dim / 32,
                            (block_q8_0 *)q, head_dim / 32, attn_score,
                            past_kv_len, 0, 1, GGML_TASK_TYPE_COMPUTE, k_type,
                            GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        }

        // attn = attn * scale
        float scale_factor = 1.0 / std::sqrt(float(head_dim));
        ggml_vec_scale_f32(bsz * past_kv_len, attn_score, scale_factor);

        // attn = attn & mask
        if (!is_full_attn) {
            for (int i = 0; i < bsz; i++) {
                for (int j = 0; j < past_kv_len; j++) {
                    int index = i * past_kv_len + j;
                    if (!(attn_mask[j / 8] & (1 << (j % 8)))) {
                        attn_score[index] =
                            std::numeric_limits<float>::lowest();
                    }
                }
            }
        }

        // attn = softmax(attn)
        for (int i = 0; i < bsz; i++) {
            float sum_exp = 0;
            for (int j = 0; j < past_kv_len; j++) {
                attn_score[i * past_kv_len + j] =
                    std::exp(attn_score[i * past_kv_len + j]);
                sum_exp += attn_score[i * past_kv_len + j];
            }
            for (int j = 0; j < past_kv_len; j++) {
                attn_score[i * past_kv_len + j] /= sum_exp;
                // printf("attn_score[%d %d: %d]: %f\n", i, j, i * past_kv_len +
                // j,
                //        attn_score[i * past_kv_len + j]);
            }
            if (lse != nullptr) {
                lse[i] = std::log(sum_exp);
            }
        }

        // output = attn * v + attn * v_anchor
        // std::vector<block_q8_0> attn_q8_0(bsz * past_kv_len / QK8_0);
        block_q8_0 *attn_q8_0 = reinterpret_cast<block_q8_0 *>(draft);
        quantize_row_q8_0(attn_score, attn_q8_0, bsz * past_kv_len);
        // std::vector<float> sum(bsz * head_dim);
        float *sum = reinterpret_cast<float *>(reinterpret_cast<char *>(draft) +
                                               sizeof(block_q8_0) * bsz *
                                                   past_kv_len / QK8_0);
        // TODO: anchor
        assert(num_v_anchor == 0);
        llamafile_sgemm(head_dim, bsz, past_kv_len / 32, (block_q4_0 *)v_cache,
                        past_kv_len / 32, attn_q8_0, past_kv_len / 32, sum,
                        head_dim, 0, 1, GGML_TASK_TYPE_COMPUTE, v_type,
                        GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_PREC_DEFAULT);

        quantize_row_q8_0(sum, (block_q8_0 *)output, bsz * head_dim);
    }
}