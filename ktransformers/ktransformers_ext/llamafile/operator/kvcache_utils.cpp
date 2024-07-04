#include "kvcache.h"

KVCacheConfig::KVCacheConfig(int layer_num, int kv_head_num, int q_head_num, int head_dim, int block_len, int anchor_num, AnchorType anchor_type)
    : layer_num(layer_num), kv_head_num(kv_head_num), q_head_num(q_head_num), head_dim(head_dim), block_len(block_len), anchor_num(anchor_num), anchor_type(anchor_type) {
    assert(q_head_num % kv_head_num == 0 && head_dim % CHUNK_SIZE == 0 &&
           block_len % CHUNK_SIZE == 0);
}
KVCache::KVCache(KVCacheConfig config) {
    this->config_ = config;
    n_gqa_ = config_.q_head_num / config_.kv_head_num;
    k_cache_.resize(config.layer_num);
    v_cache_.resize(config.layer_num);
    anchor_.resize(config.layer_num);
    importance_.resize(config.layer_num);
    past_block_num_.resize(config.layer_num);
    for (int i = 0; i < config.layer_num; i++) {
        k_cache_[i].resize(config.kv_head_num);
        v_cache_[i].resize(config.kv_head_num);
        anchor_[i].resize(config.kv_head_num);
        past_block_num_[i] = 0;
    }

    q_fp32.resize(n_gqa_ * config.head_dim);
}

void KVCache::ThreadResize(int thread_num) {
    thread_local_output_q8_0_.resize(thread_num);
    thread_local_attn_score_.resize(thread_num);
    thread_local_output_fp32_.resize(thread_num);
    thread_local_attn_lse_.resize(thread_num);
    thread_local_cur_output_fp32_.resize(thread_num);
    thread_local_cur_attn_lse_.resize(thread_num);
    thread_local_draft_.resize(thread_num);
    thread_cur_head_idx_.resize(thread_num);
    thread_local_attn_mask_.resize(thread_num);
    for (int i = 0; i < thread_num; i++) {
        thread_local_output_q8_0_[i].resize(n_gqa_ * config_.head_dim / QK8_0);
        thread_local_attn_score_[i].resize(n_gqa_ * config_.block_len);
        thread_local_output_fp32_[i].resize(n_gqa_ * config_.head_dim);
        thread_local_attn_lse_[i].resize(n_gqa_);
        thread_local_cur_output_fp32_[i].resize(n_gqa_ * config_.head_dim);
        thread_local_cur_attn_lse_[i].resize(n_gqa_);
        thread_local_draft_[i].resize(2 * n_gqa_ * config_.block_len +
                                      6 * n_gqa_ * config_.head_dim);
        thread_local_attn_mask_[i].resize(config_.block_len / 8);
    }
}
void KVCache::BatchResize(int batch_size) {
    mutex_.resize(batch_size);
    q_q8_0_.resize(batch_size);
    output_fp32_.resize(batch_size);
    attn_lse_.resize(batch_size);
    block_table_before_retrieval_.resize(batch_size);
    block_table_after_retrieval_.resize(batch_size);
    cache_seqlens_.resize(batch_size);

    for (int i = 0; i < batch_size; i++) {
        top_similar_block_.resize(batch_size);
        mutex_[i].resize(config_.kv_head_num);
        q_q8_0_[i].resize(config_.kv_head_num);
        output_fp32_[i].resize(config_.kv_head_num);
        attn_lse_[i].resize(config_.kv_head_num);

        for (int j = 0; j < config_.kv_head_num; j++) {
            if (!mutex_[i][j]) {
                mutex_[i][j] = std::make_unique<std::mutex>();
            }
            q_q8_0_[i][j].resize(n_gqa_ * config_.head_dim / QK8_0);
            output_fp32_[i][j].resize(n_gqa_ * config_.head_dim);
            attn_lse_[i][j].resize(n_gqa_);
        }
    }
}

void KVCache::calc_anchor_all_layers(int* block_table, int* cache_seqlens, int batch_size, int max_block_num, Backend* backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    // Each task updates the importance of a certain block
    seq_len_ = config_.block_len;
    backend->do_work_stealing_job(
        config_.layer_num * batch_size * max_block_num, [&](int task_id) {
            int layer_id = task_id / (batch_size * max_block_num);
            int batch_id =
                (task_id % (batch_size * max_block_num)) / max_block_num;
            int block_id = task_id % max_block_num;
            // If the block is out of the sequence length, skip it. In
            // particular, the last block of the sequence that is shorter than
            // the block length should be skipped.

            if (cache_seqlens[batch_id] / config_.block_len <= block_id) {
                return;
            }
            int block_idx = block_table[batch_id * max_block_num + block_id];

            for (int head_id = 0; head_id < config_.kv_head_num; head_id++) {
                anchor_[layer_id][head_id][block_idx].resize(
                    config_.anchor_num * config_.head_dim);
            }

            std::vector<float> block_fp32(32);
            if (config_.anchor_type == AnchorType::DYNAMIC) {
                // find top anchor_num importances and their corresponding
                // positions in the importance_ tensor
                // TODO: Move top_importances to the class member to avoid
                // repeated memory allocation
                std::priority_queue<std::pair<float, int>> top_importances;
                for (int k = 0; k < seq_len_; k++) {
                    top_importances.push(
                        std::make_pair(importance_[layer_id][block_idx][k], k));
                    if (top_importances.size() > config_.anchor_num) {
                        top_importances.pop();
                    }
                }

                // fill anchor_
                for (int k = 0; k < config_.anchor_num; k++) {
                    int top_indice = top_importances.top().second;
                    for (int head_id = 0; head_id < config_.kv_head_num;
                         head_id++) {
                        for (int l = 0; l < config_.head_dim / 32; l++) {
                            block_q4_0 block =
                                k_cache_[layer_id][head_id][block_idx]
                                        [top_indice * config_.head_dim / 32 +
                                         l];
                            dequantize_row_q4_0(&block, block_fp32.data(), 32);
                            for (int m = 0; m < 32; m++) {
                                anchor_[layer_id][head_id][block_idx]
                                       [k * config_.head_dim + l * 32 + m] =
                                           GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                        top_importances.pop();
                    }
                }
            } else if (config_.anchor_type == AnchorType::FIXED) {
                // fill anchor_
                for (int k = 0; k < config_.anchor_num; k++) {
                    int fixed_stride_indice = k * seq_len_ / config_.anchor_num;
                    for (int head_id = 0; head_id < config_.kv_head_num;
                         head_id++) {
                        for (int l = 0; l < config_.head_dim / 32; l++) {
                            block_q4_0 block =
                                k_cache_[layer_id][head_id][block_idx]
                                        [fixed_stride_indice *
                                             config_.head_dim / 32 +
                                         l];
                            dequantize_row_q4_0(&block, block_fp32.data(), 32);

                            for (int m = 0; m < 32; m++) {
                                anchor_[layer_id][head_id][block_idx]
                                       [k * config_.head_dim + l * 32 + m] =
                                           GGML_FP32_TO_FP16(block_fp32[m]);
                            }
                        }
                    }
                }
            } else {
                assert(false);
            }
        });

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("time of calc_anchor_all_layers: %f s\n", duration.count());
}