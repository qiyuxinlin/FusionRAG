#include "kvcache.h"

void KVCache::get_anchor_one_block(ggml_fp16_t *anchor, int layer_id,
                                   int block_idx, Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    anchor_data_ = const_cast<uint16_t *>(anchor);

    // printf("layer_id: %d, block_idx: %d\n", layer_id, block_idx);
    // Each task updates the anchor of a certain position
    backend->do_work_stealing_job(
        config_.kv_head_num * config_.anchor_num, [&](int task_id) {
            int k = task_id % config_.anchor_num;
            int head_id = task_id / config_.anchor_num;

            memcpy(anchor_data_ + k * config_.head_dim,
                   anchor_[layer_id_][head_id][block_idx].data() +
                       k * config_.head_dim,
                   sizeof(uint16_t) * config_.head_dim);
        });

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of reading anchor: %f s\n", layer_id,
           block_idx, duration.count());
}

void KVCache::update_anchor_one_block(const ggml_fp16_t *anchor, int layer_id,
                                      int block_idx, Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    anchor_data_ = const_cast<uint16_t *>(anchor);

    // Each task updates the anchor of a certain position
    backend->do_work_stealing_job(config_.anchor_num, [&](int task_id) {
        int k = task_id % config_.anchor_num;
        int head_id = task_id / config_.anchor_num;
        memcpy(anchor_[layer_id_][head_id][block_idx].data() +
                   k * config_.head_dim,
               anchor_data_ + k * config_.head_dim,
               sizeof(uint16_t) * config_.head_dim);
    });

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of writting anchor: %f s\n", layer_id,
           block_idx, duration.count());
}

void KVCache::update_importance_one_block(const ggml_fp16_t *importance,
                                          int layer_id, int block_idx,
                                          Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    importance_data_ = const_cast<uint16_t *>(importance);

    // Each task updates the importance of a certain position
    backend->do_work_stealing_job(config_.block_len, [&](int task_id) {
        int k = task_id;
        memcpy(importance_[layer_id_][block_idx].data() + k,
               importance_data_ + k, sizeof(uint16_t));
    });

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of writting importance: %f s\n", layer_id,
           block_idx, duration.count());
}

void KVCache::get_importance_one_block(ggml_fp16_t *importance, int layer_id,
                                       int block_idx, Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    importance_data_ = const_cast<uint16_t *>(importance);

    // Each task updates the importance of a certain position
    backend->do_work_stealing_job(config_.block_len, [&](int task_id) {
        int k = task_id;
        memcpy(importance_data_ + k,
               importance_[layer_id_][block_idx].data() + k, sizeof(uint16_t));
    });

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of reading importance: %f s\n", layer_id,
           block_idx, duration.count());
}

void KVCache::update_one_block_fp16(const ggml_fp16_t *k_in,
                                    const ggml_fp16_t *v_in, int layer_id,
                                    int block_idx, Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    block_idx = block_idx;
    seq_len_ = config_.block_len;
    k_data_ = const_cast<uint16_t *>(k_in);
    v_data_ = const_cast<uint16_t *>(v_in);

    int new_block_num = std::max((int)past_block_num_[layer_id], block_idx + 1);

    importance_[layer_id_].resize(new_block_num);

    for (int i = 0; i < config_.kv_head_num; i++) {
        k_cache_[layer_id][i].resize(new_block_num);
        v_cache_[layer_id][i].resize(new_block_num);
        anchor_[layer_id][i].resize(new_block_num);
    }

    for (int i = 0; i < new_block_num; i++) {
        importance_[layer_id][i].resize(config_.block_len);
    }

    // Each task updates the k cache or v cache of a certain header
    backend->do_work_stealing_job(config_.kv_head_num * 2, [&](int task_id) {
        // printf("block_idx: %d, task_id: %d\n", block_idx, task_id);
        std::vector<float> block_fp32(32);
        int head_id = task_id / 2;
        if (task_id & 1) {
            for (int k = 0; k < config_.anchor_num; k++) {
                anchor_[layer_id_][head_id][block_idx].resize(
                    config_.anchor_num * config_.head_dim);
            }
            // fill k_cache_
            k_cache_[layer_id_][head_id][block_idx].resize(
                config_.block_len * config_.head_dim / 32);
            for (int k = 0; k < config_.block_len; k++) {
                for (int l = 0; l < config_.head_dim / 32; l++) {
                    block_q4_0 block;
                    for (int m = 0; m < 32; m++) {
                        // block_fp32[m] = k_in[0][head_id][block_id *
                        // config_.block_len + k][l * 32 +
                        // m].item().to<float>();
                        block_fp32[m] = GGML_FP16_TO_FP32(
                            k_data_[((0 * config_.kv_head_num + head_id) *
                                         seq_len_ +
                                     0 * config_.block_len + k) *
                                        config_.head_dim +
                                    l * 32 + m]);
                    }
                    quantize_row_q4_0(block_fp32.data(), &block, 32);
                    k_cache_[layer_id_][head_id][block_idx]
                            [k * config_.head_dim / 32 + l] = block;
                }
            }
        } else {
            // fill v_cache_
            v_cache_[layer_id_][head_id][block_idx].resize(
                config_.head_dim * config_.block_len / 32);
            for (int k = 0; k < config_.block_len / 32; k++) {
                for (int l = 0; l < config_.head_dim; l++) {
                    block_q4_0 block;
                    for (int m = 0; m < 32; m++) {
                        // block_fp32[m] = v_in[0][head_id][block_id *
                        // config_.block_len + k * 32 +
                        // m][l].item().to<float>();
                        block_fp32[m] = GGML_FP16_TO_FP32(
                            v_data_[((0 * config_.kv_head_num + head_id) *
                                         seq_len_ +
                                     0 * config_.block_len + k * 32 + m) *
                                        config_.head_dim +
                                    l]);
                    }
                    quantize_row_q4_0(block_fp32.data(), &block, 32);
                    v_cache_[layer_id_][head_id][block_idx]
                            [l * config_.block_len / 32 + k] = block;
                }
            }
        }
    });
    past_block_num_[layer_id] = new_block_num;

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of writting KV Cache: %f s\n", layer_id,
           block_idx, duration.count());
    // printf("get_one_block_fp16 duration: %ld\n", duration);
}

void KVCache::get_one_block_fp16(ggml_fp16_t *k_in, ggml_fp16_t *v_in,
                                 int layer_id, int block_idx,
                                 Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    seq_len_ = config_.block_len;
    k_data_ = reinterpret_cast<uint16_t *>(k_in);
    v_data_ = reinterpret_cast<uint16_t *>(v_in);

    // printf("layer_id: %d, block_idx: %d\n", layer_id, block_idx);
    // Each task gets the k cache or v cache of a certain header
    backend->do_work_stealing_job(config_.kv_head_num * 2, [&](int task_id) {
        std::vector<float> block_fp32(32);
        int head_id = task_id / 2;
        if (task_id & 1) {
            // get k_cache_
            for (int k = 0; k < config_.block_len; k++) {
                for (int l = 0; l < config_.head_dim / 32; l++) {
                    block_q4_0 block = k_cache_[layer_id_][head_id][block_idx]
                                               [k * config_.head_dim / 32 + l];
                    dequantize_row_q4_0(&block, block_fp32.data(), 32);
                    for (int m = 0; m < 32; m++) {
                        // k_in[0][head_id][block_id * config_.block_len + k][l
                        // * 32
                        // + m] = block_fp32[m];
                        k_data_[((0 * config_.kv_head_num + head_id) *
                                     seq_len_ +
                                 0 * config_.block_len + k) *
                                    config_.head_dim +
                                l * 32 + m] = GGML_FP32_TO_FP16(block_fp32[m]);
                    }
                }
            }
        } else {
            // get v_cache_
            for (int k = 0; k < config_.block_len / 32; k++) {
                for (int l = 0; l < config_.head_dim; l++) {
                    block_q4_0 block = v_cache_[layer_id_][head_id][block_idx]
                                               [l * config_.block_len / 32 + k];
                    dequantize_row_q4_0(&block, block_fp32.data(), 32);
                    for (int m = 0; m < 32; m++) {
                        // v_in[0][head_id][block_id * config_.block_len + k *
                        // 32 + m][l] = block_fp32[m];
                        v_data_[((0 * config_.kv_head_num + head_id) *
                                     seq_len_ +
                                 0 * config_.block_len + k * 32 + m) *
                                    config_.head_dim +
                                l] = GGML_FP32_TO_FP16(block_fp32[m]);
                    }
                }
            }
        }
    });

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block %d time of reading KV Cache: %f s\n", layer_id,
           block_idx, duration.count());
    // printf("get_one_block_fp16 duration: %ld\n", duration);
}

void KVCache::get_all_kv_one_layer(int layer_id, ggml_fp16_t *k_in,
                                   ggml_fp16_t *v_in, Backend *backend) {

    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    layer_id_ = layer_id;
    seq_len_ = config_.block_len;
    block_num_ = get_cache_total_block_num();
    k_data_ = reinterpret_cast<uint16_t *>(k_in);
    v_data_ = reinterpret_cast<uint16_t *>(v_in);

    // printf("layer_id: %d, block_idx: %d\n", layer_id, block_idx);
    // Each task gets the k cache or v cache of a certain header
    backend->do_work_stealing_job(
        config_.kv_head_num * past_block_num_[layer_id] * 2, [&](int task_id) {
            std::vector<float> block_fp32(32);
            int head_id = task_id / 2 / past_block_num_[layer_id];
            int block_idx = task_id / 2 % past_block_num_[layer_id];
            if (block_idx >= block_num_)
                return;
            // printf("layer_id: %d, head_id: %d, block_idx: %d, task_id: %d\n",
            //        layer_id, head_id, block_idx, task_id);
            int max_offset = 0;
            if (task_id & 1) {
                // get k_cache_
                for (int k = 0; k < config_.block_len; k++) {
                    if (block_idx * seq_len_ + k >= cache_total_len_)
                        break;
                    for (int l = 0; l < config_.head_dim / 32; l++) {
                        block_q4_0 block =
                            k_cache_[layer_id_][head_id][block_idx]
                                    [k * config_.head_dim / 32 + l];
                        dequantize_row_q4_0(&block, block_fp32.data(), 32);
                        for (int m = 0; m < 32; m++) {
                            // k_in[0][head_id][block_id * config_.block_len +
                            // k][l
                            // * 32
                            // + m] = block_fp32[m];
                            // printf("pos: %d, dim_idx: %d, val: %f\n",
                            //        block_idx * seq_len_ + k, m,
                            //        block_fp32[m]);
                            k_data_[(head_id * cache_total_len_ +
                                     block_idx * config_.block_len + k) *
                                        config_.head_dim +
                                    l * 32 + m] =
                                GGML_FP32_TO_FP16(block_fp32[m]);
                            max_offset = std::max(
                                max_offset,
                                (int)(head_id * cache_total_len_ +
                                      block_idx * config_.block_len + k) *
                                        config_.head_dim +
                                    l * 32 + m);
                        }
                    }
                }
            } else {
                // get v_cache_
                for (int k = 0; k < config_.block_len / 32; k++) {
                    for (int l = 0; l < config_.head_dim; l++) {
                        block_q4_0 block =
                            v_cache_[layer_id_][head_id][block_idx]
                                    [l * config_.block_len / 32 + k];
                        dequantize_row_q4_0(&block, block_fp32.data(), 32);
                        for (int m = 0; m < 32; m++) {
                            // v_in[0][head_id][block_id * config_.block_len + k
                            // * 32 + m][l] = block_fp32[m];
                            if (block_idx * seq_len_ + k * 32 + m >=
                                cache_total_len_)
                                break;
                            v_data_[(head_id * cache_total_len_ +
                                     block_idx * config_.block_len + k * 32 +
                                     m) *
                                        config_.head_dim +
                                    l] = GGML_FP32_TO_FP16(block_fp32[m]);
                            max_offset =
                                std::max(max_offset,
                                         (int)((head_id * cache_total_len_ +
                                                block_idx * config_.block_len +
                                                k * 32 + m) *
                                                   config_.head_dim +
                                               l));
                        }
                    }
                }
            }
            // printf("max_offset: %d\n", max_offset);
            // printf("layer_id: %d, head_id: %d, block_idx: %d, task_id: %d\n",
            //        layer_id, head_id, block_idx, task_id);
        });

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("layer %d block num %d time of reading all KV Cache: %f s\n",
           layer_id, block_num_, duration.count());
    // printf("get_one_block_fp16 duration: %ld\n", duration);
}