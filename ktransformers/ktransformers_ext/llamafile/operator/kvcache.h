#ifndef CPUINFER_OPERATOR_KVCACHE_H
#define CPUINFER_OPERATOR_KVCACHE_H

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../backend.h"
#include "llama.cpp/ggml-common.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llamafile/sgemm.h"

#define CHUNK_SIZE 32
enum AnchorType { FIXED,
                  DYNAMIC };

struct KVCacheConfig {
    int layer_num;
    int kv_head_num;
    int q_head_num;
    int head_dim;
    int block_len;
    int anchor_num;
    AnchorType anchor_type;

    KVCacheConfig() = default;
    KVCacheConfig(int layer_num, int kv_head_num, int q_head_num, int head_dim, int block_len, int anchor_num, AnchorType anchor_type);
};

// TODO: 把resize都提出来，单独写一个方法预分配空间， 替换动态resize
class KVCache {
   public:
    KVCache(KVCacheConfig);
    void ThreadResize(int thread_num);
    void BatchResize(int batch_size);
    int get_layer_num() { return config_.layer_num; }
    int get_kv_head_num() { return config_.kv_head_num; }
    int get_q_head_num() { return config_.q_head_num; }
    int get_head_dim() { return config_.head_dim; }
    int get_block_len() { return config_.block_len; }
    int get_block_num(int layer_id) { return past_block_num_[layer_id]; }
    int get_anchor_num() { return config_.anchor_num; }
    int get_cache_total_len() { return cache_total_len_; }
    int get_cache_total_block_num() {
        return (cache_total_len_ + config_.block_len - 1) / config_.block_len;
    }
    void update_cache_total_len(int cache_total_len) {
        cache_total_len_ = cache_total_len;
    }
    void attn(const ggml_fp16_t* q_in, ggml_fp16_t* output, float* attn_lse, int layer_idx, int q_len, int batch_size, int max_block_num, int* block_table, int* cache_seqlens, int pick_block_num, int init_block_num, int local_block_num, Backend* backend);

    void update_one_block_fp16(const ggml_fp16_t* k_in, const ggml_fp16_t* v_in, int layer_id, int block_idx, Backend* backend);

    void get_one_block_fp16(ggml_fp16_t* k_in, ggml_fp16_t* v_in, int layer_id, int block_idx, Backend* backend);

    void update_importance_one_block(const ggml_fp16_t* importance,
                                     int layer_id,
                                     int block_idx,
                                     Backend* backend);
    void get_importance_one_block(ggml_fp16_t* importance, int layer_id, int block_idx, Backend* backend);

    void get_anchor_one_block(ggml_fp16_t* anchor, int layer_id, int block_idx, Backend* backend);

    void update_anchor_one_block(const ggml_fp16_t* anchor, int layer_id, int block_idx, Backend* backend);

    void calc_anchor_all_layers(int* block_table, int* cache_seqlens, int batch_size, int max_block_num, Backend* backend);

    void load_kvcache(std::string tensor_file_path, Backend* backend);
    void dump_kvcache(int* block_table, int cache_total_len, std::string tensor_file_path, Backend* backend);

    void get_all_kv_one_layer(int layer_id, ggml_fp16_t* k_in, ggml_fp16_t* v_in, Backend* backend);

   private:
    // Persistent data
    KVCacheConfig config_;
    int n_gqa_;                             // q_head_num / kv_head_num
    int cache_total_len_;                   // Number of tokens in cache
    std::vector<uint64_t> past_block_num_;  // [layer_num]
    std::vector<std::vector<std::vector<std::vector<block_q4_0>>>>
        k_cache_;  // [layer_num, kv_head_num, past_block_num, block_len *
                   // (head_dim / QK_4)]
    std::vector<std::vector<std::vector<std::vector<block_q4_0>>>>
        v_cache_;  // [layer_num, kv_head_num, past_block_num, head_dim *
                   // (block_len / QK_4)]

    std::vector<std::vector<std::vector<ggml_fp16_t>>>
        importance_;  // [layer_num, past_block_num, block_len]

    std::vector<std::vector<std::vector<std::vector<ggml_fp16_t>>>>
        anchor_;  // [layer_num, kv_head_num, past_block_num, anchor_num *
                  // head_dim]

    // Runtime data
    int64_t layer_id_;
    int64_t block_idx_;
    int* block_table_;
    uint64_t block_num_;

    // update/get
    int seq_len_;
    uint16_t* k_scales_;         // q4_0
    uint8_t* k_in_;              // q4_0
    uint16_t* v_scales_;         // q4_0
    uint8_t* v_in_;              // q4_0
    uint16_t* k_data_;           // fp16
    uint16_t* v_data_;           // fp16
    uint16_t* importance_data_;  // fp16
    uint16_t* anchor_data_;      // fp16

    // attn

    std::vector<std::priority_queue<std::pair<float, int>>> top_similar_block_;
    std::vector<int> cache_seqlens_;  // [batch_size]
    std::vector<std::vector<int>>
        block_table_before_retrieval_;  // [batch_size, max_block_num]
    std::vector<std::vector<int>>
        block_table_after_retrieval_;  // [batch_size, pick_block_num]

    std::vector<std::vector<std::unique_ptr<std::mutex>>>
        mutex_;  // [batch_size, kv_head_num]
    std::vector<std::vector<std::vector<block_q8_0>>>
        q_q8_0_;  // [batch_size, kv_head_num, n_gqa * head_dim / QK8_0]
    std::vector<std::vector<std::vector<float>>>
        output_fp32_;  // [batch_size, kv_head_num, n_gqa * head_dim]
    std::vector<std::vector<std::vector<float>>>
        attn_lse_;  // [batch_size, kv_head_num, n_gqa]

    std::vector<std::pair<int, int>> thread_cur_head_idx_;  // [thread_num]
    std::vector<std::vector<block_q8_0>>
        thread_local_output_q8_0_;  // [thread_num, n_gqa * head_dim / QK8_0]
    std::vector<std::vector<float>>
        thread_local_attn_score_;  // [thread_num, n_gqa * block_len]
    std::vector<std::vector<float>>
        thread_local_output_fp32_;  // [thread_num, n_gqa * head_dim]
    std::vector<std::vector<float>>
        thread_local_attn_lse_;  // [thread_num, n_gqa]
    std::vector<std::vector<float>>
        thread_local_cur_output_fp32_;  // [thread_num, n_gqa * head_dim]
    std::vector<std::vector<float>>
        thread_local_cur_attn_lse_;  // [thread_num, n_gqa]
    std::vector<std::vector<uint8_t>>
        thread_local_attn_mask_;  // [thread_num, block_len // 8]
    std::vector<std::vector<char>>
        thread_local_draft_;  // [thread_num, 2 * n_gqa * block_len + 6 * n_gqa *
                              // head_dim]

    // tmp space
    std::vector<float> q_fp32;  // [n_gqa * head_dim]
};

// 通用的注意力计算函数，可能单独提出来
void attn_with_kvcache_one_block(
    int head_dim,
    int bsz,
    ggml_type q_type,  // GGML data type of `Q`，只支持 fp16 和 q8_0
    // [bsz, head_dim]
    // 量化一定是 head_dim 这个维度，也就是 per_token，如果 head_dim % 32 !=
    // 0 报错 大小一定是 bsz * head_dim/32 * qtype_size
    const void* q,

    int past_kv_len,
    int past_kv_offset,
    bool is_full_attn,  // true 表示全 1 mask
    // 如果 is_full_attn = false，则传入一个 bit 矩阵表示 mask
    // [bsz, past_kv_len]
    const uint8_t* attn_mask,

    ggml_type k_type,  // GGML data type of `K Cache`，只支持 fp16 和 q4_0, q8_0
    int k_quant_type,  // 0 per_token, 1 per_channel, 其它报错
    // [seq_len, head_dim]
    // 如果 quant_type == 0， 要求 head_dim % 32 == 0
    // 如果 quant_type == 1， 要求 seq_len % 32 == 0
    const void* k_cache,

    // k_anchor_type 必须为 fp16
    int num_k_anchor,  // num_k_anchor == 0 表示没有 anchor
    // [num_k_anchor, head_dim]
    const void* k_cache_anchors,
    // 每个 token 都和前面最近的 pos 的 anchor 管，距离一样归前面管
    const int* k_cache_anchor_pos,

    // v cache 和上面一样
    ggml_type v_type,
    int v_quant_type,
    // [head_dim, seq_len]
    const void* v_cache,
    int num_v_anchor,
    const void* v_cache_anchors,
    const int* v_cache_anchor_pos,

    // [bsz, past_kv_len] 大小的中间预分配 buffer
    // 整个计算内部没有任何 malloc
    float* attn_score,

    // 输出： [bsz, head_dim]，类型和 q_type 保持一致
    void* output,
    // [bsz]
    float* lse,

    // 在外面开好的临时 buffer，(2 * bsz * past_kv_len + 6 * bsz *
    // head_dim)个字节是足够的大小
    void* draft

    // 在线加 rope，第一版本先不需要

    // rotary_cos=None,
    // rotary_sin=None,
    // cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    // cache_batch_idx: Optional[torch.Tensor] = None,
    // rotary_interleaved=True,

    // // 暂时不用支持
    // window_size=(-1, -1),  # -1 means infinite context window
    // alibi_slopes=None,
);

#endif