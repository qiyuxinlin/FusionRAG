#include "kvcache.h"

void KVCache::load_kvcache(std::string tensor_file_path, Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream ifs_tensor(tensor_file_path, std::ios::binary);
    if (!ifs_tensor) {
        throw std::runtime_error("Failed to open tensor file");
    }

    ifs_tensor.read(reinterpret_cast<char *>(&cache_total_len_),
                    sizeof(cache_total_len_));

    int past_block_num =
        (cache_total_len_ + config_.block_len - 1) / config_.block_len;

    printf("cache_total_len: %d, past_block_num: %d\n", cache_total_len_,
           past_block_num);
    for (int i = 0; i < config_.layer_num; ++i) {
        past_block_num_[i] = past_block_num;
        k_cache_[i].resize(config_.kv_head_num);
        v_cache_[i].resize(config_.kv_head_num);
        importance_[i].resize(past_block_num_[i]);
        anchor_[i].resize(config_.kv_head_num);
        for (int j = 0; j < config_.kv_head_num; ++j) {
            k_cache_[i][j].resize(past_block_num_[i]);
            v_cache_[i][j].resize(past_block_num_[i]);
            anchor_[i][j].resize(past_block_num_[i]);
            for (int k = 0; k < past_block_num_[i]; ++k) {
                k_cache_[i][j][k].resize(config_.block_len *
                                         (config_.head_dim / 32));
                v_cache_[i][j][k].resize(config_.block_len *
                                         (config_.head_dim / 32));
                anchor_[i][j][k].resize(config_.anchor_num * config_.head_dim);
                importance_[i][k].resize(config_.block_len);
            }
        }
    }

    for (int i = 0; i < config_.layer_num; ++i) {
        for (int j = 0; j < config_.kv_head_num; ++j) {
            for (int k = 0; k < past_block_num_[i]; ++k) {
                ifs_tensor.read(
                    reinterpret_cast<char *>(k_cache_[i][j][k].data()),
                    k_cache_[i][j][k].size() * sizeof(block_q4_0));
                ifs_tensor.read(
                    reinterpret_cast<char *>(v_cache_[i][j][k].data()),
                    v_cache_[i][j][k].size() * sizeof(block_q4_0));
                ifs_tensor.read(
                    reinterpret_cast<char *>(anchor_[i][j][k].data()),
                    anchor_[i][j][k].size() * sizeof(ggml_fp16_t));
            }
        }
        for (int k = 0; k < past_block_num_[i]; ++k) {
            ifs_tensor.read(reinterpret_cast<char *>(importance_[i][k].data()),
                            importance_[i][k].size() * sizeof(ggml_fp16_t));
        }
    }

    ifs_tensor.close();
    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("time of load: %f s\n", diff.count());
}
void KVCache::dump_kvcache(int *block_table, int cache_total_len,
                           std::string tensor_file_path, Backend *backend) {
    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    std::ofstream ofs(tensor_file_path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Cannot open file " << tensor_file_path << std::endl;
        return;
    }

    ofs.write(reinterpret_cast<const char *>(&cache_total_len),
              sizeof(cache_total_len));

    int past_block_num =
        (cache_total_len + config_.block_len - 1) / config_.block_len;

    printf("cache_total_len: %d, past_block_num: %d\n", cache_total_len,
           past_block_num);

    for (int i = 0; i < config_.layer_num; ++i) {
        for (int j = 0; j < config_.kv_head_num; ++j) {
            for (int k = 0; k < past_block_num; ++k) {
                int block_idx = block_table[k];
                ofs.write(reinterpret_cast<const char *>(
                              k_cache_[i][j][block_idx].data()),
                          k_cache_[i][j][block_idx].size() *
                              sizeof(block_q4_0));
                ofs.write(reinterpret_cast<const char *>(
                              v_cache_[i][j][block_idx].data()),
                          v_cache_[i][j][block_idx].size() *
                              sizeof(block_q4_0));
                ofs.write(reinterpret_cast<const char *>(
                              anchor_[i][j][block_idx].data()),
                          anchor_[i][j][block_idx].size() *
                              sizeof(ggml_fp16_t));
            }
        }
        for (int k = 0; k < past_block_num; ++k) {
            int block_idx = block_table[k];
            ofs.write(reinterpret_cast<const char *>(
                          importance_[i][block_idx].data()),
                      importance_[i][block_idx].size() * sizeof(ggml_fp16_t));
        }
    }

    ofs.close();
    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("time of dump: %f s\n", diff.count());
}