
#ifndef CPUINFER_SHAREDMEMBUFFER_H
#define CPUINFER_SHAREDMEMBUFFER_H

#include <cstdint>
#include <cstdlib>
#include <vector>

class SharedMemBuffer {
   public:
    SharedMemBuffer();
    ~SharedMemBuffer();

    void alloc(std::vector<std::pair<void**, uint64_t>> requests);

   private:
    void* buffer_;
    uint64_t size_;
    std::vector<std::vector<std::pair<void**, uint64_t>>> hist_requests_;

    void arrange(std::vector<std::pair<void**, uint64_t>> requests);
};

static SharedMemBuffer shared_mem_buffer;

#endif