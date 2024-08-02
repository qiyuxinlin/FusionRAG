
#ifndef CPUINFER_SHAREDMEMBUFFER_H
#define CPUINFER_SHAREDMEMBUFFER_H

#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

class SharedMemBuffer {
   public:
    SharedMemBuffer();
    ~SharedMemBuffer();

    void alloc(void* object, std::vector<std::pair<void**, uint64_t>> requests);
    void dealloc(void* object);

   private:
    void* buffer_;
    uint64_t size_;
    std::map<void*, std::vector<std::vector<std::pair<void**, uint64_t>>>> hist_requests_;

    void arrange(std::vector<std::pair<void**, uint64_t>> requests);
};

static SharedMemBuffer shared_mem_buffer;

#endif