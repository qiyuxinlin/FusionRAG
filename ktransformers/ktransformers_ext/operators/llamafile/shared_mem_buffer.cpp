#include "shared_mem_buffer.h"
#include <cstdio>

SharedMemBuffer::SharedMemBuffer() {
    buffer_ = nullptr;
    size_ = 0;
}

SharedMemBuffer::~SharedMemBuffer() {
    if (buffer_) {
        free(buffer_);
    }
}

void SharedMemBuffer::alloc(std::vector<std::pair<void**, uint64_t>> requests) {
    uint64_t size = 0;
    for (auto& request : requests) {
        size += request.second;
    }
    if (size > size_) {
        if (buffer_) {
            free(buffer_);
        }
        buffer_ = malloc(size);
        size_ = size;
        for (auto& requests : hist_requests_) {
            arrange(requests);
        }
    }
    arrange(requests);
    hist_requests_.push_back(requests);
}

void SharedMemBuffer::arrange(std::vector<std::pair<void**, uint64_t>> requests) {
    uint64_t offset = 0;
    for (auto& request : requests) {
        *(request.first) = buffer_ + offset;
        offset += request.second;
    }
}
