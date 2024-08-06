/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-08-06 10:33:49
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "cuda_runtime.h"

#include "backend.h"
#include "task_queue.h"

#include "llama.cpp/ggml-impl.h"

class CPUInfer {
   public:
    CPUInfer(int thread_num) {
        backend_ = new Backend(thread_num - 1);
        task_queue_ = new TaskQueue();
        for (int i = 0; i < (1 << 16); ++i) {
            ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
        }
    }

    ~CPUInfer() {
        delete backend_;
        delete task_queue_;
    }

    template <typename Func, typename Obj, typename... Args>
    void enqueue(Func f, Obj* obj, Args... args) {
        task_queue_->enqueue([=]() {
            std::invoke(f, *obj, args..., backend_);
        });
    }

    void submit(std::function<void(void*)> func_) {
        func_((void*)this);
    }

    void submit_with_cuda_stream(intptr_t user_cuda_stream, std::function<void(void*)> func_) {
        cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)*func_.target<void (*)(void*)>(), (void*)this);
    }

    void sync() {
        task_queue_->sync();
    }

    void sync_with_cuda_stream(intptr_t user_cuda_stream) {
        auto cpuinfer_sync = [](void* cpu_infer_ptr) {
            CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
            cpuinfer->sync();
        };
        cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)cpuinfer_sync, (void*)this);
    }

   public:
    Backend* backend_;
    TaskQueue* task_queue_;
};

#endif