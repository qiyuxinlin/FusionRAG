#include "backend.h"

thread_local int Backend::thread_local_id = -1;

Backend::Backend(int thread_num) {
    thread_num_ = thread_num;
    thread_state_.resize(thread_num);
    for (int i = 0; i < thread_num; i++) {
        thread_state_[i].curr = std::make_unique<std::atomic<int>>();
        thread_state_[i].status = std::make_unique<std::atomic<ThreadStatus>>(ThreadStatus::STEALING);
    }
    workers_.resize(thread_num);
    for (int i = 1; i < thread_num; i++) {
        workers_[i] = std::thread(&Backend::worker_thread, this, i);
    }
}

Backend::~Backend() {
    for (int i = 0; i < thread_num_; i++) {
        thread_state_[i].status->store(ThreadStatus::EXIT, std::memory_order_relaxed);
    }
    for (int i = 1; i < thread_num_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }
}

int Backend::get_thread_num() {
    return thread_num_;
}

void Backend::do_parallel_job(int task_num, std::function<void(int)> func) {
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_num_; i++) {
        threads.push_back(std::thread([&](int thread_id) {
            for (int j = thread_id; j < task_num; j += thread_num_) {
                printf("thread_id = %d, task_id = %d\n", thread_id, j);
                func(j);
            }
        },
                                      i));
    }
    for (int i = 0; i < thread_num_; i++) {
        threads[i].join();
    }
}

void Backend::do_work_stealing_job(int task_num, std::function<void(int)> func) {
    func_ = func;
    int base = task_num / thread_num_;
    int remain = task_num % thread_num_;
    thread_state_[0].curr->store(0, std::memory_order_relaxed);
    thread_state_[0].end = base + (0 < remain);
    thread_state_[0].status->store(ThreadStatus::WORKING);

    // 为主线程设置 thread_local_id
    thread_local_id = 0;

    for (int i = 1; i < thread_num_; i++) {
        thread_state_[i].curr->store(thread_state_[i - 1].end, std::memory_order_relaxed);
        thread_state_[i].end = thread_state_[i].curr->load(std::memory_order_relaxed) + base + (i < remain);
        thread_state_[i].status->store(ThreadStatus::WORKING);
    }
    process_tasks(0);
    for (int i = 1; i < thread_num_; i++) {
        while (thread_state_[i].status->load(std::memory_order_relaxed) == ThreadStatus::WORKING) {
        }
    }
}

void Backend::process_tasks(int thread_id) {
    while (true) {
        int task_id = thread_state_[thread_id].curr->fetch_add(1, std::memory_order_relaxed);
        if (task_id >= thread_state_[thread_id].end) {
            break;
        }
        func_(task_id);
    }
    thread_state_[thread_id].status->store(ThreadStatus::STEALING);
    for (int t_offset = 1; t_offset < thread_num_; t_offset++) {
        int t_i = (thread_id + t_offset) % thread_num_;
        ThreadStatus status = thread_state_[t_i].status->load(std::memory_order_relaxed);
        if (status == ThreadStatus::STEALING) {
            continue;
        }
        while (true) {
            int task_id = thread_state_[thread_id].curr->fetch_add(1, std::memory_order_relaxed);
            if (task_id >= thread_state_[thread_id].end) {
                break;
            }
            func_(task_id);
        }
    }
}

void Backend::worker_thread(int thread_id) {
    auto start = std::chrono::steady_clock::now();
    thread_local_id = thread_id;  // 设置线程本地变量
    while (true) {
        ThreadStatus status = thread_state_[thread_id].status->load(std::memory_order_relaxed);
        if (status == ThreadStatus::WORKING) {
            process_tasks(thread_id);
            start = std::chrono::steady_clock::now();
        } else if (status == ThreadStatus::STEALING) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (duration > 10) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
    }
}