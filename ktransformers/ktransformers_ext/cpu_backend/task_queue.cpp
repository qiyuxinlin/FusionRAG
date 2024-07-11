#include "task_queue.h"

TaskQueue::TaskQueue() {
    worker = std::thread(&TaskQueue::processTasks, this);
    sync_flag.store(true, std::memory_order_seq_cst);
    exit_flag.store(false, std::memory_order_seq_cst);
}

TaskQueue::~TaskQueue() {
    exit_flag.store(true, std::memory_order_seq_cst);
    if (worker.joinable()) {
        worker.join();
    }
}

void TaskQueue::enqueue(std::function<void()> task) {
    mutex.lock();
    tasks.push(task);
    sync_flag.store(false, std::memory_order_seq_cst);
    mutex.unlock();
}

void TaskQueue::sync() {
    while (!sync_flag.load(std::memory_order_seq_cst));
}

void TaskQueue::processTasks() {
    while (true) {
        mutex.lock();
        if (tasks.empty()) {
            if (exit_flag.load(std::memory_order_seq_cst)) {
                return;
            }
            mutex.unlock();
            continue;
        }
        std::function<void()> task = tasks.front();
        mutex.unlock();
        task();
        mutex.lock();
        tasks.pop();
        if (tasks.empty()) {
            sync_flag.store(true, std::memory_order_seq_cst);
        }
        mutex.unlock();
    }
}
