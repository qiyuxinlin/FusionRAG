#include "task_queue.h"

TaskQueue::TaskQueue()
    : stop(false) {
    worker = std::thread(&TaskQueue::processTasks, this);
}

TaskQueue::~TaskQueue() {
    {
        std::unique_lock<std::mutex> lock(mutex);
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void TaskQueue::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(mutex);
        tasks.push(task);
    }
    cv.notify_one();
}

void TaskQueue::sync() {
    std::unique_lock<std::mutex> lock(mutex);
    cvSync.wait(lock, [this] { return tasks.empty(); });
}

void TaskQueue::processTasks() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [this] { return !tasks.empty() || stop; });

            if (stop && tasks.empty()) {
                return;
            }

            task = tasks.front();
        }
        task();
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.pop();
            if (tasks.empty()) {
                cvSync.notify_all();
            }
        }
    }
}
