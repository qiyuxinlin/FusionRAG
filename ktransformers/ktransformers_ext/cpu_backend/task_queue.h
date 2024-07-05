#ifndef CPUINFER_TASKQUEUE_H
#define CPUINFER_TASKQUEUE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class TaskQueue {
   public:
    TaskQueue();
    ~TaskQueue();

    void enqueue(std::function<void()>);

    void sync();

   private:
    void processTasks();

    std::queue<std::function<void()>> tasks;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable cvSync;
    bool stop;
};
#endif