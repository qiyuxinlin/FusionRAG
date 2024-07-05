#ifndef CPUINFER_BACKEND_H
#define CPUINFER_BACKEND_H

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

enum ThreadStatus {
    WORKING,
    STEALING,
    EXIT,
};

struct ThreadState {
    std::unique_ptr<std::atomic<ThreadStatus>> status;
    std::unique_ptr<std::atomic<int>> curr;
    int end;
};

class Backend {
   public:
    Backend(int);
    ~Backend();
    int get_thread_num();
    void do_work_stealing_job(int, std::function<void(int)>);

   private:
    int thread_num_;
    std::vector<ThreadState> thread_state_;  // [thread_num]
    std::function<void(int)> func_;
    std::vector<std::thread> workers_;

    void process_tasks(int);
    void worker_thread(int);
};

#endif