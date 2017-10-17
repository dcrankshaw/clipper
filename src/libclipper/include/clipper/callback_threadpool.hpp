#ifndef CLIPPER_LIB_CALLBACK_THREADPOOL_HPP
#define CLIPPER_LIB_CALLBACK_THREADPOOL_HPP

#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include <boost/thread.hpp>

#include "datatypes.hpp"
#include "logging.hpp"
#include "threadpool.hpp"

namespace clipper {

// const std::string LOGGING_TAG_THREADPOOL = "CALLBACKTHREADPOOL";

/// Implementation adapted from
/// https://goo.gl/Iav87R

// template <typename T>
// class ThreadSafeQueue {
//  public:
//   #<{(|*
//    * Destructor.
//    |)}>#
//   ~ThreadSafeQueue(void) { invalidate(); }
//
//   #<{(|*
//    * Attempt to get the first value in the queue.
//    * Returns true if a value was successfully written to the out parameter,
//    * false otherwise.
//    |)}>#
//   bool try_pop(T& out) {
//     std::lock_guard<std::mutex> lock{mutex_};
//     if (queue_.empty() || !valid_) {
//       return false;
//     }
//     out = std::move(queue_.front());
//     queue_.pop();
//     return true;
//   }
//
//   #<{(|*
//    * Get the first value in the queue.
//    * Will block until a value is available unless clear is called or the
//    * instance is destructed.
//    * Returns true if a value was successfully written to the out parameter,
//    * false otherwise.
//    |)}>#
//   bool wait_pop(T& out) {
//     std::unique_lock<std::mutex> lock{mutex_};
//     condition_.wait(lock, [this]() { return !queue_.empty() || !valid_; });
//     #<{(|
//      * Using the condition in the predicate ensures that spurious wakeups with a
//      * valid
//      * but empty queue will not proceed, so only need to check for validity
//      * before proceeding.
//      |)}>#
//     if (!valid_) {
//       return false;
//     }
//     out = std::move(queue_.front());
//     queue_.pop();
//     return true;
//   }
//
//   #<{(|*
//    * Push a new value onto the queue.
//    |)}>#
//   void push(T value) {
//     std::lock_guard<std::mutex> lock{mutex_};
//     queue_.push(std::move(value));
//     condition_.notify_one();
//   }
//
//   #<{(|*
//    * Check whether or not the queue is empty.
//    |)}>#
//   bool empty(void) const {
//     std::lock_guard<std::mutex> lock{mutex_};
//     return queue_.empty();
//   }
//
//   #<{(|*
//    * Clear all items from the queue.
//    |)}>#
//   void clear(void) {
//     std::lock_guard<std::mutex> lock{mutex_};
//     while (!queue_.empty()) {
//       queue_.pop();
//     }
//     condition_.notify_all();
//   }
//
//   #<{(|*
//    * Invalidate the queue.
//    * Used to ensure no conditions are being waited on in wait_pop when
//    * a thread or the application is trying to exit.
//    * The queue is invalid after calling this method and it is an error
//    * to continue using a queue after this method has been called.
//    |)}>#
//   void invalidate(void) {
//     std::lock_guard<std::mutex> lock{mutex_};
//     valid_ = false;
//     condition_.notify_all();
//   }
//
//   #<{(|*
//    * Returns whether or not this queue is valid.
//    |)}>#
//   bool is_valid(void) const {
//     std::lock_guard<std::mutex> lock{mutex_};
//     return valid_;
//   }
//
//  private:
//   std::atomic_bool valid_{true};
//   mutable std::mutex mutex_;
//   std::queue<T> queue_;
//   std::condition_variable condition_;
// };

class CallbackThreadPool {
 private:
  class IThreadTask {
   public:
    IThreadTask(void) = default;
    virtual ~IThreadTask(void) = default;
    IThreadTask(const IThreadTask& rhs) = delete;
    IThreadTask& operator=(const IThreadTask& rhs) = delete;
    IThreadTask(IThreadTask&& other) = default;
    IThreadTask& operator=(IThreadTask&& other) = default;

    /**
     * Run the task.
     */
    virtual void execute() = 0;
  };

  template <typename Func>
  class ThreadTask : public IThreadTask {
   public:
    ThreadTask(Func&& func) : func_{std::move(func)} {}

    ~ThreadTask(void) override = default;
    ThreadTask(const ThreadTask& rhs) = delete;
    ThreadTask& operator=(const ThreadTask& rhs) = delete;
    ThreadTask(ThreadTask&& other) = default;
    ThreadTask& operator=(ThreadTask&& other) = default;

    /**
     * Run the task.
     */
    void execute() override { func_(); }

   private:
    Func func_;
  };

 public:
  explicit CallbackThreadPool(const std::uint32_t numThreads) : done_{false}, queue_{}, threads_{} {
    try {
      for(std::uint32_t i = 0u; i < numThreads; ++i) {
        threads_.emplace_back(&CallbackThreadPool::worker, this);
      }
    }
    catch(...) {
      destroy();
      throw;
    }
  }

  /**
   * Non-copyable.
   */
  CallbackThreadPool(const CallbackThreadPool& rhs) = delete;

  /**
   * Non-assignable.
   */
  CallbackThreadPool& operator=(const CallbackThreadPool& rhs) = delete;

  /**
   * Destructor.
   */
  ~CallbackThreadPool(void) { destroy(); }

  /**
   * Submit a job to be run by the thread pool.
   */
  template <typename Func, typename... Args>
  auto submit(Func&& func, Args&&... args) {
    auto boundTask =
        std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    using ResultType = std::result_of_t<decltype(boundTask)()>;
    using PackagedTask = boost::packaged_task<ResultType()>;
    using TaskType = ThreadTask<PackagedTask>;
    PackagedTask task{std::move(boundTask)};
    auto result_future = task.get_future();
    queue_.push(std::make_unique<TaskType>(std::move(task)));
    return result_future;
  }

 private:
  /**
   * Constantly running function each thread uses to acquire work items from the
   * queue.
   */
  void worker() {
    while (!done_) {
      std::unique_ptr<IThreadTask> pTask{nullptr};
      // NOTE: The use of try_pop here means the worker will spin instead of
      // block while waiting for work. This is intentional. We defer to the
      // submitted tasks to block when no work is available.
      if (queue_.try_pop(pTask)) {
        pTask->execute();
      }
    }
  }

  /**
   * Invalidates the queue and joins all running threads.
   */
  void destroy(void) {
    log_info(LOGGING_TAG_THREADPOOL, "Destroying threadpool");
    done_ = true;
    queue_.invalidate();
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

 private:
  std::atomic_bool done_;
  ThreadSafeQueue<std::unique_ptr<IThreadTask>> queue_;
  std::vector<std::thread> threads_;
};

}

// namespace TaskExecutionThreadPool {
//
// #<{(|*
//  * Convenience method to get the task execution thread pool for the application.
//  |)}>#
// inline ThreadPool& get_thread_pool(void) {
//   static ThreadPool taskExecutionPool;
//   return taskExecutionPool;
// }
//
// #<{(|*
//  * Submit a job to the task execution thread pool.
//  |)}>#
// template <typename Func, typename... Args>
// inline auto submit_job(VersionedModelId vm, int replica_id, Func&& func,
//                        Args&&... args) {
//   return get_thread_pool().submit(vm, replica_id, std::forward<Func>(func),
//                                   std::forward<Args>(args)...);
// }
//
// inline void create_queue(VersionedModelId vm, int replica_id) {
//   get_thread_pool().create_queue(vm, replica_id);
// }
//
// }  // namespace DefaultThreadPool
// }  // namespace clipper

#endif  // CLIPPER_LIB_CALLBACK_THREADPOOL_HPP
