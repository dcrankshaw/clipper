#ifndef CLIPPER_DAG_EXECUTOR_HPP
#define CLIPPER_DAG_EXECUTOR_HPP

#include <memory>
#include <queue>
#include <unordered_map>

#include <clipper/datatypes.hpp>
#include <clipper/rpc_service.hpp>

namespace clipper {

// We use the system clock for the deadline time point
// due to its cross-platform consistency (consistent epoch, resolution)
using Deadline = std::chrono::time_point<std::chrono::system_clock>;

using ObjectId = std::vector<uint8_t>;

struct DeadlineCompare {
  bool operator()(const std::pair<Deadline, ObjectId> &lhs,
                  const std::pair<Deadline, ObjectId> &rhs) {
    return lhs.first > rhs.first;
  }
};

// thread safe model queue
class ModelQueue {
 public:
  ModelQueue();

  // Disallow copy and assign
  ModelQueue(const ModelQueue &) = delete;
  ModelQueue &operator=(const ModelQueue &) = delete;

  ModelQueue(ModelQueue &&) = default;
  ModelQueue &operator=(ModelQueue &&) = default;

  ~ModelQueue() = default;

  void add_task(ObjectId task);

  int get_size();

  std::vector<ObjectId> get_batch(
      std::function<int(Deadline)> &&get_batch_size);

 private:
  // Min PriorityQueue so that the task with the earliest
  // deadline is at the front of the queue
  using ModelPQueue =
      std::priority_queue<std::pair<Deadline, ObjectId>,
                          std::vector<std::pair<Deadline, ObjectId>>,
                          DeadlineCompare>;
  ModelPQueue queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_not_empty_condition_;

  // Deletes tasks with deadlines prior or equivalent to the
  // current system time. This method should only be called
  // when a unique lock on the queue_mutex is held.
  void remove_tasks_with_elapsed_deadlines();
};

class DAGExecutor {
 public:
  // This constructor does not perform any correctness checking
  // on the graph structure. So don't call it with a graph with cycles.
  DAGExecutor(std::vector<VersionedModelId> nodes,
              std::unordered_map<VersionedModelId, VersionedModelId> edges);

  // Disallow copy and assign
  DAGExecutor(const DAGExecutor &) = delete;
  DAGExecutor &operator=(const DAGExecutor &) = delete;

  DAGExecutor(DAGExecutor &&) = default;
  DAGExecutor &operator=(DAGExecutor &&) = default;

  ~DAGExecutor();

  // void add_new_container(VersionedModelId model, int replica_id,
  //                        int zmq_connection_id, InputType input_type);

  void process_response(VersionedModelId vm, std::vector<ObjectId> objects);

  void send_next_batch(VersionedModelId vm, int replica_id);

 private:
  std::unordered_map<VersionedModelId, std::shared_ptr<ModelQueue>>
      node_queues_;
  std::unordered_map<VersionedModelId, VersionedModelId> edges_;
  std::unique_ptr<rpc::RPCService> rpc_;
  std::shared_ptr<ActiveContainers> active_containers_;
  std::shared_ptr<std::atomic_bool> active_;
};

}  // namespace clipper

#endif  // CLIPPER_DAG_EXECUTOR_HPP
