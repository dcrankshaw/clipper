
#include <clipper/constants.hpp>
#include <clipper/containers.hpp>
#include <clipper/dag_executor.hpp>
#include <clipper/logging.hpp>
#include <clipper/rpc_service.hpp>
#include <clipper/threadpool.hpp>

namespace clipper {

const std::string LOGGING_TAG_DAG_EXECUTOR = "DAG_EXECUTOR";

// NOTE: This doesn't really do anything, it's just to preserve
// the deadline code which we'll need in the future. Deadlines
// are arbitrary right now though.
const int latency_slo = 10000;

constexpr int batch_size = 10;

DAGExecutor::DAGExecutor(
    std::vector<VersionedModelId> nodes,
    std::unordered_map<VersionedModelId, VersionedModelId> edges)
    : node_queues_({}),
      edges_(edges),
      rpc_(std::make_unique<rpc::RPCService>()),
      active_containers_(std::make_shared<ActiveContainers>()),
      active_(std::make_shared<std::atomic_bool>(true)) {
  for (auto node : nodes) {
    node_queues_.emplace(std::make_pair(node, std::make_shared<ModelQueue>()));
  }

  rpc_->start(
      "*", RPC_SERVICE_PORT,
      // [ this, task_executor_valid = active_ ](
      //                            VersionedModelId model, int replica_id) {
      //   if (*task_executor_valid) {
      //     send_next_batch(model, replica_id);
      //   } else {
      //     log_info(LOGGING_TAG_DAG_EXECUTOR,
      //              "Not running send_next_batch callback because "
      //              "DAGExecutor has been destroyed.");
      //   }
      // },
      [ this, task_executor_valid = active_ ](VersionedModelId model,
                                              int replica_id,
                                              std::vector<ObjectId> objects) {
        if (*task_executor_valid) {
          process_response(model, std::move(objects));
          std::shared_ptr<ModelContainer> container =
              active_containers_->get_model_replica(model, replica_id);
          if (container->get_type() != ContainerType::Source) {
            TaskExecutionThreadPool::submit_job(
                model, replica_id, [ this, task_executor_valid = active_ ](
                                       VersionedModelId model, int replica_id) {
                  if (*task_executor_valid) {
                    send_next_batch(model, replica_id);
                  } else {
                    log_info(LOGGING_TAG_DAG_EXECUTOR,
                             "Not running send_next_batch callback because "
                             "DAGExecutor has been destroyed.");
                  }
                },
                model, replica_id);
          }
        } else {
          log_info(LOGGING_TAG_DAG_EXECUTOR,
                   "Not running process_response callback because "
                   "DAGExecutor has been destroyed.");
        }

      },
      [ this, task_executor_valid = active_ ](
          VersionedModelId model, int replica_id, int zmq_connection_id,
          InputType input_type, ContainerType ct) {
        if (*task_executor_valid) {
          log_info_formatted(LOGGING_TAG_DAG_EXECUTOR,
                             "NEW CONNECTION: Replica {} for model {}",
                             std::to_string(replica_id), model.serialize());
          active_containers_->add_container(model, zmq_connection_id,
                                            replica_id, input_type, ct);
          std::shared_ptr<ModelContainer> container =
              active_containers_->get_model_replica(model, replica_id);
          if (container->get_type() != ContainerType::Source) {
            TaskExecutionThreadPool::submit_job(
                model, replica_id, [ this, task_executor_valid = active_ ](
                                       VersionedModelId model, int replica_id) {
                  if (*task_executor_valid) {
                    send_next_batch(model, replica_id);
                  } else {
                    log_info(LOGGING_TAG_DAG_EXECUTOR,
                             "Not running send_next_batch callback because "
                             "DAGExecutor has been destroyed.");
                  }
                },
                model, replica_id);
          }
        } else {
          log_info(LOGGING_TAG_DAG_EXECUTOR,
                   "Not running new_container callback because "
                   "DAGExecutor has been destroyed.");
        }
      });
}

DAGExecutor::~DAGExecutor() { active_->store(false); }

void DAGExecutor::process_response(VersionedModelId vm,
                                   std::vector<ObjectId> objects) {
  log_info_formatted(LOGGING_TAG_DAG_EXECUTOR,
                     "Processing response from model {}", vm.serialize());
  auto next_node = edges_.find(vm);
  if (next_node != edges_.end()) {
    auto q = node_queues_[next_node->second];
    for (auto o : objects) {
      q->add_task(o);
    }
    log_info_formatted(LOGGING_TAG_DAG_EXECUTOR, "Added {} tasks to {} queue",
                       objects.size(), next_node->second.serialize());
  } else {
    log_error_formatted(LOGGING_TAG_DAG_EXECUTOR,
                        "No outgoing node found from model {}", vm.serialize());
  }
}

void DAGExecutor::send_next_batch(VersionedModelId vm, int replica_id) {
  log_info_formatted(LOGGING_TAG_DAG_EXECUTOR,
                     "Ready to send next batch for model {} replica {}",
                     vm.serialize(), std::to_string(replica_id));
  auto q = node_queues_[vm];
  log_info_formatted(LOGGING_TAG_DAG_EXECUTOR, "{} queue size {}",
                     vm.serialize(), std::to_string(q->get_size()));
  std::vector<ObjectId> batch =
      q->get_batch([](Deadline /*deadline*/) { return batch_size; });
  log_info(LOGGING_TAG_DAG_EXECUTOR, "Found {} item batch", batch.size());
  std::shared_ptr<ModelContainer> container =
      active_containers_->get_model_replica(vm, replica_id);
  if (batch.size() > 0) {
    if (!container) {
      throw std::runtime_error(
          "TaskExecutor failed to find previously registered active "
          "container!");
    }
    TransformerBatchMessage msg{batch};
    rpc_->send_transformer_message(msg, container->container_id_);
    log_info_formatted(LOGGING_TAG_DAG_EXECUTOR, "Sent batch to model {}",
                       vm.serialize());
  }

  // TODO: need some back pressure here?
  // If this is a sink node, don't wait for a response,
  // immediately allow the next batch of messages to be sent
  if (container->get_type() == ContainerType::Sink) {
    TaskExecutionThreadPool::submit_job(
        vm, replica_id, [ this, task_executor_valid = active_ ](
                            VersionedModelId model, int replica_id) {
          if (*task_executor_valid) {
            send_next_batch(model, replica_id);

          } else {
            log_info(LOGGING_TAG_DAG_EXECUTOR,
                     "Not running send_next_batch callback because "
                     "DAGExecutor has been destroyed.");
          }
        },
        vm, replica_id);
  }
}

ModelQueue::ModelQueue() : queue_(ModelPQueue{}) {}

void ModelQueue::add_task(ObjectId task) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  Deadline deadline =
      std::chrono::system_clock::now() + std::chrono::microseconds(latency_slo);
  queue_.emplace(deadline, std::move(task));
  queue_not_empty_condition_.notify_one();
}

int ModelQueue::get_size() {
  std::unique_lock<std::mutex> l(queue_mutex_);
  return queue_.size();
}

std::vector<ObjectId> ModelQueue::get_batch(
    std::function<int(Deadline)>&& get_batch_size) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  // remove_tasks_with_elapsed_deadlines();
  queue_not_empty_condition_.wait(lock, [this]() { return !queue_.empty(); });
  // remove_tasks_with_elapsed_deadlines();
  Deadline deadline = queue_.top().first;
  int max_batch_size = get_batch_size(deadline);
  std::vector<ObjectId> batch;
  while (batch.size() < (size_t)max_batch_size && queue_.size() > 0) {
    batch.push_back(queue_.top().second);
    queue_.pop();
  }
  return batch;
}

void ModelQueue::remove_tasks_with_elapsed_deadlines() {
  std::chrono::time_point<std::chrono::system_clock> current_time =
      std::chrono::system_clock::now();
  while (!queue_.empty()) {
    Deadline first_deadline = queue_.top().first;
    if (first_deadline <= current_time) {
      // If a task's deadline has already elapsed,
      // we should not process it
      queue_.pop();
    } else {
      break;
    }
  }
}
}
