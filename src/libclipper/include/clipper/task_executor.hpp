#ifndef CLIPPER_LIB_TASK_EXECUTOR_H
#define CLIPPER_LIB_TASK_EXECUTOR_H

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <boost/optional.hpp>

#include <redox.hpp>
#include <zmq.hpp>

#include <clipper/callback_threadpool.hpp>
#include <clipper/config.hpp>
#include <clipper/containers.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/redis.hpp>
#include <clipper/rpc_service.hpp>
#include <clipper/threadpool.hpp>
#include <clipper/util.hpp>

namespace clipper {

const std::string LOGGING_TAG_TASK_EXECUTOR = "TASKEXECUTOR";

class ModelMetrics {
 public:
  explicit ModelMetrics(VersionedModelId model)
      : model_(model),
        latency_(metrics::MetricsRegistry::get_metrics().create_histogram(
            "model:" + model.serialize() + ":prediction_latency",
            "microseconds", 4096)),
        throughput_(metrics::MetricsRegistry::get_metrics().create_meter(
            "model:" + model.serialize() + ":prediction_throughput")),
        num_predictions_(metrics::MetricsRegistry::get_metrics().create_counter(
            "model:" + model.serialize() + ":num_predictions")),
        cache_hit_ratio_(
            metrics::MetricsRegistry::get_metrics().create_ratio_counter(
                "model:" + model.serialize() + ":cache_hit_ratio")),
        batch_size_(metrics::MetricsRegistry::get_metrics().create_histogram(
            "model:" + model.serialize() + ":batch_size", "queries", 4096)) {}
  ~ModelMetrics() = default;
  ModelMetrics(const ModelMetrics &) = default;
  ModelMetrics &operator=(const ModelMetrics &) = default;

  ModelMetrics(ModelMetrics &&) = default;
  ModelMetrics &operator=(ModelMetrics &&) = default;

  VersionedModelId model_;
  std::shared_ptr<metrics::Histogram> latency_;
  std::shared_ptr<metrics::Meter> throughput_;
  std::shared_ptr<metrics::Counter> num_predictions_;
  std::shared_ptr<metrics::RatioCounter> cache_hit_ratio_;
  std::shared_ptr<metrics::Histogram> batch_size_;
};

class CacheEntry {
 public:
  CacheEntry();
  ~CacheEntry() = default;

  CacheEntry(const CacheEntry &) = delete;
  CacheEntry &operator=(const CacheEntry &) = delete;

  CacheEntry(CacheEntry &&) = default;
  CacheEntry &operator=(CacheEntry &&) = default;

  bool completed_ = false;
  bool used_ = true;
  Output value_;
  std::vector<std::function<void(Output)>> value_callbacks_;
};

// A cache page is a pair of <hash, entry_size>
using CachePage = std::pair<long, long>;

// NOTE: Prediction cache is now a query cache
class QueryCache {
 public:
  QueryCache(size_t size_bytes);
  bool fetch(const VersionedModelId &model, const QueryId query_id,
             std::function<void(Output)> callback);

  void put(const VersionedModelId &model, const QueryId query_id,
           Output output);

 private:
  size_t hash(const VersionedModelId &model, const QueryId query_id) const;
  void insert_entry(const long key, CacheEntry &value);
  void evict_entries(long space_needed_bytes);

  std::mutex m_;
  const size_t max_size_bytes_;
  size_t size_bytes_ = 0;
  // TODO cache needs a promise as well?
  std::unordered_map<long, CacheEntry> entries_;
  std::vector<long> page_buffer_;
  size_t page_buffer_index_ = 0;
  std::shared_ptr<metrics::Counter> lookups_counter_;
  std::shared_ptr<metrics::RatioCounter> hit_ratio_;
  CallbackThreadPool callback_threadpool_;
};

struct DeadlineCompare {
  bool operator()(const std::pair<Deadline, PredictTask> &lhs,
                  const std::pair<Deadline, PredictTask> &rhs) {
    return lhs.first > rhs.first;
  }
};

// thread safe model queue
class ModelQueue {
 public:
  ModelQueue(std::string name)
      : queue_(ModelPQueue{}),
        lock_latency_hist_(
            metrics::MetricsRegistry::get_metrics().create_histogram(
                name + ":lock_latency", "microseconds", 4096)),
        queue_size_hist_(
            metrics::MetricsRegistry::get_metrics().create_histogram(
                name + ":queue_size", "microseconds", 1000)) {}

  // Disallow copy and assign
  ModelQueue(const ModelQueue &) = delete;
  ModelQueue &operator=(const ModelQueue &) = delete;

  ModelQueue(ModelQueue &&) = default;
  ModelQueue &operator=(ModelQueue &&) = default;

  ~ModelQueue() = default;

  void add_task(PredictTask task) {
    std::chrono::time_point<std::chrono::system_clock> start_time =
        std::chrono::system_clock::now();
    std::lock_guard<std::mutex> lock(queue_mutex_);

    std::chrono::time_point<std::chrono::system_clock> current_time =
        std::chrono::system_clock::now();

    auto lock_latency = current_time - start_time;
    long lock_latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(lock_latency)
            .count();
    lock_latency_hist_->insert(static_cast<int64_t>(lock_latency_micros));

    Deadline deadline =
        current_time + std::chrono::microseconds(task.latency_slo_micros_);
    queue_.emplace(deadline, std::move(task));
    queue_not_empty_condition_.notify_one();
  }

  int get_size() {
    std::unique_lock<std::mutex> l(queue_mutex_);
    return queue_.size();
  }

  std::vector<PredictTask> get_batch(
      std::function<int(Deadline)> &&get_batch_size) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_not_empty_condition_.wait(lock, [this]() { return !queue_.empty(); });
    Deadline deadline = queue_.top().first;
    int max_batch_size = get_batch_size(deadline);
    std::vector<PredictTask> batch;
    while (batch.size() < (size_t)max_batch_size && queue_.size() > 0) {
      batch.push_back(queue_.top().second);
      queue_.pop();
    }
    queue_size_hist_->insert(static_cast<int64_t>(queue_.size()));
    return batch;
  }

  void drain_queue() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_ = ModelPQueue{};
  }

 private:
  // Min PriorityQueue so that the task with the earliest
  // deadline is at the front of the queue
  using ModelPQueue =
      std::priority_queue<std::pair<Deadline, PredictTask>,
                          std::vector<std::pair<Deadline, PredictTask>>,
                          DeadlineCompare>;
  ModelPQueue queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_not_empty_condition_;
  std::shared_ptr<metrics::Histogram> lock_latency_hist_;
  std::shared_ptr<metrics::Histogram> queue_size_hist_;

  // Deletes tasks with deadlines prior or equivalent to the
  // current system time. This method should only be called
  // when a unique lock on the queue_mutex is held.
  void remove_tasks_with_elapsed_deadlines() {
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
};

class InflightMessage {
 public:
  InflightMessage(
      const std::chrono::time_point<std::chrono::system_clock> send_time,
      const int container_id, const VersionedModelId model,
      const int replica_id, const InputVector input, const QueryId query_id)
      : send_time_(send_time),
        container_id_(container_id),
        model_(model),
        replica_id_(replica_id),
        input_(input),
        query_id_(query_id) {}

  // Default copy and move constructors
  InflightMessage(const InflightMessage &) = default;
  InflightMessage(InflightMessage &&) = default;

  // Default assignment operators
  InflightMessage &operator=(const InflightMessage &) = default;
  InflightMessage &operator=(InflightMessage &&) = default;

  std::chrono::time_point<std::chrono::system_clock> send_time_;
  int container_id_;
  VersionedModelId model_;
  int replica_id_;
  InputVector input_;
  QueryId query_id_;
};

void noop_free(void *data, void *hint);

void real_free(void *data, void *hint);

std::vector<zmq::message_t> construct_batch_message(
    std::vector<PredictTask> tasks);

class TaskExecutor {
 public:
  ~TaskExecutor() { active_->store(false); };
  explicit TaskExecutor()
      : active_(std::make_shared<std::atomic_bool>(true)),
        active_containers_(std::make_shared<ActiveContainers>()),
        rpc_(std::make_unique<rpc::RPCService>()),
        cache_(std::make_unique<QueryCache>(0)),
        model_queues_({}),
        model_metrics_({}) {
    log_info(LOGGING_TAG_TASK_EXECUTOR, "TaskExecutor started");
    rpc_->start(
        "*", RPC_SERVICE_SEND_PORT, RPC_SERVICE_RECV_PORT, [ this, task_executor_valid = active_ ](
                                   VersionedModelId model, int replica_id) {
          if (*task_executor_valid) {
            on_container_ready(model, replica_id);
          } else {
            log_info(LOGGING_TAG_TASK_EXECUTOR,
                     "Not running on_container_ready callback because "
                     "TaskExecutor has been destroyed.");
          }
        },
        [ this, task_executor_valid = active_ ](rpc::RPCResponse response) {
          if (*task_executor_valid) {
            on_response_recv(std::move(response));
          } else {
            log_info(LOGGING_TAG_TASK_EXECUTOR,
                     "Not running on_response_recv callback because "
                     "TaskExecutor has been destroyed.");
          }

        });
    Config &conf = get_config();
    while (!redis_connection_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_TASK_EXECUTOR,
                "TaskExecutor failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_TASK_EXECUTOR,
                "TaskExecutor subscriber failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    redis::subscribe_to_model_changes(redis_subscriber_, [
      this, task_executor_valid = active_
    ](const std::string &key, const std::string &event_type) {
      if (event_type == "hset" && *task_executor_valid) {
        auto model_info = redis::get_model_by_key(redis_connection_, key);
        VersionedModelId model_id = VersionedModelId(
            model_info["model_name"], model_info["model_version"]);
        int batch_size = std::stoi(model_info["batch_size"]);
        active_containers_->register_batch_size(model_id, batch_size);
      }
    });

    std::vector<VersionedModelId> models = redis::get_all_models(redis_connection_);
    for (auto model_id: models) {
      auto model_info = redis::get_model(redis_connection_, model_id);
      // VersionedModelId model_id = VersionedModelId(
      //     model_info["model_name"], model_info["model_version"]);
      int batch_size = std::stoi(model_info["batch_size"]);
      active_containers_->register_batch_size(model_id, batch_size);
    }

    redis::send_cmd_no_reply<std::string>(
        redis_connection_, {"CONFIG", "SET", "notify-keyspace-events", "AKE"});
    redis::subscribe_to_container_changes(
        redis_subscriber_,
        // event_type corresponds to one of the Redis event types
        // documented in https://redis.io/topics/notifications.
        [ this, task_executor_valid = active_ ](const std::string &key,
                                                const std::string &event_type) {
          if (event_type == "hset" && *task_executor_valid) {
            auto container_info =
                redis::get_container_by_key(redis_connection_, key);
            VersionedModelId vm = VersionedModelId(
                container_info["model_name"], container_info["model_version"]);
            int replica_id = std::stoi(container_info["model_replica_id"]);

            active_containers_->add_container(
                vm, std::stoi(container_info["zmq_connection_id"]), replica_id,
                parse_input_type(container_info["input_type"]));

            TaskExecutionThreadPool::create_queue(vm, replica_id);
            TaskExecutionThreadPool::submit_job(
                vm, replica_id, [this, vm, replica_id]() {
                  on_container_ready(vm, replica_id);
                });
            TaskExecutionThreadPool::submit_job(
                vm, replica_id, [this, vm, replica_id]() {
                  on_container_ready(vm, replica_id);
                });
            bool created_queue = create_model_queue_if_necessary(vm);
            if (created_queue) {
              log_info_formatted(LOGGING_TAG_TASK_EXECUTOR,
                                 "Created queue for new model: {} : {}",
                                 vm.get_name(), vm.get_id());
            }
          } else if (!*task_executor_valid) {
            log_info(LOGGING_TAG_TASK_EXECUTOR,
                     "Not running TaskExecutor's "
                     "subscribe_to_container_changes callback because "
                     "TaskExecutor has been destroyed.");
          }

        });
    throughput_meter_ = metrics::MetricsRegistry::get_metrics().create_meter(
        "internal:aggregate_model_throughput");
    predictions_counter_ =
        metrics::MetricsRegistry::get_metrics().create_counter(
            "internal:aggregate_num_predictions");
  }

  // Disallow copy
  TaskExecutor(const TaskExecutor &other) = delete;
  TaskExecutor &operator=(const TaskExecutor &other) = delete;

  TaskExecutor(TaskExecutor &&other) = default;
  TaskExecutor &operator=(TaskExecutor &&other) = default;

  void schedule_prediction(
      PredictTask task,
      std::function<void(Output)> &&task_completion_callback) {
    predictions_counter_->increment(1);
    // add each task to the queue corresponding to its associated model
    boost::shared_lock<boost::shared_mutex> lock(model_queues_mutex_);
    auto model_queue_entry = model_queues_.find(task.model_);
    if (model_queue_entry != model_queues_.end()) {
      bool cached = cache_->fetch(task.model_, task.query_id_,
                                  std::move(task_completion_callback));
      if (!cached) {
        task.recv_time_ = std::chrono::system_clock::now();
        model_queue_entry->second->add_task(task);
        log_info_formatted(LOGGING_TAG_TASK_EXECUTOR,
                           "Adding task to queue. QueryID: {}, model: {}",
                           task.query_id_, task.model_.serialize());
      }
    } else {
      log_error_formatted(LOGGING_TAG_TASK_EXECUTOR,
                          "Received task for unknown model: {} : {}",
                          task.model_.get_name(), task.model_.get_id());
    }
  }

  void drain_queues() {
    boost::unique_lock<boost::shared_mutex> lock(model_queues_mutex_);
    for (auto entry: model_queues_) {
      entry.second->drain_queue();
    }

  }

 private:
  // active_containers_ is shared with the RPC service so it can add new
  // containers to the collection when they connect
  std::shared_ptr<std::atomic_bool> active_;
  std::shared_ptr<ActiveContainers> active_containers_;
  std::unique_ptr<rpc::RPCService> rpc_;
  std::unique_ptr<QueryCache> cache_;
  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
  std::mutex inflight_messages_mutex_;
  std::unordered_map<int, std::vector<InflightMessage>> inflight_messages_;
  std::shared_ptr<metrics::Counter> predictions_counter_;
  std::shared_ptr<metrics::Meter> throughput_meter_;
  boost::shared_mutex model_queues_mutex_;
  std::unordered_map<VersionedModelId, std::shared_ptr<ModelQueue>>
      model_queues_;
  boost::shared_mutex model_metrics_mutex_;
  std::unordered_map<VersionedModelId, ModelMetrics> model_metrics_;
  static constexpr int INITIAL_MODEL_QUEUES_MAP_SIZE = 100;

  bool create_model_queue_if_necessary(const VersionedModelId &model_id) {
    // Adds a new <model_id, task_queue> entry to the queues map, if one
    // does not already exist
    boost::unique_lock<boost::shared_mutex> l(model_queues_mutex_);
    auto queue_added = model_queues_.emplace(std::make_pair(
        model_id, std::make_shared<ModelQueue>(model_id.serialize())));
    bool queue_created = queue_added.second;
    if (queue_created) {
      boost::unique_lock<boost::shared_mutex> l(model_metrics_mutex_);
      model_metrics_.insert(std::make_pair(model_id, ModelMetrics(model_id)));
      // model_metrics_.emplace(std::piecewise_construct,
      //                        std::forward_as_tuple(model_id),
      //                        std::forward_as_tuple(model_id));
    }
    return queue_created;
  }

  void on_container_ready(VersionedModelId model_id, int replica_id) {
    std::shared_ptr<ModelContainer> container =
        active_containers_->get_model_replica(model_id, replica_id);
    if (!container) {
      throw std::runtime_error(
          "TaskExecutor failed to find previously registered active "
          "container!");
    }
    boost::shared_lock<boost::shared_mutex> l(model_queues_mutex_);
    auto model_queue_entry = model_queues_.find(container->model_);
    if (model_queue_entry == model_queues_.end()) {
      throw std::runtime_error(
          "Failed to find model queue associated with a previously registered "
          "container!");
    }
    std::shared_ptr<ModelQueue> current_model_queue = model_queue_entry->second;


    // NOTE: It is safe to unlock here because we copy the shared_ptr to
    // the ModelQueue object so even if that entry in the map gets deleted,
    // the ModelQueue object won't be destroyed until our copy of the pointer
    // goes out of scope.
    l.unlock();

    std::vector<PredictTask> batch = current_model_queue->get_batch([container](
        Deadline deadline) { return container->get_batch_size(deadline); });

    // Create a histogram "queue size hist"

    if (batch.size() > 0) {
      // move the lock up here, so that nothing can pull from the
      // inflight_messages_
      // map between the time a message is sent and when it gets inserted
      // into the map
      std::unique_lock<std::mutex> l(inflight_messages_mutex_);
      std::vector<InflightMessage> cur_batch;

      std::vector<PredictTask> batch_tasks;
      std::chrono::time_point<std::chrono::system_clock> current_time =
          std::chrono::system_clock::now();
      for (auto b : batch) {
        batch_tasks.push_back(b);
        cur_batch.emplace_back(current_time, container->container_id_, b.model_,
                               container->replica_id_, b.input_, b.query_id_);
      }
      int message_id = rpc_->send_message(construct_batch_message(batch_tasks),
                                          container->container_id_);
      inflight_messages_.emplace(message_id, std::move(cur_batch));
    } else {
      log_error_formatted(
          LOGGING_TAG_TASK_EXECUTOR,
          "ModelQueue returned empty batch for model {}, replica {}",
          model_id.serialize(), std::to_string(replica_id));
    }
  }

  void on_response_recv(rpc::RPCResponse response) {
    std::unique_lock<std::mutex> l(inflight_messages_mutex_);
    int msg_id;
    DataType data_type;
    std::shared_ptr<void> data;
    std::tie(msg_id, data_type, data) = response;

    auto keys = inflight_messages_[msg_id];
    boost::shared_lock<boost::shared_mutex> metrics_lock(model_metrics_mutex_);

    inflight_messages_.erase(msg_id);
    l.unlock();
    rpc::PredictionResponse parsed_response =
        rpc::PredictionResponse::deserialize_prediction_response(data_type,
                                                                 data);
    assert(parsed_response.outputs_.size() == keys.size());
    int batch_size = keys.size();
    throughput_meter_->mark(batch_size);
    std::chrono::time_point<std::chrono::system_clock> current_time =
        std::chrono::system_clock::now();
    if (batch_size > 0) {
      InflightMessage &first_message = keys[0];
      const VersionedModelId &cur_model = first_message.model_;
      const int cur_replica_id = first_message.replica_id_;
      auto task_latency = current_time - first_message.send_time_;
      long task_latency_micros =
          std::chrono::duration_cast<std::chrono::microseconds>(task_latency)
              .count();

      std::shared_ptr<ModelContainer> processing_container =
          active_containers_->get_model_replica(cur_model, cur_replica_id);

      processing_container->update_throughput(batch_size, task_latency_micros);
      processing_container->latency_hist_.insert(task_latency_micros);

      boost::optional<ModelMetrics> cur_model_metric;
      auto cur_model_metric_entry = model_metrics_.find(cur_model);
      if (cur_model_metric_entry != model_metrics_.end()) {
        cur_model_metric = cur_model_metric_entry->second;
      }
      if (cur_model_metric) {
        (*cur_model_metric).throughput_->mark(batch_size);
        (*cur_model_metric).num_predictions_->increment(batch_size);
        (*cur_model_metric).batch_size_->insert(batch_size);
        (*cur_model_metric)
            .latency_->insert(static_cast<int64_t>(task_latency_micros));
      }
      for (int batch_num = 0; batch_num < batch_size; ++batch_num) {
        InflightMessage completed_msg = keys[batch_num];
        cache_->put(completed_msg.model_, completed_msg.query_id_,
                    Output{parsed_response.outputs_[batch_num],
                           {completed_msg.model_}});
      }
    }
  }
};

}  // namespace clipper

#endif  // CLIPPER_LIB_TASK_EXECUTOR_H
