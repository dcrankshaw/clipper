#include <cstdlib>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <algorithm>

#include <cxxopts.hpp>

#include <clipper/config.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/redis.hpp>
#include <clipper/rpc_service.hpp>
#include <clipper/task_executor.hpp>
#include <clipper/util.hpp>

using namespace clipper;

const std::string LOGGING_TAG_RPC_BENCH = "RPCBENCH";

// taken from http://stackoverflow.com/a/12468109/814642
std::string gen_random_string(size_t length) {
  std::srand(time(NULL));
  auto randchar = []() -> char {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[std::rand() % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

std::string get_thread_id() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  return ss.str();
}

template <typename T, class N>
std::vector<std::shared_ptr<Input>> get_primitive_inputs(
    int message_size, int input_len, DataType type, std::vector<T> data_vector,
    std::vector<std::shared_ptr<N>> input_vector) {
  input_vector.clear();
  std::vector<std::shared_ptr<Input>> generic_input_vector;
  for (int k = 0; k < message_size; ++k) {
    for (int j = 0; j < input_len; ++j) {
      if (type == DataType::Bytes) {
        uint8_t *bytes = reinterpret_cast<uint8_t *>(&j);
        for (int i = 0; i < (int)(sizeof(int) / sizeof(uint8_t)); i++) {
          data_vector.push_back(*(bytes + i));
        }
      } else {
        data_vector.push_back(static_cast<T>(j));
      }
    }
    std::shared_ptr<T> input_data(
        static_cast<T *>(malloc(sizeof(T) * data_vector.size())), free);
    memcpy(input_data.get(), data_vector.data(),
           data_vector.size() * sizeof(T));

    std::shared_ptr<N> input =
        std::make_shared<N>(input_data, data_vector.size());
    generic_input_vector.push_back(std::dynamic_pointer_cast<Input>(input));
    data_vector.clear();
  }
  return generic_input_vector;
}

rpc::PredictionRequest generate_bytes_request(int message_size) {
  std::vector<uint8_t> type_vec;
  std::vector<std::shared_ptr<ByteVector>> input_vec;
  std::vector<std::shared_ptr<Input>> inputs = get_primitive_inputs(
      message_size, 784, DataType::Bytes, type_vec, input_vec);
  rpc::PredictionRequest request(inputs, DataType::Bytes);
  return request;
}

rpc::PredictionRequest generate_floats_request(int message_size) {
  std::vector<float> type_vec;
  std::vector<std::shared_ptr<FloatVector>> input_vec;
  std::vector<std::shared_ptr<Input>> inputs = get_primitive_inputs(
      message_size, 784, DataType::Floats, type_vec, input_vec);
  rpc::PredictionRequest request(inputs, DataType::Floats);
  return request;
}

rpc::PredictionRequest generate_ints_request(int message_size) {
  std::vector<int> type_vec;
  std::vector<std::shared_ptr<IntVector>> input_vec;
  std::vector<std::shared_ptr<Input>> inputs = get_primitive_inputs(
      message_size, 784, DataType::Ints, type_vec, input_vec);
  rpc::PredictionRequest request(inputs, DataType::Ints);
  return request;
}

rpc::PredictionRequest generate_doubles_request(int message_size) {
  std::vector<double> type_vec;
  std::vector<std::shared_ptr<DoubleVector>> input_vec;
  std::vector<std::shared_ptr<Input>> inputs = get_primitive_inputs(
      message_size, 784, DataType::Doubles, type_vec, input_vec);
  rpc::PredictionRequest request(inputs, DataType::Doubles);
  return request;
}

rpc::PredictionRequest generate_string_request(int message_size) {
  rpc::PredictionRequest request(DataType::Strings);
  for (int i = 0; i < message_size; ++i) {
    std::string str = gen_random_string(150);
    std::shared_ptr<char> input_data(
        static_cast<char *>(malloc(str.length() * sizeof(char))), free);
    memcpy(input_data.get(), str.data(), str.size());
    std::shared_ptr<SerializableString> input =
        std::make_shared<SerializableString>(input_data, str.size());
    request.add_input(input);
  }
  return request;
}

rpc::PredictionRequest create_request(DataType input_type, int message_size) {
  switch (input_type) {
    case DataType::Strings: return generate_string_request(message_size);
    case DataType::Doubles: return generate_doubles_request(message_size);
    case DataType::Floats: return generate_floats_request(message_size);
    case DataType::Bytes: return generate_bytes_request(message_size);
    case DataType::Ints: return generate_ints_request(message_size);
    case DataType::Invalid:
    default: throw std::invalid_argument("Unsupported input type");
  }
}

class SerialBenchmarker {
 public:
  SerialBenchmarker(int num_messages, int message_size, DataType input_type)
      : num_messages_(num_messages),
        rpc_(std::make_unique<rpc::RPCService>()),
        request_(create_request(input_type, message_size)) {}

  void start() {
    rpc_->start("*", RPC_SERVICE_PORT, [](VersionedModelId, int) {},
                [this](rpc::RPCResponse response) {
                  on_response_recv(std::move(response));
                });

    msg_latency_hist_ =
        metrics::MetricsRegistry::get_metrics().create_histogram(
            "rpc_bench_msg_latency", "milliseconds", 8260);
    throughput_meter_ = metrics::MetricsRegistry::get_metrics().create_meter(
        "rpc_bench_throughput");

    Config &conf = get_config();
    while (!redis_connection_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_RPC_BENCH, "RPCBench failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_RPC_BENCH,
                "RPCBench subscriber failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    redis::send_cmd_no_reply<std::string>(
        redis_connection_, {"CONFIG", "SET", "notify-keyspace-events", "AKE"});
    redis::subscribe_to_container_changes(
        redis_subscriber_,
        // event_type corresponds to one of the Redis event types
        // documented in https://redis.io/topics/notifications.
        [this](const std::string &key, const std::string &event_type) {
          if (event_type == "hset") {
            auto container_info =
                redis::get_container_by_key(redis_connection_, key);
            benchmark_container_id_ =
                std::stoi(container_info["zmq_connection_id"]);

            // SEND FIRST MESSAGE
            send_message();
          }

        });
  }

  SerialBenchmarker(const SerialBenchmarker &other) = delete;
  SerialBenchmarker &operator=(const SerialBenchmarker &other) = delete;

  SerialBenchmarker(SerialBenchmarker &&other) = default;
  SerialBenchmarker &operator=(SerialBenchmarker &&other) = default;
  ~SerialBenchmarker() {
    std::unique_lock<std::mutex> l(bench_completed_cv_mutex_);
    bench_completed_cv_.wait(l, [this]() { return bench_completed_ == true; });
  }

  void send_message() {
    cur_msg_start_time_millis_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    cur_message_id_ =
        rpc_->send_message(request_.serialize(), benchmark_container_id_);
  }

  void on_response_recv(rpc::RPCResponse response) {
    long recv_time_millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    if (std::get<0>(response) != cur_message_id_) {
      std::stringstream ss;
      ss << "Response message ID ";
      ss << std::get<0>(response);
      ss << " did not match in flight message ID ";
      ss << cur_message_id_;
      throw std::logic_error(ss.str());
    }
    long message_duration_millis =
        recv_time_millis - cur_msg_start_time_millis_;
    msg_latency_hist_->insert(message_duration_millis);
    throughput_meter_->mark(1);
    messages_completed_ += 1;

    if (messages_completed_ < num_messages_) {
      send_message();
    } else {
      bench_completed_ = true;
      bench_completed_cv_.notify_all();
    }
  }

  std::condition_variable_any bench_completed_cv_;
  std::mutex bench_completed_cv_mutex_;
  std::atomic<bool> bench_completed_{false};

 private:
  std::shared_ptr<metrics::Histogram> msg_latency_hist_;
  std::shared_ptr<metrics::Meter> throughput_meter_;
  int num_messages_;
  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
  std::atomic<int> messages_completed_{0};
  std::unique_ptr<rpc::RPCService> rpc_;
  std::atomic<long> cur_msg_start_time_millis_;
  std::atomic<int> benchmark_container_id_;
  rpc::PredictionRequest request_;
  int cur_message_id_;
};

class ParallelBenchmarker {
 public:
  ParallelBenchmarker(int message_size_inputs, int num_containers, DataType input_type)
    : rpc_(std::make_unique<rpc::RPCService>()),
      message_size_inputs_(message_size_inputs),
      num_containers_(num_containers),
      serialized_request_(create_request(input_type, message_size_inputs).serialize()),
      active_(true) {}

  void start() {
    rpc_->start("*", RPC_SERVICE_PORT, [](VersionedModelId, int) {},
                [](rpc::RPCResponse response) {});

    msg_latency_hist_ =
        metrics::MetricsRegistry::get_metrics().create_histogram(
            "rpc_bench_msg_latency", "milliseconds", 8260);
    throughput_meter_ = metrics::MetricsRegistry::get_metrics().create_meter(
        "rpc_bench_throughput");

    recv_thread_ = std::thread([this]() {
      recv_messages();
    });

    Config &conf = get_config();
    while (!redis_connection_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_RPC_BENCH, "RPCBench failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_RPC_BENCH,
                "RPCBench subscriber failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    redis::send_cmd_no_reply<std::string>(
        redis_connection_, {"CONFIG", "SET", "notify-keyspace-events", "AKE"});
    redis::subscribe_to_container_changes(
        redis_subscriber_,
        // event_type corresponds to one of the Redis event types
        // documented in https://redis.io/topics/notifications.
        [this](const std::string &key, const std::string &event_type) {
          if (event_type == "hset") {
            auto container_info =
                redis::get_container_by_key(redis_connection_, key);
            int benchmark_container_id =
                std::stoi(container_info["zmq_connection_id"]);

            if(benchmark_container_ids.size() >= num_containers_) {
              return;
            }

            auto container_id_search =
                std::find(benchmark_container_ids.begin(), benchmark_container_ids.end(), benchmark_container_id);
            if(container_id_search == benchmark_container_ids.end()) {
              benchmark_container_ids.push_back(benchmark_container_id);
            }

            if(benchmark_container_ids.size() == num_containers_) {
              send_thread_ = std::thread([this]() {
                send_messages();
              });
            }
          }
        });
  }

  void stop() {
    active_ = false;
    send_thread_.join();
    recv_thread_.join();

  }

 private:
  void recv_messages() {
    while(active_) {
      std::vector<rpc::RPCResponse> responses = rpc_->try_get_responses(10000);
      if(responses.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        throughput_meter_->mark(responses.size() * message_size_inputs_);
      }
    }
  }

  void send_messages() {
    while(active_) {
      rpc_->send_message(serialized_request_, benchmark_container_ids[last_sent_container_index]);
      last_sent_container_index = (last_sent_container_index + 1) % num_containers_;
      std::this_thread::sleep_for(std::chrono::microseconds(250));
    }
  }

  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
  std::unique_ptr<rpc::RPCService> rpc_;
  int message_size_inputs_;
  int num_containers_;
  int last_sent_container_index = 0;
  std::vector<ByteBuffer> serialized_request_;
  std::shared_ptr<metrics::Histogram> msg_latency_hist_;
  std::shared_ptr<metrics::Meter> throughput_meter_;
  std::atomic_bool active_;
  std::vector<int> benchmark_container_ids;
  std::thread send_thread_;
  std::thread recv_thread_;
};

void run_serial_benchmarker(cxxopts::Options& options) {
  DataType input_type =
      clipper::parse_input_type(options["input_type"].as<std::string>());
  SerialBenchmarker SerialBenchmarker(options["num_messages"].as<int>(),
                                      options["message_size"].as<int>(), input_type);
  SerialBenchmarker.start();
  std::unique_lock<std::mutex> l(SerialBenchmarker.bench_completed_cv_mutex_);
  SerialBenchmarker.bench_completed_cv_.wait(
      l, [&SerialBenchmarker]() { return SerialBenchmarker.bench_completed_ == true; });
  metrics::MetricsRegistry &registry = metrics::MetricsRegistry::get_metrics();
  std::string metrics_report = registry.report_metrics();
  log_info(LOGGING_TAG_RPC_BENCH, "METRICS", metrics_report);
}

void run_parallel_benchmarker(cxxopts::Options& options) {
  DataType input_type =
      clipper::parse_input_type(options["input_type"].as<std::string>());
  ParallelBenchmarker parallel_benchmarker(
      options["message_size"].as<int>(),
      options["num_containers"].as<int>(),
      input_type);
  parallel_benchmarker.start();
  std::this_thread::sleep_for(std::chrono::seconds(20));
  parallel_benchmarker.stop();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  metrics::MetricsRegistry &registry = metrics::MetricsRegistry::get_metrics();
  std::string metrics_report = registry.report_metrics();
  log_info(LOGGING_TAG_RPC_BENCH, "METRICS", metrics_report);
}

int main(int argc, char *argv[]) {
  cxxopts::Options options("rpc_bench", "Clipper RPC Benchmark");
  // clang-format off
  options.add_options()
    ("redis_ip", "Redis address",
        cxxopts::value<std::string>()->default_value("localhost"))
    ("redis_port", "Redis port",
        cxxopts::value<int>()->default_value("6379"))
    ("m,num_messages", "Number of messages to send",
        cxxopts::value<int>()->default_value("100"))
    ("s,message_size", "Number of inputs per message",
        cxxopts::value<int>()->default_value("500"))
    ("c,num_containers", "Expected number of containers",
        cxxopts::value<int>()->default_value("1"))
    ("input_type", "Can be bytes, ints, floats, doubles, or strings",
        cxxopts::value<std::string>()->default_value("doubles"));
  // clang-format on
  options.parse(argc, argv);

  clipper::Config &conf = clipper::get_config();
  conf.set_redis_address(options["redis_ip"].as<std::string>());
  conf.set_redis_port(options["redis_port"].as<int>());
  conf.set_rpc_max_send(10);
  conf.set_rpc_max_recv(100);
  conf.ready();

  //run_serial_benchmarker(options);
  run_parallel_benchmarker(options);
  return 0;
}
