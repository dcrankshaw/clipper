#include <chrono>
#include <clipper/clock.hpp>
#include <clipper/metrics.hpp>
#include <cxxopts.hpp>
#include <random>
#include <string>

using namespace clipper;

class ProfilerMetrics {
 public:
  explicit ProfilerMetrics(std::string name)
      : name_(name),
        latency_(
            clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
                name_ ":prediction_latency", "microseconds", 32768)),
        latency_list_(clipper::metrics::MetricsRegistry::get_metrics()
                          .create_data_list<long long>(
                              name_ ":prediction_latencies", "microseconds")),
        throughput_(
            clipper::metrics::MetricsRegistry::get_metrics().create_meter(
                name_ + ":prediction_throughput")),
        num_predictions_(
            clipper::metrics::MetricsRegistry::get_metrics().create_counter(
                name_ + ":num_predictions")) {}

  ~ProfilerMetrics() = default;

  ProfilerMetrics(const ProfilerMetrics&) = default;

  ProfilerMetrics& operator=(const ProfilerMetrics&) = default;

  ProfilerMetrics(ProfilerMetrics&&) = default;
  ProfilerMetrics& operator=(ProfilerMetrics&&) = default;

  std::string name_;
  std::shared_ptr<clipper::metrics::Histogram> latency_;
  std::shared_ptr<clipper::metrics::DataList<long long>> latency_list_;
  std::shared_ptr<clipper::metrics::Meter> throughput_;
  std::shared_ptr<clipper::metrics::Counter> num_predictions_;
};

void predict(FrontendRPCClient client, std::string name,
             ClientFeatureVector input, ProfilerMetrics metrics,
             std::atomic<int>& prediction_counter) {
  auto start_time = std::chrono::system_clock::now();
  client.send_request(name, input, [metrics, prediction_counter,
                                    start_time](ClientFeatureVector output) {
    if (output.type_ == DataType::Strings) {
      std::string output_str = std::string(
          reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    }
    auto latency = std::chrono::system_clock::now() - start_time;
    long latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.latency_->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_list_->insert(static_cast<int64_t>(latency_micros));
    metrics.throughput_->mark(1);
    metrics.num_predictions_->increment(1);
    prediction_counter.fetch_add(1);

  });
}

std::vector<ClientFeatureVector> generate_float_inputs(int input_length) {
  int num_points = 1000;
  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 generator(
      rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.0, 1.0);
  std::vector<ClientFeatureVector> inputs;
  for (int i = 0; i < num_points; ++i) {
    float* input_buffer =
        reinterpret_cast<float*>(malloc(input_length * sizeof(float)));
    for (int j = 0; j < input_length; ++j) {
      input_buffer[j] = distribution(generator);
    }
    std::shared_ptr<void> input_ptr(reinterpret_cast<void*>(input_buffer),
                                    free);
    inputs.emplace_back(input_ptr, input_length, input_length * sizeof(float),
                        DataType::Floats);
  }
  return inputs;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("profiler", "InferLine profiler");
  // clang-format off
  options.add_options()
      ("name", "Model name",
       cxxopts::value<std::string>())
      ("input_type", "Only \"float\" supported for now.",
       cxxopts::value<std::string>()->default_value("float"))
      ("input_size", "length of each input",
       cxxopts::value<int>())
      ("request_delay_micros", "Request delay in integer microseconds",
       cxxopts::value<int>())
      ("trial_length", "Number of queries per trial",
       cxxopts::value<int>())
      ("num_trials", "Number of trials",
       cxxopts::value<int>())
      ("log_file", "location of log file",
       cxxopts::value<std::string>())
      ("clipper_address", "IP address or hostname of ZMQ frontend",
       cxxopts::value<std::string>())
       ;
  // clang-format on
  options.parse(argc, argv);

  // Request the system uptime so that a clock instance is created as
  // soon as the frontend starts
  clock::ClipperClock::get_clock().get_uptime();
  if (options["input_type"].as<std::string>() == "float") {
    std::vector<ClientFeatureVector> inputs =
        generate_float_inputs(options["input_size"].as<int>());
    std::string name = options["name"].as<std::string>();
    ProfilerMetrics metrics{name};
    auto predict_func = [metrics, name](FrontendRPCClient client,
                                        ClientFeatureVector input,
                                        std::atomic<int>& prediction_counter) {
      predict(client, name, input, metrics, prediction_counter);
    };
    Driver driver(
        predict_func, inputs, options["request_delay_micros"].as<int>(),
        options["trial_length"].as<int>(), options["num_trials"].as<int>(),
        options["log_file"].as<std::string>(),
        options["clipper_address"].as<std::string>());
    driver_start();
  }
}
