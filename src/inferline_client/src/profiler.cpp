#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <iostream>

#include <cxxopts.hpp>

#include <clipper/clock.hpp>
#include <clipper/metrics.hpp>

#include "driver.hpp"
#include "zmq_client.hpp"

using namespace clipper;
using namespace zmq_client;

class ProfilerMetrics {
 public:
  explicit ProfilerMetrics(std::string name)
      : name_(name),
        latency_(
            clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
                name_ + ":prediction_latency", "microseconds", 32768)),
        latency_list_(clipper::metrics::MetricsRegistry::get_metrics()
                          .create_data_list<long long>(
                              name_ + ":prediction_latencies", "microseconds")),
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

void predict(FrontendRPCClient& client, std::string name,
             ClientFeatureVector input, ProfilerMetrics metrics,
             std::atomic<int>& prediction_counter,
             std::ofstream& query_lineage_file,
             std::mutex& query_file_mutex) {
  auto start_time = std::chrono::system_clock::now();
  client.send_request(name, input, [metrics, &prediction_counter, start_time,
      &query_lineage_file, &query_file_mutex](
                                       ClientFeatureVector output,
                                       std::shared_ptr<QueryLineage> lineage) {
    if (output.type_ == DataType::Strings) {
      std::string output_str = std::string(
          reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - start_time;
    long latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.latency_->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_list_->insert(static_cast<int64_t>(latency_micros));
    metrics.throughput_->mark(1);
    metrics.num_predictions_->increment(1);
    prediction_counter += 1;
    lineage->add_timestamp("driver::send", 
          std::chrono::duration_cast<std::chrono::microseconds>(
              start_time.time_since_epoch())
              .count());

    lineage->add_timestamp("driver::recv", 
          std::chrono::duration_cast<std::chrono::microseconds>(
              cur_time.time_since_epoch())
              .count());

    std::unique_lock<std::mutex> lock;
    query_lineage_file << "{";
    int num_entries = lineage->get_timestamps().size();
    int idx = 0;
    for (auto& entry : lineage->get_timestamps()) {
      query_lineage_file << "\"" << entry.first << "\": " << std::to_string(entry.second);
      if (idx < num_entries - 1) {
        query_lineage_file << ", ";
      }
      idx += 1;
    }
    query_lineage_file << "}" << std::endl;
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

// void spin_sleep(int duration_micros) {
//   auto start_time = std::chrono::system_clock::now();
//   long cur_delay_micros = 0;
//   while (cur_delay_micros < duration_micros) {
//     auto cur_delay = std::chrono::system_clock::now() - start_time;
//     cur_delay_micros =
//         std::chrono::duration_cast<std::chrono::microseconds>(cur_delay).count();
//   }
// }
//
// void timer_test(int sleep_time, int num_trials) {
//
//   struct timespec req = {0};
//   req.tv_sec = 0;
//   req.tv_nsec = sleep_time * 1000;
//   std::vector<long> times;
//   for (size_t i = 0; i < num_trials; ++i) {
//     auto start_time = std::chrono::system_clock::now();
//     // nanosleep(&req, (struct timespec *)NULL);
//     spin_sleep(sleep_time);
//     auto latency = std::chrono::system_clock::now() - start_time;
//     long latency_micros =
//       std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
//     times.push_back(latency_micros);
//   }
//   long sum = 0;
//   for (int i = 0; i < times.size(); ++i) {
//     sum += times[i];
//   }
//   double mean = ((double) sum) / ((double) times.size());
//   double sum_of_diffs = 0.0;
//   for (int i = 0; i < times.size(); ++i) {
//     double diff = ((double) times[i]) - mean;
//     sum_of_diffs += diff * diff;
//   }
//   double stdev = std::sqrt(1.0 / ((double) times.size() - 1.0) *
//   sum_of_diffs);
//   std::cout << "Sleep time: " << std::to_string(sleep_time) << ": " <<
//   std::to_string(mean)
//     << " +- " << std::to_string(stdev) << ". Diff: " << std::to_string(mean -
//     sleep_time) << std::endl;
//
// }

int main(int argc, char* argv[]) {
  // size_t num_trials = 2000;
  // std::vector<int> sleep_times = {1000, 2000, 3000, 4000, 5000, 6000, 7000,
  // 8000, 9000, 10000, 15000,
  //   20000, 30000, 40000, 50000, 75000, 100000};
  // for (int i = 0; i < sleep_times.size(); ++i) {
  //   timer_test(sleep_times[i], num_trials);
  // }
  // return 0;
  // ///////////////////////////////////////////////////////////

  cxxopts::Options options("profiler", "InferLine profiler");
  // clang-format off
  options.add_options()
      ("name", "Model name",
       cxxopts::value<std::string>())
      ("input_type", "Only \"float\" supported for now.",
       cxxopts::value<std::string>()->default_value("float"))
      ("input_size", "length of each input",
       cxxopts::value<int>())
      // ("request_delay_micros", "Request delay in integer microseconds",
      //  cxxopts::value<int>())
      ("target_throughput", "Mean throughput to target in qps",
       cxxopts::value<float>())
      ("request_distribution", "Distribution to sample request delay from. "
       "Can be 'constant', 'poisson', or 'batch'. 'batch' sends a single batch at a time.",
       cxxopts::value<std::string>())
      ("trial_length", "Number of queries per trial",
       cxxopts::value<int>())
      ("num_trials", "Number of trials",
       cxxopts::value<int>())
      ("batch_size", "Batch size",
       cxxopts::value<int>()->default_value(-1))
      ("log_file", "location of log file",
       cxxopts::value<std::string>())
      ("clipper_address", "IP address or hostname of ZMQ frontend",
       cxxopts::value<std::string>())
       ;
  // clang-format on
  options.parse(argc, argv);
  std::string distribution = options["request_distribution"].as<std::string>();
  if (!(distribution == "poisson" || distribution == "constant" || distribution == "batch")) {
    std::cerr << "Invalid distribution: " << distribution << std::endl;
    return 1;
  }

  // Request the system uptime so that a clock instance is created as
  // soon as the frontend starts
  clock::ClipperClock::get_clock().get_uptime();
  if (options["input_type"].as<std::string>() == "floats") {
    std::vector<ClientFeatureVector> inputs =
        generate_float_inputs(options["input_size"].as<int>());
    std::string name = options["name"].as<std::string>();
    ProfilerMetrics metrics{name};

    std::ofstream query_lineage_file;
    std::mutex query_file_mutex;
    query_lineage_file.open(options["log_file"].as<std::string>() + "-query_lineage.txt");
    auto predict_func = [metrics, name, &query_lineage_file, &query_file_mutex](FrontendRPCClient& client,
                                        ClientFeatureVector input,
                                        std::atomic<int>& prediction_counter) {
      predict(client, name, input, metrics, prediction_counter,
          query_lineage_file, query_file_mutex);
    };
    Driver driver(predict_func, std::move(inputs),
                  options["target_throughput"].as<float>(),
                  distribution,
                  options["trial_length"].as<int>(),
                  options["num_trials"].as<int>(),
                  options["log_file"].as<std::string>(),
                  options["clipper_address"].as<std::string>(),
                  options["batch_size"].as<int>());
    std::cout << "Starting driver" << std::endl;
    driver.start();
    std::cout << "Driver completed" << std::endl;
    query_lineage_file.close();
    return 0;
  } else {
    std::cerr << "Invalid input type "
              << options["input_type"].as<std::string>() << std::endl;
    return 1;
  }
}
