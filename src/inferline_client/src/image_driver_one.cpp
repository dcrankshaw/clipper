#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <cxxopts.hpp>

#include <clipper/clock.hpp>
#include <clipper/metrics.hpp>

#include "driver.hpp"
#include "zmq_client.hpp"

using namespace clipper;
using namespace zmq_client;

static const std::string TF_RESNET = "tf-resnet-feats";
static const std::string TF_KERNEL_SVM = "tf-kernel-svm";

static const std::string INCEPTION_FEATS = "inception";
static const std::string TF_LOG_REG = "tf-log-reg";

class ImageDriverOneMetrics {
 public:
  explicit ImageDriverOneMetrics()
      : e2e_latency_(clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            "e2e:prediction_latency", "microseconds", 32768)),
        resnet_latency_(clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            TF_RESNET + ":prediction_latency", "microseconds", 32768)),
        inception_latency_(clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            INCEPTION_FEATS + ":prediction_latency", "microseconds", 32768)),
        ksvm_latency_(clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            TF_KERNEL_SVM + ":prediction_latency", "microseconds", 32768)),
        log_reg_latency_(clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            TF_LOG_REG + ":prediction_latency", "microseconds", 32768)),
        e2e_latency_list_(
            clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
                "e2e:prediction_latencies", "microseconds")),
        resnet_latency_list_(
            clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
                TF_RESNET + ":prediction_latencies", "microseconds")),
        inception_latency_list_(
            clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
                INCEPTION_FEATS + ":prediction_latencies", "microseconds")),
        ksvm_latency_list_(
            clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
                TF_KERNEL_SVM + ":prediction_latencies", "microseconds")),
        log_reg_latency_list_(
            clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
                TF_LOG_REG + ":prediction_latencies", "microseconds")),
        e2e_throughput_(clipper::metrics::MetricsRegistry::get_metrics().create_meter(
            "e2e:prediction_throughput")),
        resnet_throughput_(clipper::metrics::MetricsRegistry::get_metrics().create_meter(
            TF_RESNET + ":prediction_throughput")),
        inception_throughput_(clipper::metrics::MetricsRegistry::get_metrics().create_meter(
            INCEPTION_FEATS + ":prediction_throughput")),
        ksvm_throughput_(clipper::metrics::MetricsRegistry::get_metrics().create_meter(
            TF_KERNEL_SVM + ":prediction_throughput")),
        log_reg_throughput_(clipper::metrics::MetricsRegistry::get_metrics().create_meter(
            TF_LOG_REG + ":prediction_throughput")),
        e2e_num_predictions_(
            clipper::metrics::MetricsRegistry::get_metrics().create_counter("e2e:num_predictions")),
        resnet_num_predictions_(clipper::metrics::MetricsRegistry::get_metrics().create_counter(
            TF_RESNET + ":num_predictions")),
        inception_num_predictions_(clipper::metrics::MetricsRegistry::get_metrics().create_counter(
            INCEPTION_FEATS + ":num_predictions")),
        ksvm_num_predictions_(clipper::metrics::MetricsRegistry::get_metrics().create_counter(
            TF_KERNEL_SVM + ":num_predictions")),
        log_reg_num_predictions_(clipper::metrics::MetricsRegistry::get_metrics().create_counter(
            TF_LOG_REG + ":num_predictions")) {}

  ~ImageDriverOneMetrics() = default;

  ImageDriverOneMetrics(const ImageDriverOneMetrics&) = default;

  ImageDriverOneMetrics& operator=(const ImageDriverOneMetrics&) = default;

  ImageDriverOneMetrics(ImageDriverOneMetrics&&) = default;
  ImageDriverOneMetrics& operator=(ImageDriverOneMetrics&&) = default;

  std::string name_ = "image_driver_one";
  std::shared_ptr<clipper::metrics::Histogram> e2e_latency_;
  std::shared_ptr<clipper::metrics::Histogram> resnet_latency_;
  std::shared_ptr<clipper::metrics::Histogram> inception_latency_;
  std::shared_ptr<clipper::metrics::Histogram> ksvm_latency_;
  std::shared_ptr<clipper::metrics::Histogram> log_reg_latency_;

  std::shared_ptr<clipper::metrics::DataList<long long>> e2e_latency_list_;
  std::shared_ptr<clipper::metrics::DataList<long long>> resnet_latency_list_;
  std::shared_ptr<clipper::metrics::DataList<long long>> inception_latency_list_;
  std::shared_ptr<clipper::metrics::DataList<long long>> ksvm_latency_list_;
  std::shared_ptr<clipper::metrics::DataList<long long>> log_reg_latency_list_;

  std::shared_ptr<clipper::metrics::Meter> e2e_throughput_;
  std::shared_ptr<clipper::metrics::Meter> resnet_throughput_;
  std::shared_ptr<clipper::metrics::Meter> inception_throughput_;
  std::shared_ptr<clipper::metrics::Meter> ksvm_throughput_;
  std::shared_ptr<clipper::metrics::Meter> log_reg_throughput_;

  std::shared_ptr<clipper::metrics::Counter> e2e_num_predictions_;
  std::shared_ptr<clipper::metrics::Counter> resnet_num_predictions_;
  std::shared_ptr<clipper::metrics::Counter> inception_num_predictions_;
  std::shared_ptr<clipper::metrics::Counter> ksvm_num_predictions_;
  std::shared_ptr<clipper::metrics::Counter> log_reg_num_predictions_;
};

void predict(FrontendRPCClient& client, ClientFeatureVector input, ImageDriverOneMetrics metrics,
             std::atomic<int>& prediction_counter,
             std::unordered_map<std::string, std::ofstream>& lineage_file_map,
             std::unordered_map<std::string, std::mutex>& lineage_mutex_map) {
  auto start_time = std::chrono::system_clock::now();
  size_t resnet_input_length = 224 * 224 * 3;
  ClientFeatureVector resnet_input(input.data_, resnet_input_length,
                                   resnet_input_length * sizeof(float), DataType::Floats);

  std::shared_ptr<std::atomic_int> branches_completed = std::make_shared<std::atomic_int>(0);
  auto completion_callback = [metrics, &prediction_counter, start_time]() {
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.e2e_latency_->insert(static_cast<int64_t>(latency_micros));
    metrics.e2e_latency_list_->insert(static_cast<int64_t>(latency_micros));
    metrics.e2e_throughput_->mark(1);
    metrics.e2e_num_predictions_->increment(1);
    prediction_counter += 1;
  };

  auto ksvm_callback = [&client, metrics, branches_completed, completion_callback,
                        &lineage_file_map, &lineage_mutex_map](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage,
      std::chrono::time_point<std::chrono::system_clock> request_start_time) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - request_start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.ksvm_latency_->insert(static_cast<int64_t>(latency_micros));
    metrics.ksvm_latency_list_->insert(static_cast<int64_t>(latency_micros));
    metrics.ksvm_throughput_->mark(1);
    metrics.ksvm_num_predictions_->increment(1);

    lineage->add_timestamp(
        "driver::send",
        std::chrono::duration_cast<std::chrono::microseconds>(request_start_time.time_since_epoch())
            .count());

    lineage->add_timestamp(
        "driver::recv",
        std::chrono::duration_cast<std::chrono::microseconds>(cur_time.time_since_epoch()).count());
    std::unique_lock<std::mutex> lock(lineage_mutex_map[TF_KERNEL_SVM]);
    auto& query_lineage_file = lineage_file_map[TF_KERNEL_SVM];
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

    int num_branches_completed = branches_completed->fetch_add(1);
    if (num_branches_completed == 1) {
      completion_callback();
    }
  };

  auto log_reg_callback = [&client, metrics, branches_completed, &completion_callback,
                           &lineage_file_map, &lineage_mutex_map](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage,
      std::chrono::time_point<std::chrono::system_clock> request_start_time) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - request_start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.log_reg_latency_->insert(static_cast<int64_t>(latency_micros));
    metrics.log_reg_latency_list_->insert(static_cast<int64_t>(latency_micros));
    metrics.log_reg_throughput_->mark(1);
    metrics.log_reg_num_predictions_->increment(1);

    lineage->add_timestamp(
        "driver::send",
        std::chrono::duration_cast<std::chrono::microseconds>(request_start_time.time_since_epoch())
            .count());

    lineage->add_timestamp(
        "driver::recv",
        std::chrono::duration_cast<std::chrono::microseconds>(cur_time.time_since_epoch()).count());
    std::unique_lock<std::mutex> lock(lineage_mutex_map[TF_LOG_REG]);
    auto& query_lineage_file = lineage_file_map[TF_LOG_REG];
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

    int num_branches_completed = branches_completed->fetch_add(1);
    if (num_branches_completed == 1) {
      completion_callback();
    }
  };

  auto inception_callback = [&client, metrics, start_time, log_reg_callback, &lineage_file_map,
                             &lineage_mutex_map](ClientFeatureVector output,
                                                 std::shared_ptr<QueryLineage> lineage) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    client.send_request(TF_LOG_REG, output,
                        [cur_time, log_reg_callback](ClientFeatureVector output,
                                                     std::shared_ptr<QueryLineage> lineage) {
                          log_reg_callback(output, lineage, cur_time);
                        });
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.inception_latency_->insert(static_cast<int64_t>(latency_micros));
    metrics.inception_latency_list_->insert(static_cast<int64_t>(latency_micros));
    metrics.inception_throughput_->mark(1);
    metrics.inception_num_predictions_->increment(1);

    lineage->add_timestamp(
        "driver::send",
        std::chrono::duration_cast<std::chrono::microseconds>(start_time.time_since_epoch())
            .count());

    lineage->add_timestamp(
        "driver::recv",
        std::chrono::duration_cast<std::chrono::microseconds>(cur_time.time_since_epoch()).count());
    std::unique_lock<std::mutex> lock(lineage_mutex_map[INCEPTION_FEATS]);
    auto& query_lineage_file = lineage_file_map[INCEPTION_FEATS];
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
  };

  auto resnet_callback = [&client, metrics, start_time, ksvm_callback, &lineage_file_map,
                          &lineage_mutex_map](ClientFeatureVector output,
                                              std::shared_ptr<QueryLineage> lineage) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    client.send_request(TF_KERNEL_SVM, output,
                        [cur_time, ksvm_callback](ClientFeatureVector output,
                                                  std::shared_ptr<QueryLineage> lineage) {
                          ksvm_callback(output, lineage, cur_time);
                        });
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.resnet_latency_->insert(static_cast<int64_t>(latency_micros));
    metrics.resnet_latency_list_->insert(static_cast<int64_t>(latency_micros));
    metrics.resnet_throughput_->mark(1);
    metrics.resnet_num_predictions_->increment(1);

    lineage->add_timestamp(
        "driver::send",
        std::chrono::duration_cast<std::chrono::microseconds>(start_time.time_since_epoch())
            .count());

    lineage->add_timestamp(
        "driver::recv",
        std::chrono::duration_cast<std::chrono::microseconds>(cur_time.time_since_epoch()).count());
    std::unique_lock<std::mutex> lock(lineage_mutex_map[TF_RESNET]);
    auto& query_lineage_file = lineage_file_map[TF_RESNET];
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
  };

  client.send_request(INCEPTION_FEATS, input, inception_callback);
  client.send_request(TF_RESNET, resnet_input, resnet_callback);
}

std::vector<ClientFeatureVector> generate_float_inputs(int input_length) {
  int num_points = 1000;
  std::random_device rd;         // Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.0, 1.0);
  std::vector<ClientFeatureVector> inputs;
  for (int i = 0; i < num_points; ++i) {
    float* input_buffer = reinterpret_cast<float*>(malloc(input_length * sizeof(float)));
    for (int j = 0; j < input_length; ++j) {
      input_buffer[j] = distribution(generator);
    }
    std::shared_ptr<void> input_ptr(reinterpret_cast<void*>(input_buffer), free);
    inputs.emplace_back(input_ptr, input_length, input_length * sizeof(float), DataType::Floats);
  }
  return inputs;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("image_driver_one", "Image Driver One");
  // clang-format off
  options.add_options()
      ("target_throughput", "Mean throughput to target in qps",
       cxxopts::value<float>())
      ("request_distribution", "Distribution to sample request delay from. "
       "Can be 'constant', 'poisson', or 'batch', or 'file'. 'batch' sends a single batch at a time."
       "'file' uses the delays provided in the request_delay_file argument.",
       cxxopts::value<std::string>())
      ("trial_length", "Number of queries per trial",
       cxxopts::value<int>())
      ("num_trials", "Number of trials",
       cxxopts::value<int>())
      ("log_file", "location of log file",
       cxxopts::value<std::string>())
      ("clipper_address", "IP address or hostname of ZMQ frontend",
       cxxopts::value<std::string>())
      ("request_delay_file", "Path to file containing a list of inter-arrival delays, one per line.",
       cxxopts::value<std::string>())
       ;
  // clang-format on
  options.parse(argc, argv);
  std::string distribution = options["request_distribution"].as<std::string>();
  if (!(distribution == "poisson" || distribution == "constant"
        || distribution == "batch" || distribution == "file")) {
    std::cerr << "Invalid distribution: " << distribution << std::endl;
    return 1;
  }

  // Request the system uptime so that a clock instance is created as
  // soon as the frontend starts
  clock::ClipperClock::get_clock().get_uptime();
  std::vector<ClientFeatureVector> inputs = generate_float_inputs(299 * 299 * 3);
  ImageDriverOneMetrics metrics;

  std::vector<std::string> models = {TF_RESNET, INCEPTION_FEATS, TF_KERNEL_SVM, TF_LOG_REG};
  std::unordered_map<std::string, std::ofstream> lineage_file_map;
  // std::unordered_map<std::string, std::ofstream&> lineage_file_map_refs;
  std::unordered_map<std::string, std::mutex> lineage_mutex_map;
  // std::unordered_map<std::string, std::mutex&> lineage_mutex_map_refs;

  for (auto model : models) {
    lineage_file_map.emplace(model, std::ofstream{});
    lineage_file_map[model].open(options["log_file"].as<std::string>() + "-" + model +
                            "-query_lineage.txt");
    // lineage_file_map_refs.emplace(model, lineage_file_map[model]);
    lineage_mutex_map.emplace(std::piecewise_construct, std::make_tuple(model), std::make_tuple());
    // lineage_mutex_map_refs.emplace(model, lineage_mutex_map[model]);
  }

  auto predict_func = [metrics, &lineage_file_map, &lineage_mutex_map](
      FrontendRPCClient& client, ClientFeatureVector input, std::atomic<int>& prediction_counter) {
    predict(client, input, metrics, prediction_counter, lineage_file_map, lineage_mutex_map);
  };
  std::vector<float> delays_ms;
  if (distribution == "file") {
    std::ifstream delay_file_stream(options["request_delay_file"].as<std::string>());
    std::string line;
    while (std::getline(delay_file_stream, line)) {
      delays_ms.push_back(std::stof(line));
    }
    delay_file_stream.close();
    std::cout << "Loaded delays file: " << std::to_string(delays_ms.size()) << " lines" << std::endl;
  }
  Driver driver(predict_func, std::move(inputs), options["target_throughput"].as<float>(),
                distribution, options["trial_length"].as<int>(), options["num_trials"].as<int>(),
                options["log_file"].as<std::string>(), options["clipper_address"].as<std::string>(),
                -1, delays_ms);
  std::cout << "Starting driver" << std::endl;
  driver.start();
  std::cout << "Driver completed" << std::endl;
  for (auto model : models) {
    lineage_file_map[model].close();
  }
  return 0;
}
