#include <algorithm>
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

static const std::string LANG_DETECT_MODEL_APP_NAME = "tf-lang-detect";
static const std::string NMT_MODEL_APP_NAME = "tf-nmt";
static const std::string LSTM_MODEL_APP_NAME = "tf-lstm";

static const std::string LANG_DETECT_IMAGE_NAME = "model-comp/tf-lang-detect";
static const std::string NMT_IMAGE_NAME = "model-comp/tf-nmt";
static const std::string LSTM_IMAGE_NAME = "model-comp/tf-lstm";

std::vector<std::string> MODEL_NAMES{LANG_DETECT_MODEL_APP_NAME, LSTM_MODEL_APP_NAME,
                                     NMT_MODEL_APP_NAME};

static const std::string LANG_DETECT_WORKLOAD_RELATIVE_PATH = "lang_detect_workload";

static const std::string LANG_CLASSIFICATION_ENGLISH = "en";
static const std::string LANG_CLASSIFICATION_GERMAN = "de";

class ModelMetrics {
 public:
  explicit ModelMetrics(const std::string& model_name)
      : model_latency_(clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            model_name + ":prediction_latency", "microseconds", 32768)),
        model_latency_list_(
            clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
                model_name + ":prediction_latencies", "microseconds")),
        model_throughput_(clipper::metrics::MetricsRegistry::get_metrics().create_meter(
            model_name + ":prediction_throughput")),
        model_num_predictions_(clipper::metrics::MetricsRegistry::get_metrics().create_counter(
            model_name + ":num_predictions")) {}

  std::shared_ptr<clipper::metrics::Histogram> model_latency_;
  std::shared_ptr<clipper::metrics::DataList<long long>> model_latency_list_;
  std::shared_ptr<clipper::metrics::Meter> model_throughput_;
  std::shared_ptr<clipper::metrics::Counter> model_num_predictions_;
};

class TextDriverOneMetrics {
 public:
  explicit TextDriverOneMetrics()
      : e2e_latency_(clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            "e2e:prediction_latency", "microseconds", 32768)),
        e2e_latency_list_(
            clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
                "e2e:prediction_latencies", "microseconds")),
        e2e_throughput_(clipper::metrics::MetricsRegistry::get_metrics().create_meter(
            "e2e:prediction_throughput")),
        e2e_num_predictions_(clipper::metrics::MetricsRegistry::get_metrics().create_counter(
            "e2e:num_predictions")) {
    for (const std::string& model_name : MODEL_NAMES) {
      std::shared_ptr<ModelMetrics> metrics_item = std::make_shared<ModelMetrics>(model_name);
      model_metrics_.emplace(model_name, std::move(metrics_item));
    }
  }

  ~TextDriverOneMetrics() = default;

  TextDriverOneMetrics(const TextDriverOneMetrics&) = default;

  TextDriverOneMetrics& operator=(const TextDriverOneMetrics&) = default;

  TextDriverOneMetrics(TextDriverOneMetrics&&) = default;
  TextDriverOneMetrics& operator=(TextDriverOneMetrics&&) = default;

  std::shared_ptr<ModelMetrics> get_model_metrics(const std::string model_name) const {
    return model_metrics_.find(model_name)->second;
  }

  std::string name_ = "text_driver_one";
  std::shared_ptr<clipper::metrics::Histogram> e2e_latency_;
  std::shared_ptr<clipper::metrics::DataList<long long>> e2e_latency_list_;
  std::shared_ptr<clipper::metrics::Meter> e2e_throughput_;
  std::shared_ptr<clipper::metrics::Counter> e2e_num_predictions_;

  std::unordered_map<std::string, std::shared_ptr<ModelMetrics>> model_metrics_;
};

void predict(FrontendRPCClient& client, ClientFeatureVector text_input,
             TextDriverOneMetrics metrics, std::atomic<int>& prediction_counter,
             std::unordered_map<std::string, std::ofstream&>& lineage_file_map,
             std::unordered_map<std::string, std::mutex&>& lineage_mutex_map) {
  auto start_time = std::chrono::system_clock::now();

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

  auto lstm_callback = [&client, metrics, completion_callback](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage,
      std::chrono::time_point<std::chrono::system_clock> request_start_time) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    } else {
      throw std::runtime_error("Received output of wrong datatype from LSTM query");
    }

    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - request_start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();

    auto lstm_metrics = metrics.get_model_metrics(LSTM_MODEL_APP_NAME);

    lstm_metrics->model_latency_->insert(static_cast<int64_t>(latency_micros));
    lstm_metrics->model_latency_list_->insert(static_cast<int64_t>(latency_micros));
    lstm_metrics->model_throughput_->mark(1);
    lstm_metrics->model_num_predictions_->increment(1);

    completion_callback();
  };

  auto nmt_callback = [&client, metrics, start_time, lstm_callback](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage,
      std::chrono::system_clock::time_point start_time) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    } else {
      throw std::runtime_error("Received output of wrong datatype from NMT query");
    }

    auto cur_time = std::chrono::system_clock::now();
    client.send_request(LSTM_MODEL_APP_NAME, output,
                        [cur_time, lstm_callback](ClientFeatureVector output,
                                                  std::shared_ptr<QueryLineage> lineage) {
                          lstm_callback(output, lineage, cur_time);
                        });
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();

    auto nmt_metrics = metrics.get_model_metrics(NMT_MODEL_APP_NAME);

    nmt_metrics->model_latency_->insert(static_cast<int64_t>(latency_micros));
    nmt_metrics->model_latency_list_->insert(static_cast<int64_t>(latency_micros));
    nmt_metrics->model_throughput_->mark(1);
    nmt_metrics->model_num_predictions_->increment(1);
  };

  auto lang_detect_callback = [&client, metrics, start_time, text_input, nmt_callback,
                               lstm_callback, completion_callback](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage) {
    if (output.type_ != DataType::Strings) {
      throw std::runtime_error("Received output of wrong datatype from LANG DETECT query");
    }

    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();

    std::string output_str =
        std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
    if (output_str == "TIMEOUT") {
      return;
    } else if (output_str == LANG_CLASSIFICATION_GERMAN) {
      client.send_request(NMT_MODEL_APP_NAME, text_input,
                          [cur_time, nmt_callback](ClientFeatureVector output,
                                                   std::shared_ptr<QueryLineage> lineage) {
                            nmt_callback(output, lineage, cur_time);
                          });
    } else if (output_str == LANG_CLASSIFICATION_ENGLISH) {
      client.send_request(LSTM_MODEL_APP_NAME, text_input,
                          [cur_time, lstm_callback](ClientFeatureVector output,
                                                    std::shared_ptr<QueryLineage> lineage) {
                            lstm_callback(output, lineage, cur_time);
                          });
    } else {
      // The language is not English or translateable
      completion_callback();
    }

    auto lang_detect_metrics = metrics.get_model_metrics(LANG_DETECT_MODEL_APP_NAME);

    lang_detect_metrics->model_latency_->insert(static_cast<int64_t>(latency_micros));
    lang_detect_metrics->model_latency_list_->insert(static_cast<int64_t>(latency_micros));
    lang_detect_metrics->model_throughput_->mark(1);
    lang_detect_metrics->model_num_predictions_->increment(1);
  };

  client.send_request(LANG_DETECT_MODEL_APP_NAME, text_input, lang_detect_callback);
}

std::vector<ClientFeatureVector> generate_text_inputs(size_t num_inputs,
                                                      size_t desired_input_length) {
  std::random_device rd;         // Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  std::vector<ClientFeatureVector> all_inputs;
  std::ifstream text_file(LANG_DETECT_WORKLOAD_RELATIVE_PATH);

  std::string line;
  while (std::getline(text_file, line)) {
    size_t input_size_bytes = line.size() * sizeof(char);
    size_t desired_input_length_bytes = desired_input_length * sizeof(char);
    size_t cp_unit_size = std::min(input_size_bytes, desired_input_length_bytes);

    std::shared_ptr<void> input_data(malloc(desired_input_length_bytes), free);
    char* raw_input_data = static_cast<char*>(input_data.get());
    size_t curr_cp_idx = 0;
    size_t curr_size = 0;
    while (curr_size < desired_input_length) {
      memcpy(raw_input_data + curr_cp_idx, line.data(), cp_unit_size);
      curr_size += cp_unit_size;
    }
    ClientFeatureVector input(input_data, desired_input_length, desired_input_length_bytes,
                              DataType::Strings);
    all_inputs.push_back(std::move(input));
  }

  std::vector<ClientFeatureVector> selected_inputs;
  selected_inputs.reserve(num_inputs);

  for (size_t i = 0; i < num_inputs; i++) {
    size_t idx = static_cast<size_t>(distribution(generator) * num_inputs);
    selected_inputs.push_back(all_inputs[idx]);
  }
  return selected_inputs;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("text_driver_one", "Text Driver One");
  // clang-format off
  options.add_options()
      ("target_throughput", "Mean throughput to target in qps",
       cxxopts::value<float>())
      ("request_distribution", "Distribution to sample request delay from. "
       "Can be 'constant', 'poisson', or 'batch'. 'batch' sends a single batch at a time.",
       cxxopts::value<std::string>())
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
  std::string distribution = options["request_distribution"].as<std::string>();
  if (!(distribution == "poisson" || distribution == "constant" || distribution == "batch")) {
    std::cerr << "Invalid distribution: " << distribution << std::endl;
    return 1;
  }

  // Request the system uptime so that a clock instance is created as
  // soon as the frontend starts
  clock::ClipperClock::get_clock().get_uptime();
  size_t num_inputs = 1000;
  size_t input_length = 20;
  std::vector<ClientFeatureVector> inputs = generate_text_inputs(num_inputs, input_length);
  TextDriverOneMetrics metrics;

  std::unordered_map<std::string, std::ofstream> lineage_file_map;
  std::unordered_map<std::string, std::ofstream&> lineage_file_map_refs;
  std::unordered_map<std::string, std::mutex> lineage_mutex_map;
  std::unordered_map<std::string, std::mutex&> lineage_mutex_map_refs;

  for (const std::string& model : MODEL_NAMES) {
    std::ofstream query_lineage_file;
    std::mutex query_file_mutex;
    query_lineage_file.open(options["log_file"].as<std::string>() + "-" + model +
                            "-query_lineage.txt");
    lineage_file_map.emplace(model, std::ofstream{});
    lineage_file_map_refs.emplace(model, lineage_file_map[model]);
    lineage_mutex_map.emplace(std::piecewise_construct, std::make_tuple(model), std::make_tuple());
    lineage_mutex_map_refs.emplace(model, lineage_mutex_map[model]);
  }

  auto predict_func = [metrics, &lineage_file_map_refs, &lineage_mutex_map_refs](
      FrontendRPCClient& client, ClientFeatureVector input, std::atomic<int>& prediction_counter) {
    predict(client, input, metrics, prediction_counter, lineage_file_map_refs,
            lineage_mutex_map_refs);
  };
  Driver driver(predict_func, std::move(inputs), options["target_throughput"].as<float>(),
                distribution, options["trial_length"].as<int>(), options["num_trials"].as<int>(),
                options["log_file"].as<std::string>(), options["clipper_address"].as<std::string>(),
                -1);
  std::cout << "Starting driver" << std::endl;
  driver.start();
  std::cout << "Driver completed" << std::endl;
  for (auto model : MODEL_NAMES) {
    lineage_file_map[model].close();
  }
  return 0;
}
