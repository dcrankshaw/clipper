#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <algorithm>
#include <limits>

#include <cxxopts.hpp>

#include <clipper/clock.hpp>
#include <clipper/metrics.hpp>

#include "client_metrics.hpp"
#include "driver.hpp"
#include "zmq_client.hpp"

using namespace clipper;
using namespace zmq_client;

static const std::string TF_RESNET = "tf-resnet-feats";
static const std::string TF_KERNEL_SVM = "tf-kernel-svm";

static const std::string INCEPTION_FEATS = "inception";
static const std::string TF_LOG_REG = "tf-log-reg";

static const std::string E2E = "e2e";

void predict(std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>> clients, ClientFeatureVector input,
             ClientMetrics &metrics, std::atomic<int>& prediction_counter,
             std::unordered_map<std::string, std::ofstream>& lineage_file_map,
             std::unordered_map<std::string, std::mutex>& lineage_mutex_map,
             int latency_budget_micros) {
  auto start_time = std::chrono::system_clock::now();
  size_t resnet_input_length = 224 * 224 * 3;
  ClientFeatureVector resnet_input(input.data_, resnet_input_length,
                                   resnet_input_length * sizeof(float), DataType::Floats);

  std::shared_ptr<std::atomic_int> branches_completed = std::make_shared<std::atomic_int>(0);
  std::shared_ptr<std::atomic_bool> expired = std::make_shared<std::atomic_bool>(false);
  auto completion_callback = [&metrics, &prediction_counter, start_time, expired]() {
    if (*expired) {
      std::cout << "Received expired query" << std::endl;
      long latency_micros = std::numeric_limits<int>::max();
      metrics.latency_lists_.find("e2e")->second->insert(static_cast<int64_t>(latency_micros));
      metrics.latencies_.find("e2e")->second->insert(static_cast<int64_t>(latency_micros));
    } else {
      auto cur_time = std::chrono::system_clock::now();
      auto latency = cur_time - start_time;
      long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
      auto search = metrics.latencies_.find("e2e");
      if (search != metrics.latencies_.end()) {
        metrics.latencies_.find("e2e")->second->insert(static_cast<int64_t>(latency_micros));
      } else {
        std::cout << "PRINTING ELEMENTS: ";
        for (auto l : metrics.latencies_) {
          std::cout << l.first << ", ";
        }
        std::cout << std::endl;
        throw std::runtime_error("couldn't find e2e latencies");
      }
      metrics.latency_lists_.find("e2e")->second->insert(static_cast<int64_t>(latency_micros));
      metrics.throughputs_.find("e2e")->second->mark(1);
      metrics.num_predictions_.find("e2e")->second->increment(1);
    }
    prediction_counter += 1;
  };

  auto ksvm_callback = [&metrics, branches_completed, completion_callback, &lineage_file_map,
                        &lineage_mutex_map, expired](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage,
      std::chrono::time_point<std::chrono::system_clock> request_start_time) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        // NOTE: This ignores lineage
        long latency_micros = std::numeric_limits<int>::max();
        expired->store(true);
        metrics.latencies_.find(TF_KERNEL_SVM)->second->insert(static_cast<int64_t>(latency_micros));
        metrics.latency_lists_.find(TF_KERNEL_SVM)
            ->second->insert(static_cast<int64_t>(latency_micros));
        int num_branches_completed = branches_completed->fetch_add(1);
        if (num_branches_completed == 1) {
          completion_callback();
        }
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - request_start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.latencies_.find(TF_KERNEL_SVM)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_lists_.find(TF_KERNEL_SVM)
        ->second->insert(static_cast<int64_t>(latency_micros));
    metrics.throughputs_.find(TF_KERNEL_SVM)->second->mark(1);
    metrics.num_predictions_.find(TF_KERNEL_SVM)->second->increment(1);

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

  auto log_reg_callback = [&metrics, branches_completed, &completion_callback, &lineage_file_map,
                           &lineage_mutex_map, expired](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage,
      std::chrono::time_point<std::chrono::system_clock> request_start_time) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        // NOTE: This ignores lineage
        long latency_micros = std::numeric_limits<int>::max();
        expired->store(true);
        metrics.latencies_.find(TF_LOG_REG)->second->insert(static_cast<int64_t>(latency_micros));
        metrics.latency_lists_.find(TF_LOG_REG)
            ->second->insert(static_cast<int64_t>(latency_micros));
        int num_branches_completed = branches_completed->fetch_add(1);
        if (num_branches_completed == 1) {
          completion_callback();
        }
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - request_start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.latencies_.find(TF_LOG_REG)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_lists_.find(TF_LOG_REG)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.throughputs_.find(TF_LOG_REG)->second->mark(1);
    metrics.num_predictions_.find(TF_LOG_REG)->second->increment(1);

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

  auto inception_callback = [clients, &metrics, start_time, log_reg_callback, &lineage_file_map,
                             &lineage_mutex_map, latency_budget_micros, completion_callback,
                             expired, branches_completed](ClientFeatureVector output,
                                                 std::shared_ptr<QueryLineage> lineage) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        // NOTE: This ignores lineage
        long latency_micros = std::numeric_limits<int>::max();
        expired->store(true);
        metrics.latencies_.find(INCEPTION_FEATS)->second->insert(static_cast<int64_t>(latency_micros));
        metrics.latency_lists_.find(INCEPTION_FEATS)
            ->second->insert(static_cast<int64_t>(latency_micros));
        int num_branches_completed = branches_completed->fetch_add(1);
        if (num_branches_completed == 1) {
          completion_callback();
        }
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    int new_latency_budget = std::max(0L, latency_budget_micros - latency_micros);
    clients[INCEPTION_FEATS]->send_request(
        TF_LOG_REG, output, new_latency_budget, [cur_time, log_reg_callback](ClientFeatureVector output,
                                                         std::shared_ptr<QueryLineage> lineage) {
          log_reg_callback(output, lineage, cur_time);
        });
    metrics.latencies_.find(INCEPTION_FEATS)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_lists_.find(INCEPTION_FEATS)
        ->second->insert(static_cast<int64_t>(latency_micros));
    metrics.throughputs_.find(INCEPTION_FEATS)->second->mark(1);
    metrics.num_predictions_.find(INCEPTION_FEATS)->second->increment(1);

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

  auto resnet_callback = [clients, &metrics, start_time, ksvm_callback, &lineage_file_map,
                          &lineage_mutex_map, latency_budget_micros, completion_callback,
                          expired, branches_completed](ClientFeatureVector output,
                                              std::shared_ptr<QueryLineage> lineage) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        // NOTE: This ignores lineage
        long latency_micros = std::numeric_limits<int>::max();
        expired->store(true);
        metrics.latencies_.find(INCEPTION_FEATS)->second->insert(static_cast<int64_t>(latency_micros));
        metrics.latency_lists_.find(INCEPTION_FEATS)
            ->second->insert(static_cast<int64_t>(latency_micros));
        int num_branches_completed = branches_completed->fetch_add(1);
        if (num_branches_completed == 1) {
          completion_callback();
        }
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    int new_latency_budget = std::max(0L, latency_budget_micros - latency_micros);
    clients[TF_RESNET]->send_request(
        TF_KERNEL_SVM, output, new_latency_budget, [cur_time, ksvm_callback](ClientFeatureVector output,
                                                         std::shared_ptr<QueryLineage> lineage) {
          ksvm_callback(output, lineage, cur_time);
        });
    metrics.latencies_.find(TF_RESNET)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_lists_.find(TF_RESNET)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.throughputs_.find(TF_RESNET)->second->mark(1);
    metrics.num_predictions_.find(TF_RESNET)->second->increment(1);

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
  metrics.throughputs_.find("ingest")->second->mark(1);

  clients[INCEPTION_FEATS]->send_request(INCEPTION_FEATS, input, latency_budget_micros, inception_callback);
  clients[TF_RESNET]->send_request(TF_RESNET, resnet_input, latency_budget_micros, resnet_callback);
}

std::vector<ClientFeatureVector> generate_float_inputs(int input_length) {
  int num_points = 1000;
  std::random_device rd;         // Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> distribution(0.1, 1.0);
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
      ("clipper_address_resnet", "IP address or hostname of ZMQ frontend to use for the resnet branch",
       cxxopts::value<std::string>())
      ("clipper_address_inception", "IP address or hostname of ZMQ frontend to user for the inception branch",
       cxxopts::value<std::string>())
      ("request_delay_file", "Path to file containing a list of inter-arrival delays, one per line.",
       cxxopts::value<std::string>())
      ("get_clipper_metrics", "Collect Clipper metrics",
       cxxopts::value<bool>())
      ("latency_budget_micros", "end to end latency budget for query in microseconds",
       cxxopts::value<int>()) // TODO TODO TODO
       ;
  // clang-format on
  options.parse(argc, argv);
  std::string distribution = options["request_distribution"].as<std::string>();
  if (!(distribution == "poisson" || distribution == "constant" || distribution == "batch" ||
        distribution == "file")) {
    std::cerr << "Invalid distribution: " << distribution << std::endl;
    return 1;
  }

  // Request the system uptime so that a clock instance is created as
  // soon as the frontend starts
  clock::ClipperClock::get_clock().get_uptime();
  std::vector<ClientFeatureVector> inputs = generate_float_inputs(299 * 299 * 3);
  std::vector<std::string> models = {TF_RESNET, INCEPTION_FEATS, TF_KERNEL_SVM, TF_LOG_REG};
  ClientMetrics metrics(models);
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

  int latency_budget_micros = options["latency_budget_micros"].as<int>();
  auto predict_func = [&metrics, &lineage_file_map, &lineage_mutex_map, latency_budget_micros](
      std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>> clients, ClientFeatureVector input,
      std::atomic<int>& prediction_counter) {
    predict(clients, input, metrics, prediction_counter, lineage_file_map, lineage_mutex_map, latency_budget_micros);
  };
  std::vector<float> delays_ms;
  if (distribution == "file") {
    std::ifstream delay_file_stream(options["request_delay_file"].as<std::string>());
    std::string line;
    while (std::getline(delay_file_stream, line)) {
      delays_ms.push_back(std::stof(line));
    }
    delay_file_stream.close();
    std::cout << "Loaded delays file: " << std::to_string(delays_ms.size()) << " lines"
              << std::endl;
  }
  std::unordered_map<std::string, std::string> addresses;
  addresses.emplace(TF_RESNET, options["clipper_address_resnet"].as<std::string>());
  addresses.emplace(INCEPTION_FEATS, options["clipper_address_inception"].as<std::string>());
  Driver driver(predict_func, std::move(inputs), options["target_throughput"].as<float>(),
                distribution, options["trial_length"].as<int>(), options["num_trials"].as<int>(),
                options["log_file"].as<std::string>(), addresses, -1, delays_ms,
                options["get_clipper_metrics"].as<bool>());
  std::cout << "Starting driver" << std::endl;
  driver.start();
  std::cout << "Driver completed" << std::endl;
  for (auto model : models) {
    lineage_file_map[model].close();
  }
  return 0;
}
