#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <cxxopts.hpp>

#include <clipper/clock.hpp>
#include <clipper/metrics.hpp>

#include "client_metrics.hpp"
#include "driver.hpp"
#include "zmq_client.hpp"

using namespace clipper;
using namespace zmq_client;

static const std::string RES50 = "res50";
static const std::string RES152 = "res152";
static const std::string ALEXNET = "alexnet";

// From https://stackoverflow.com/a/29710970/814642
double double_rand(double min, double max) {
  thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<double> distribution(min, max);
  return distribution(generator);
}

void predict(std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>> clients, ClientFeatureVector input,
             ClientMetrics &metrics, std::atomic<int>& prediction_counter,
             std::unordered_map<std::string, std::ofstream>& lineage_file_map,
             std::unordered_map<std::string, std::mutex>& lineage_mutex_map) {
  auto start_time = std::chrono::system_clock::now();

  auto completion_callback = [&metrics, &prediction_counter, start_time]() {
    auto cur_time = std::chrono::system_clock::now();
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.latencies_.find("e2e")->second->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_lists_.find("e2e")->second->insert(static_cast<int64_t>(latency_micros));
    metrics.throughputs_.find("e2e")->second->mark(1);
    metrics.num_predictions_.find("e2e")->second->increment(1);
    prediction_counter += 1;
  };

  auto res152_callback = [&metrics, completion_callback, &lineage_file_map,
                          &lineage_mutex_map](
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

    completion_callback();
    auto latency = cur_time - request_start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.latencies_.find(RES152)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_lists_.find(RES152)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.throughputs_.find(RES152)->second->mark(1);
    metrics.num_predictions_.find(RES152)->second->increment(1);

    lineage->add_timestamp(
        "driver::send",
        std::chrono::duration_cast<std::chrono::microseconds>(request_start_time.time_since_epoch())
            .count());

    lineage->add_timestamp(
        "driver::recv",
        std::chrono::duration_cast<std::chrono::microseconds>(cur_time.time_since_epoch()).count());
    std::unique_lock<std::mutex> lock(lineage_mutex_map[RES152]);
    auto& query_lineage_file = lineage_file_map[RES152];
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

  auto res50_callback = [input, &metrics, clients, res152_callback, completion_callback,
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

    double idk = double_rand(0.0, 1.0);
    if (idk > 0.4633) {
      clients[RES152]->send_request(
          RES152, input, [cur_time, res152_callback](ClientFeatureVector output,
                                                     std::shared_ptr<QueryLineage> lineage) {
            res152_callback(output, lineage, cur_time);
          });
    } else {
      completion_callback();
    }
    auto latency = cur_time - request_start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.latencies_.find(RES50)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_lists_.find(RES50)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.throughputs_.find(RES50)->second->mark(1);
    metrics.num_predictions_.find(RES50)->second->increment(1);

    lineage->add_timestamp(
        "driver::send",
        std::chrono::duration_cast<std::chrono::microseconds>(request_start_time.time_since_epoch())
            .count());

    lineage->add_timestamp(
        "driver::recv",
        std::chrono::duration_cast<std::chrono::microseconds>(cur_time.time_since_epoch()).count());
    std::unique_lock<std::mutex> lock(lineage_mutex_map[RES50]);
    auto& query_lineage_file = lineage_file_map[RES50];
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

  auto alexnet_callback = [input, &metrics, clients, res50_callback, completion_callback,
                           &lineage_file_map, &lineage_mutex_map, start_time](
      ClientFeatureVector output, std::shared_ptr<QueryLineage> lineage) {
    if (output.type_ == DataType::Strings) {
      std::string output_str =
          std::string(reinterpret_cast<char*>(output.get_data()), output.size_typed_);
      if (output_str == "TIMEOUT") {
        return;
      }
    }
    auto cur_time = std::chrono::system_clock::now();

    double idk = double_rand(0.0, 1.0);
    if (idk > 0.192) {
      clients[RES50]->send_request(
          RES50, input, [cur_time, res50_callback](ClientFeatureVector output,
                                                   std::shared_ptr<QueryLineage> lineage) {
            res50_callback(output, lineage, cur_time);
          });
    } else {
      completion_callback();
    }
    auto latency = cur_time - start_time;
    long latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
    metrics.latencies_.find(ALEXNET)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.latency_lists_.find(ALEXNET)->second->insert(static_cast<int64_t>(latency_micros));
    metrics.throughputs_.find(ALEXNET)->second->mark(1);
    metrics.num_predictions_.find(ALEXNET)->second->increment(1);

    lineage->add_timestamp(
        "driver::send",
        std::chrono::duration_cast<std::chrono::microseconds>(start_time.time_since_epoch())
            .count());

    lineage->add_timestamp(
        "driver::recv",
        std::chrono::duration_cast<std::chrono::microseconds>(cur_time.time_since_epoch()).count());
    std::unique_lock<std::mutex> lock(lineage_mutex_map[ALEXNET]);
    auto& query_lineage_file = lineage_file_map[ALEXNET];
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

  clients[ALEXNET]->send_request(ALEXNET, input, alexnet_callback);
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
  cxxopts::Options options("resnet_cascade", "Resnet Cascade");
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
      ("clipper_address_alexnet", "IP address or hostname of ZMQ frontend to use for alexnet",
       cxxopts::value<std::string>())
      ("clipper_address_res50", "IP address or hostname of ZMQ frontend to user for resnet50",
       cxxopts::value<std::string>())
      ("clipper_address_res152", "IP address or hostname of ZMQ frontend to user for resnet152",
       cxxopts::value<std::string>())
      ("request_delay_file", "Path to file containing a list of inter-arrival delays, one per line.",
       cxxopts::value<std::string>())
      ("get_clipper_metrics", "Collect Clipper metrics",
       cxxopts::value<bool>())
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

  std::vector<std::string> models = {ALEXNET, RES50, RES152};
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

  auto predict_func = [&metrics, &lineage_file_map, &lineage_mutex_map](
      std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>> clients, ClientFeatureVector input,
      std::atomic<int>& prediction_counter) {
    predict(clients, input, metrics, prediction_counter, lineage_file_map, lineage_mutex_map);
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
  addresses.emplace(ALEXNET, options["clipper_address_alexnet"].as<std::string>());
  addresses.emplace(RES50, options["clipper_address_res50"].as<std::string>());
  addresses.emplace(RES152, options["clipper_address_res152"].as<std::string>());
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
