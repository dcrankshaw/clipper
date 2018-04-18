#include <time.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>

// #include "httplib.h"
// #define HTTP_IMPLEMENTATION
// #include "http.h"
#include "driver.hpp"
#include "zmq_client.hpp"

namespace zmq_client {

using namespace clipper;

constexpr int SEND_PORT = 4456;
constexpr int RECV_PORT = 4455;

Driver::Driver(std::function<void(std::unordered_map<std::string, FrontendRPCClient> &,
                                  ClientFeatureVector, std::atomic<int> &)>
                   predict_func,
               std::vector<ClientFeatureVector> inputs, float target_throughput,
               std::string distribution, int trial_length, int num_trials, std::string log_file,
               std::unordered_map<std::string, std::string> addresses, int batch_size,
               std::vector<float> delay_ms, bool collect_clipper_metrics)
    : predict_func_(predict_func),
      inputs_(inputs),
      target_throughput_(target_throughput),
      distribution_(distribution),
      trial_length_(trial_length),
      num_trials_(num_trials),
      log_file_(log_file),
      done_(false),
      prediction_counter_(0),
      addresses_(addresses),
      batch_size_(batch_size),
      delay_ms_(delay_ms),
      collect_clipper_metrics_{collect_clipper_metrics} {
  for (auto address : addresses_) {
    auto addr_find = clients_.find(address.first);
    if (addr_find == clients_.end()) {
      // clients_.emplace(std::string(address.first), 2);
      clients_.emplace(std::piecewise_construct, std::forward_as_tuple(address.first),
          std::forward_as_tuple(2));
      clients_[address.first].start(address.second, SEND_PORT, RECV_PORT);
      // FrontendRPCClient client(3);
      // client.start(address.second, SEND_PORT, RECV_PORT);
      // clients_.emplace(address.first, std::move(client));
    }
  }
  std::cout << "Starting " << std::to_string(clients_.size()) << " ZMQ clients." << std::endl;
}

void spin_sleep(long duration_micros) {
  auto start_time = std::chrono::system_clock::now();
  long cur_delay_micros = 0;
  while (cur_delay_micros < duration_micros) {
    auto cur_delay = std::chrono::system_clock::now() - start_time;
    cur_delay_micros = std::chrono::duration_cast<std::chrono::microseconds>(cur_delay).count();
  }
}

void Driver::start() {
  if (!(distribution_ == "poisson" || distribution_ == "constant" || distribution_ == "batch" ||
        distribution_ == "file")) {
    std::cerr << "Invalid distribution: " << distribution_ << std::endl;
    return;
  }
  auto monitor_thread = std::thread([this]() { monitor_results(); });
  if (distribution_ == "batch") {
    std::cout << "starting batch arrival process with batch size " << std::to_string(batch_size_)
              << std::endl;
    int cur_idx = 0;

    predict_func_(clients_, inputs_[cur_idx], prediction_counter_);
    cur_idx += 1;
    spin_sleep(1000 * 1000L);

    while (!done_) {
      // Get the current pred counter
      int cur_pred_counter = prediction_counter_;
      // Send a batch
      for (int j = 0; j < batch_size_; ++j) {
        predict_func_(clients_, inputs_[cur_idx], prediction_counter_);
        cur_idx += 1;
        if (cur_idx >= inputs_.size()) {
          cur_idx = 0;
        }
      }
      // spin until the batch completes
      while (prediction_counter_ < cur_pred_counter + batch_size_) {
      }
    }
  } else {
    while (!done_) {
      int delay_idx = 0;
      long seed = 1000;
      std::mt19937 gen(seed);
      std::exponential_distribution<> exp_dist(target_throughput_);
      long constant_request_delay_micros = std::lround(1.0 / target_throughput_ * 1000.0 * 1000.0);
      for (ClientFeatureVector f : inputs_) {
        if (done_) {
          break;
        }
        predict_func_(clients_, f, prediction_counter_);

        if (distribution_ == "poisson") {
          float delay_secs = exp_dist(gen);
          long delay_micros = lround(delay_secs * 1000.0 * 1000.0);
          spin_sleep(delay_micros);
        } else if (distribution_ == "constant") {
          spin_sleep(constant_request_delay_micros);
        } else if (distribution_ == "file") {
          float cur_sleep = delay_ms_[delay_idx];
          delay_idx += 1;
          if (delay_idx >= delay_ms_.size()) {
            delay_idx = 0;
          }
          long delay_micros = lround(cur_sleep * 1000.0);
          spin_sleep(delay_micros);
        }
      }
    }
  }
  for (auto &client : clients_) {
    client.second.stop();
  }
  monitor_thread.join();
}

void fetch_clipper_metrics(std::ofstream &clipper_metrics_file, std::string clipper_address) {
  std::string address = "http://" + clipper_address + ":" + std::to_string(1337) + "/metrics";
  std::string cmd_str = "curl -s -S " + address + " > curl_out.txt";
  std::system(cmd_str.c_str());
  std::ifstream curl_output("curl_out.txt");
  std::stringstream curl_str_buf;
  curl_output >> curl_str_buf.rdbuf();
  std::string curl_str = curl_str_buf.str();
  clipper_metrics_file << curl_str;
  clipper_metrics_file << "," << std::endl;
  curl_output.close();
}

void Driver::monitor_results() {
  int num_completed_trials = 0;
  std::ofstream client_metrics_file;
  client_metrics_file.open(log_file_ + "-client_metrics.json");
  client_metrics_file << "[" << std::endl;
  std::unordered_map<std::string, std::shared_ptr<std::ofstream>> clipper_metrics_map;
  for (auto &addr : addresses_) {
    clipper_metrics_map[addr.first] =
        std::make_shared<std::ofstream>(log_file_ + "-clipper_metrics_" + addr.first + ".json");
    *clipper_metrics_map[addr.first] << "{" << std::endl;
    // std::shared_ptr<std::ofstream> cur_metrics_file =
    //     std::make_shared<std::ofstream>(log_file_ + "-clipper_metrics_" + addr.first + ".json");
    // *cur_metrics_file << "{" << std::endl;
    // clipper_metrics_map.emplace(addr, cur_metrics_file);
  }

  metrics::MetricsRegistry &registry = metrics::MetricsRegistry::get_metrics();

  while (!done_) {
    int current_count = prediction_counter_;
    if (current_count > trial_length_ * (num_completed_trials + 1)) {
      num_completed_trials += 1;
      std::cout << "Trial " << std::to_string(num_completed_trials) << " completed" << std::endl;
      std::string metrics_report =
          // registry.report_metrics(false);
          registry.report_metrics(true);
      client_metrics_file << metrics_report;
      client_metrics_file << "," << std::endl;

      if (collect_clipper_metrics_) {
        for (auto &addr : clipper_metrics_map) {
          fetch_clipper_metrics(*addr.second, addr.first);
        }
      }
    }

    if (num_completed_trials >= num_trials_) {
      done_ = true;
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  // client_metrics_file << "]";
  client_metrics_file.close();
  // clipper_metrics_file << "]";
  if (collect_clipper_metrics_) {
    for (auto &addr : clipper_metrics_map) {
      addr.second->close();
    }
  }
  return;
}
}
