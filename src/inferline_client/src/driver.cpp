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

Driver::Driver(
    std::function<void(FrontendRPCClient &, FrontendRPCClient &, ClientFeatureVector, std::atomic<int> &)> predict_func,
    std::vector<ClientFeatureVector> inputs, float target_throughput, std::string distribution,
    int trial_length, int num_trials, std::string log_file, std::string clipper_address_resnet, std::string clipper_address_inception,
    int batch_size, std::vector<float> delay_ms)
    : predict_func_(predict_func),
      inputs_(inputs),
      target_throughput_(target_throughput),
      distribution_(distribution),
      trial_length_(trial_length),
      num_trials_(num_trials),
      log_file_(log_file),
      resnet_client_{4},
      inception_client_{4},
      done_(false),
      prediction_counter_(0),
      clipper_address_resnet_(clipper_address_resnet),
      clipper_address_inception_(clipper_address_inception),
      batch_size_(batch_size),
      delay_ms_(delay_ms) {
  resnet_client_.start(clipper_address_resnet, SEND_PORT, RECV_PORT);
  if (clipper_address_resnet == clipper_address_inception) {
    std::cout << "Using same client for resnet and inception" << std::endl;
    different_clients_ = false;
  } else {
    inception_client_.start(clipper_address_inception, SEND_PORT, RECV_PORT);
    different_clients_ = true;
  }
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

    // Send a query to flush the system
    if (different_clients_) {
      predict_func_(resnet_client_, inception_client_, inputs_[cur_idx], prediction_counter_);
    } else {
      predict_func_(resnet_client_, resnet_client_, inputs_[cur_idx], prediction_counter_);
    }
    cur_idx += 1;
    spin_sleep(1000 * 1000L);

    while (!done_) {
      // Get the current pred counter
      int cur_pred_counter = prediction_counter_;
      // Send a batch
      for (int j = 0; j < batch_size_; ++j) {
        if (different_clients_) {
          predict_func_(resnet_client_, inception_client_, inputs_[cur_idx], prediction_counter_);
        } else {
          predict_func_(resnet_client_, resnet_client_, inputs_[cur_idx], prediction_counter_);
        }
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
        if (different_clients_) {
          predict_func_(resnet_client_, inception_client_, f, prediction_counter_);
        } else {
          predict_func_(resnet_client_, resnet_client_, f, prediction_counter_);
        }

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
  resnet_client_.stop();
  if (different_clients_) {
    inception_client_.stop();
  }
  monitor_thread.join();
}

void Driver::monitor_results() {
  int num_completed_trials = 0;
  std::ofstream client_metrics_file;
  // std::ofstream clipper_metrics_file;
  client_metrics_file.open(log_file_ + "-client_metrics.json");
  // clipper_metrics_file.open(log_file_ + "-clipper_metrics.json");
  client_metrics_file << "[" << std::endl;
  // clipper_metrics_file << "[" << std::endl;

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
      // std::string address = "http://" + clipper_address_ + ":" + std::to_string(1337) + "/metrics";
      // std::string cmd_str = "curl -s -S " + address + " > curl_out.txt";
      // std::system(cmd_str.c_str());
      // std::ifstream curl_output("curl_out.txt");
      // std::stringstream curl_str_buf;
      // curl_output >> curl_str_buf.rdbuf();
      // std::string curl_str = curl_str_buf.str();
      // clipper_metrics_file << curl_str;
      // clipper_metrics_file << "," << std::endl;
      // curl_output.close();
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
  // clipper_metrics_file.close();
  return;
}
}
