#include "predictor.hpp"
#include <httplib.h>
#include <fstream>
#include <iostream>
#include "zmq_client.hpp"

namespace zmq_client {

using namespace clipper;

constexpr int SEND_PORT = 4456;
constexpr int RECV_PORT = 4455;

Driver::Driver(std::function<void(FrontendRPCClient, ClientFeatureVector,
                                  std::atomic<int>&)>
                   predict_func,
               std::vector<ClientFeatureVector> inputs,
               int request_delay_micros, int trial_length, int num_trials,
               std::string log_file, std::string clipper_address)
    : predict_func_(predict_func),
      inputs_(inputs),
      request_delay_micros_(request_delay_micros),
      trial_length_(trial_length),
      num_trials_(num_trials),
      log_file_(log_file),
      client{2},
      done_(false),
      prediction_counter_(0),
      clipper_address_(clipper_address) {
  client.start(clipper_address, SEND_PORT, RECV_PORT);
}

void Driver::start() {
  auto monitor_thread = std::thread([this]() { monitor_results(); });

  while (!done) {
    for (ClientFeatureVector f : inputs_) {
      predict_func(client_, f, prediction_counter);
      std::this_thread::sleep_for(
          std::chrono::microseconds(request_delay_micros_));
    }
  }
  client_.stop();
  monitor_thread.join();
}

void Driver::monitor_results() {
  int num_completed_trials = 0;
  std::ofstream client_metrics_file;
  std::ofstream clipper_metrics_file;
  client_metrics_file.open(log_file_ + "-client_metrics.json");
  clipper_metrics_file.open(log_file_ + "-clipper_metrics.json");
  client_metrics_file << "[" << std::endl;
  clipper_metrics_file << "[" << std::endl;
  httplib::Client http_client(clipper_address, 1337);

  while (!done) {
    int current_count = prediction_counter_;
    if (current_count > trial_length) {
      prediction_counter_ = 0;
      num_completed_trials += 1;
      std::string metrics_report =
          // registry.report_metrics(false);
          registry.report_metrics(true);
      client_metrics_file << metrics_report;
      client_metrics_file << "," << std::endl;

      auto result = http_client.get("/metrics");
      if (result && result->status == 200) {
        clipper_metrics_file << res->body;
        clipper_metrics_file << "," << std::endl;
      }
    }

    if (num_completed_trials >= num_trials_) {
      done_ = true;
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  client_metrics_file << "]";
  client_metrics_file.close();
  clipper_metrics_file << "]";
  clipper_metrics_file.close();
}
}
