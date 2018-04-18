
#include <atomic> #include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <csdtlib>

#include <cxxopts.hpp>

#include <clipper/clock.hpp>
#include <clipper/metrics.hpp>

using namespace clipper;

namespace zmq_client {

class ClientMetrics {
 public:
  explicit ClientMetrics(std::vector<std::string>> model_names) {
          latencies_.emplace("e2e", clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            "e2e:prediction_latency", "microseconds", 32768));
          latency_lists_.emplace("e2e", clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
              "e2e:prediction_latencies", "microseconds"));
          throughputs_.emplace("e2e", clipper::metrics::MetricsRegistry::get_metrics().create_meter(
              "e2e:prediction_throughput"));
          num_predictions_.emplace("e2e", clipper::metrics::MetricsRegistry::get_metrics().create_counter(
                "e2e:num_predictions"));
          for (auto model : model_names) {
            latencies_.emplace(model, clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
            model + ":prediction_latency", "microseconds", 32768));
            latency_lists_.emplace(model, clipper::metrics::MetricsRegistry::get_metrics().create_data_list<long long>(
              model + ":prediction_latencies", "microseconds"));
            throughputs_.emplace(model, clipper::metrics::MetricsRegistry::get_metrics().create_meter(
                model + ":prediction_throughput"));
            num_predictions_.emplace(model, clipper::metrics::MetricsRegistry::get_metrics().create_counter(
                  model + ":num_predictions"));
          }
        }

  ~ClientMetrics() = default;

  ClientMetrics(const ClientMetrics&) = default;

  ClientMetrics& operator=(const ClientMetrics&) = default;

  ClientMetrics(ClientMetrics&&) = default;
  ClientMetrics& operator=(ClientMetrics&&) = default;

  std::unordered_map<std::string, std::shared_ptr<clipper::metrics::Histogram>> latencies_;
  std::unordered_map<std::string, std::shared_ptr<clipper::metrics::DataList<long long>>> latency_lists_;
  std::unordered_map<std::string, std::shared_ptr<clipper::metrics::Meter>> throughputs_;
  std::unordered_map<std::string, std::shared_ptr<clipper::metrics::Counter>> num_predictions_;

};
}
