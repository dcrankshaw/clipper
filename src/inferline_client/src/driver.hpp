#ifndef CLIPPER_PREDICTOR_HPP
#define CLIPPER_PREDICTOR_HPP

#include <clipper/metrics.hpp>
#include "zmq_client.hpp"

namespace zmq_client {

using namespace clipper;

void spin_sleep(int duration_micros);

class Driver {
 public:
  Driver(std::function<void(std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>>,
                            ClientFeatureVector, std::atomic<int>&)>
             predict_func,
         std::vector<ClientFeatureVector> inputs, float target_throughput, std::string distribution,
         int trial_length, int num_trials, std::string log_file,
         std::unordered_map<std::string, std::string> addresses, int batch_size,
         std::vector<float> delay_ms, bool collect_clipper_metrics);

  void start();

 private:
  void monitor_results();
  std::function<void(std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>>, ClientFeatureVector,
                     std::atomic<int>&)>
      predict_func_;
  std::vector<ClientFeatureVector> inputs_;
  float target_throughput_;
  std::string distribution_;
  int trial_length_;
  int num_trials_;
  std::string log_file_;
  // Map from model name to client
  std::unordered_map<std::string, std::shared_ptr<FrontendRPCClient>> clients_;
  std::atomic_bool done_;
  std::atomic<int> prediction_counter_;
  // Map from model name to address
  std::unordered_map<std::string, std::string> addresses_;
  int batch_size_;
  std::vector<float> delay_ms_;
  bool collect_clipper_metrics_;
};

}  // namespace zmq_client

#endif  // CLIPPER_PREDICTOR_HPP
