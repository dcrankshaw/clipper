#ifndef CLIPPER_PREDICTOR_HPP
#define CLIPPER_PREDICTOR_HPP

#include <clipper/metrics.hpp>
#include "zmq_client.hpp"

namespace zmq_client {

using namespace clipper;

void spin_sleep(int duration_micros);

class Driver {
 public:
  Driver(std::function<void(FrontendRPCClient&, FrontendRPCClient&, ClientFeatureVector,
                            std::atomic<int>&)>
             predict_func,
         std::vector<ClientFeatureVector> inputs, float target_throughput,
         std::string distribution, int trial_length, int num_trials,
         std::string log_file, std::string clipper_address_resnet,
         std::string clipper_address_inception, int batch_size,
         std::vector<float> delay_ms, bool collect_clipper_metrics);

  void start();

 private:
  void monitor_results();
  std::function<void(FrontendRPCClient&, FrontendRPCClient&, ClientFeatureVector,
                     std::atomic<int>&)>
      predict_func_;
  std::vector<ClientFeatureVector> inputs_;
  float target_throughput_;
  std::string distribution_;
  int trial_length_;
  int num_trials_;
  std::string log_file_;
  FrontendRPCClient resnet_client_;
  FrontendRPCClient inception_client_;
  bool different_clients_;
  std::atomic_bool done_;
  std::atomic<int> prediction_counter_;
  std::string clipper_address_resnet_;
  std::string clipper_address_inception_;
  int batch_size_;
  std::vector<float> delay_ms_;
  bool get_clipper_metrics_;
};

}  // namespace zmq_client

#endif  // CLIPPER_PREDICTOR_HPP
